# IS567FP/models/multinomial_naive_bayes_bow_baseline.py
import logging
import numpy as np
from typing import Optional, Tuple, Dict, Any # Added Dict, Any
import pandas as pd
from scipy.sparse import hstack, csr_matrix # Import sparse matrix tools

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Inherit from the new base class
# Make sure necessary helpers are imported or available in the base class
from .baseline_base import TextBaselineModel, TextFeatureExtractorBase, clean_dataset, _evaluate_model_performance
# Import DatabaseHandler if load_raw_text_data needs it explicitly passed
# from utils.database import DatabaseHandler # Uncomment if needed

logger = logging.getLogger(__name__)

class BoWExtractor(TextFeatureExtractorBase):
    """Specialized Bag-of-Words extractor using CountVectorizer."""
    def __init__(self, args: Optional[object] = None, alpha: float = 1.0, max_features: Optional[int] = 10000, ngram_range: Tuple[int, int] = (1, 1)):
        if args is not None:
            alpha = getattr(args, 'alpha', alpha)
            max_features = getattr(args, 'max_features', max_features)
        extractor = BoWExtractor(max_features=max_features, ngram_range=ngram_range)
        model_instance = MultinomialNB(alpha=alpha)
        super().__init__(extractor, model_instance)
        self.alpha = alpha
        self.args = args


    def transform(self, data: pd.DataFrame, text_col1: str = 'premise_text', text_col2: str = 'hypothesis_text') -> csr_matrix:
         """Transforms text into sparse BoW features (concatenated)."""
         if not self.is_fitted:
              raise RuntimeError("BoW vectorizer must be fitted before transforming.")
         if text_col1 not in data.columns or text_col2 not in data.columns:
              raise ValueError(f"DataFrame must contain '{text_col1}' and '{text_col2}' columns.")

         logger.debug(f"Transforming {len(data)} samples with BoW...")
         premise_bow = self.vectorizer.transform(data[text_col1].fillna(''))
         hypothesis_bow = self.vectorizer.transform(data[text_col2].fillna(''))

         # Concatenate sparse matrices horizontally
         features = hstack([premise_bow, hypothesis_bow], format='csr')
         logger.debug(f"BoW transformation complete. Feature shape: {features.shape}")
         return features


class MultinomialNaiveBayesBaseline(TextBaselineModel):
    """Baseline NLI model using Multinomial Naive Bayes with Bag-of-Words features."""

    # Added args to __init__ to accept the args object from the trainer
    def __init__(self, args: object, alpha: float = 1.0, max_features: Optional[int] = 10000, ngram_range: Tuple[int, int] = (1, 1)):
        # Potentially extract needed params from args if not passed directly
        alpha = getattr(args, 'alpha', alpha)
        max_features = getattr(args, 'max_features', max_features)
        # ngram_range might need parsing if passed via args

        extractor = BoWExtractor(max_features=max_features, ngram_range=ngram_range)
        model_instance = MultinomialNB(alpha=alpha)
        super().__init__(extractor, model_instance)
        self.alpha = alpha # Store specific param
        self.args = args # Store args if needed later, e.g., for db_handler path

        # Instantiate db_handler here if needed by load_raw_text_data
        # from utils.database import DatabaseHandler # Import locally
        # self.db_handler = DatabaseHandler()


    # def extract_features(self, data: pd.DataFrame) -> csr_matrix: # Return sparse matrix
    #     """Extracts BoW features using the assigned extractor."""
    #     if not self.extractor.is_fitted:
    #         raise RuntimeError("BoW extractor must be fitted or loaded first.")
    #     return self.extractor.transform(data)

    def extract_features(self, data: pd.DataFrame) -> Any:
        """
        Extracts features using the extractor.
        Handles both raw-text inputs (premise/hypothesis) and precomputed features.
        """
        if self.extractor is None:
            raise RuntimeError("No extractor found for this model.")

        # Case 1: Raw text data is available (standard path for baseline-3)
        if 'premise_text' in data.columns and 'hypothesis_text' in data.columns:
            if not self.extractor.is_fitted:
                raise RuntimeError("Extractor not fitted. Ensure the model is trained or properly loaded.")
            return self.extractor.transform(data)

        # Case 2: Precomputed feature columns are already present (e.g., from Parquet)
        feature_cols = [col for col in data.columns if col.startswith("feature_") or col.startswith("bow_")]
        if feature_cols:
            logger.info(f"Detected {len(feature_cols)} precomputed feature columns.")
            X = data[feature_cols].copy()
            return X.to_numpy()

        # If neither applies, raise error
        raise ValueError("Input DataFrame lacks both raw text and recognized feature columns.")


    # --- ADDED evaluate METHOD ---
    def evaluate(self, dataset_name: str, split: str, suffix: str) -> Optional[Dict[str, Any]]:
        """
        Evaluates the trained model on a given dataset split.

        Args:
            dataset_name (str): Name of the dataset (e.g., 'SNLI').
            split (str): Data split to evaluate on (e.g., 'test', 'validation').
            suffix (str): Suffix indicating sample size ('full' or 'sampleXXX').

        Returns:
            dict: Dictionary containing evaluation metrics (e.g., accuracy, f1), or None if evaluation fails.
        """
        logger.info(f"Evaluating {self.__class__.__name__} on {dataset_name}/{split}/{suffix}...")

        # 1. Load Raw Data - Assuming load_raw_text_data is accessible
        #    May need to pass db_handler if the static method requires it
        try:
            # If db_handler is needed: data_df = self.load_raw_text_data(dataset_name, split, suffix, db_handler=self.db_handler)
            # If not needed:
            # NOTE: Check if load_raw_text_data needs db_handler passed or if it creates its own
            # Assuming it doesn't need it passed explicitly for now, based on baseline_base.py usage patterns
            # You might need to adjust this based on your DatabaseHandler setup.
            from utils.database import DatabaseHandler # Temporary import if needed here
            db_handler = DatabaseHandler() # Create instance if needed
            data_df = self.load_raw_text_data(dataset_name, split, suffix, db_handler=db_handler) # Pass db_handler
        except Exception as e:
            logger.error(f"Failed to load raw data for evaluation: {e}", exc_info=True)
            return None

        if data_df is None or data_df.empty:
            logger.error(f"No data loaded for evaluation for {dataset_name}/{split}/{suffix}.")
            return None
        logger.info(f"Loaded {len(data_df)} samples for evaluation.")

        # 2. Clean Data and Prepare Labels
        cleaned_data = clean_dataset(data_df)
        if cleaned_data is None:
            logger.error("Data became empty or invalid after cleaning.")
            return None
        df_cleaned, y_true = cleaned_data
        logger.info(f"Data cleaned. {len(df_cleaned)} samples remaining.")

        if len(df_cleaned) == 0:
             logger.warning("No valid samples left after cleaning for evaluation.")
             return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'eval_time': 0.0}


        # 3. Extract Features
        try:
            X_eval = self.extract_features(df_cleaned)
            logger.info(f"Features extracted for evaluation. Shape: {X_eval.shape}")
        except Exception as e:
            logger.error(f"Failed to extract features during evaluation: {e}", exc_info=True)
            return None

        # 4. Evaluate (using the helper function from baseline_base)
        # Ensure the model is trained (should be if loaded correctly)
        if not self.is_trained:
             logger.error("Model is not trained. Cannot evaluate.")
             # Or raise an error: raise RuntimeError("Model is not trained.")
             return None

        try:
            eval_time, metrics = _evaluate_model_performance(self, X_eval, y_true)
            metrics['eval_time'] = eval_time # Add timing info
            logger.info(f"Evaluation complete. Metrics: {metrics}")
            return metrics
        except Exception as e:
            logger.error(f"Error during model performance evaluation: {e}", exc_info=True)
            return None
    # --- END ADDED METHOD ---

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict labels for a batch of samples and return a DataFrame with predictions.
        """
        LABEL_MAP_REVERSE: Dict[int, str] = {0: 'entailment', 1: 'contradiction', 2: 'neutral'}

        if not self.is_trained:
            raise RuntimeError("Model is not trained. Cannot perform batch prediction.")

        # Extract features
        X = self.extract_features(df)

        # Predict
        predictions = self.model.predict(X)

        # Prepare output
        result_df = pd.DataFrame({
            'pair_id': df['pair_id'].values if 'pair_id' in df.columns else range(len(df)),
            'gold_label': df['label'].values if 'label' in df.columns else ['unknown'] * len(predictions),
            'predicted_label': [LABEL_MAP_REVERSE.get(pred, 'unknown') for pred in predictions]
        })

        return result_df


    # train, predict, save, load are inherited from TextBaselineModel
    # load_raw_text_data static method is now in TextBaselineModel (but called within evaluate)

# Removed MultinomialNaiveBayesTrainer class