# IS567FP/models/logistic_tf_idf_baseline.py
import logging
import numpy as np
from typing import Optional, Tuple, Dict, Any # Added Dict, Any
import pandas as pd
import joblib
import os # Added os for path operations
import time # Added time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Inherit from the new base class
# Ensure TextBaselineModel exists and has expected structure (like inheriting NLIModel)
from .baseline_base import TextBaselineModel, TextFeatureExtractorBase
# Import evaluation helpers and data cleaning
from .baseline_base import _evaluate_model_performance, clean_dataset

logger = logging.getLogger(__name__)

# Define a simple loader for this context if not using DB handler directly
# Example: Simple Parquet Loader (adjust path logic as needed)
# --- REUSE OR ADAPT SimpleParquetLoader FROM PREVIOUS FIX ---
from config import DATA_DIR # Import DATA_DIR if used by loader
class SimpleParquetLoader:
    def load_data(self, dataset_name, split, suffix):
        # Construct path - adjust based on where parquet files are stored
        # This might need to align with how FeatureExtractor saves features, or use a dedicated path
        cache_dir = os.path.join(DATA_DIR, 'cache', 'parquet') # Example cache dir
        # --- !!! IMPORTANT: Adjust filename pattern !!! ---
        # Does TFIDF use a specific feature type name? Or just suffix?
        # Example assuming a generic name based on suffix:
        filename = f"{dataset_name}_{split}_{suffix}.parquet"
        filepath = os.path.join(cache_dir, filename)
        logger.info(f"Attempting to load parquet data from: {filepath}")
        if not os.path.exists(filepath):
             # Try alternative common naming convention if first fails
             alt_filename = f"{dataset_name}_{split}_features_{suffix}.parquet"
             alt_filepath = os.path.join(cache_dir, alt_filename)
             if not os.path.exists(alt_filepath):
                  raise FileNotFoundError(f"Could not find parquet data at {filepath} or {alt_filepath}")
             else:
                  filepath = alt_filepath


        df = pd.read_parquet(filepath)
        logger.info(f"Loaded {len(df)} rows from {filepath}")
         # Ensure required columns are present after loading
         # Adjust required columns based on TFIDFExtractor needs (e.g., premise_text)
        req_cols = ['premise_text', 'hypothesis_text', 'label'] # Example, adjust if needed
        if not all(col in df.columns for col in req_cols):
            logger.error(f"Loaded parquet file {filepath} is missing required columns ({req_cols}). Available: {df.columns.tolist()}")
            raise ValueError(f"Missing required columns in {filepath}")
        return df
# --- END SimpleParquetLoader ---


class TFIDFExtractor(TextFeatureExtractorBase):
    """Specialized TF-IDF extractor."""
    def __init__(self, max_features: Optional[int] = 10000, ngram_range: Tuple[int, int] = (1, 2), **kwargs):
        # Pass TfidfVectorizer and relevant args to base
        super().__init__(
            TfidfVectorizer,
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            **kwargs # Pass any other TfidfVectorizer args
        )

    def transform(self, data: pd.DataFrame, text_col1: str = 'premise_text', text_col2: str = 'hypothesis_text') -> np.ndarray:
        """Transforms premise and hypothesis text into TF-IDF features (concatenated)."""
        if not self.is_fitted:
            raise RuntimeError("TF-IDF vectorizer must be fitted before transforming.")
        # Use provided column names
        if text_col1 not in data.columns or text_col2 not in data.columns:
            raise ValueError(f"DataFrame must contain '{text_col1}' and '{text_col2}' columns.")

        logger.debug(f"Transforming {len(data)} samples with TF-IDF...")
        # Ensure input is string and handle NaNs
        premise_tfidf = self.vectorizer.transform(data[text_col1].astype(str).fillna(''))
        hypothesis_tfidf = self.vectorizer.transform(data[text_col2].astype(str).fillna(''))


        # Concatenate features - returning sparse might be better for Logistic Regression memory-wise
        # features = np.concatenate([premise_tfidf.toarray(), hypothesis_tfidf.toarray()], axis=1) # Dense
        from scipy.sparse import hstack
        features = hstack([premise_tfidf, hypothesis_tfidf]) # Keep sparse

        logger.debug(f"TF-IDF transformation complete. Feature shape: {features.shape}")
        return features # Return sparse matrix

    # fit and fit_transform are inherited from TextFeatureExtractorBase


class LogisticTFIDFBaseline(TextBaselineModel):
    """Baseline NLI model using Logistic Regression with TF-IDF features."""
    MODEL_NAME = "Logistic_TF-IDF_Baseline"
    EXTRACTOR_CLS = TFIDFExtractor # Needed for TextBaselineModel.load

    def __init__(self, args: Optional[object] = None, C: float = 1.0, max_iter: int = 1000, max_features: Optional[int] = 10000, ngram_range: Tuple[int, int] = (1, 2), **kwargs):
        # Handle args object if passed
        if args:
            # Use tfidf_max_features specific arg if available, else fallback
            max_features = getattr(args, 'tfidf_max_features', getattr(args, 'max_features', max_features))
            C = getattr(args, 'C', C)
            max_iter = getattr(args, 'max_iter', max_iter)
            # ngram_range is not typically passed via args, but could be added if needed
            # random_state = getattr(args, 'random_state', 42) # Get random state if needed

        extractor = TFIDFExtractor(max_features=max_features, ngram_range=ngram_range)
        # Increase max_iter if convergence issues arise
        model_instance = LogisticRegression(C=C, max_iter=max_iter, solver='liblinear', random_state=42) # Pass random_state if gotten from args
        super().__init__(extractor, model_instance)
        # Store specific params if needed later
        self.C = C
        self.max_iter = max_iter
        self.loader = SimpleParquetLoader() # Instantiate loader

    # --- ADD THIS METHOD ---
    def extract_features(self, data: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """
        Extracts TF-IDF features using the assigned extractor.
        Implementation of the abstract method from TextBaselineModel.
        Delegates to the extractor's fit_transform or transform method.
        """
        logger.debug(f"Extracting features within {self.MODEL_NAME} (fit={fit}) using {type(self.extractor).__name__}")
        if fit:
            # Use fit_transform from the base TextFeatureExtractorBase (or TFIDFExtractor if overridden)
            # This fits the vectorizer and transforms the data
            return self.extractor.fit_transform(data)
        else:
            # Use transform from the specific TFIDFExtractor
            # This only transforms the data using the already fitted vectorizer
            if not self.extractor.is_fitted:
                # Ensure the extractor was fitted during training before calling transform
                raise RuntimeError("Extractor must be fitted (usually during training) before transforming new data.")
            return self.extractor.transform(data)

    # --- END OF ADDED METHOD ---

    # --- ADD TRAIN METHOD ---
        # --- 添加/修改 train 方法 ---
    def train(self, train_dataset: str, train_split: str, train_suffix: str,
              val_dataset: Optional[str] = None, val_split: Optional[str] = None, val_suffix: Optional[str] = None,
              **kwargs) -> Optional[Dict[str, Any]]:
        """
        Loads data, extracts TF-IDF features, and trains the Logistic Regression model.
        Overrides the base train method to handle data loading.
        """
        logger.info(
            f"Starting training process for {self.MODEL_NAME} on {train_dataset}/{train_split} ({train_suffix})")
        train_start_time = time.time()

        # 1. Load Training Data using self.loader
        try:
            df_train = self.loader.load_data(train_dataset, train_split, train_suffix)
        except FileNotFoundError as e:
            logger.error(f"Training data not found: {e}")
            return {'error': 'Training data not found'}
        except ValueError as e:  # Catch missing columns error from loader
            logger.error(f"Error loading training data: {e}")
            return {'error': f'Error loading training data: {e}'}

        # 2. Clean Training Data
        cleaned_data = clean_dataset(df_train)
        if cleaned_data is None:
            logger.error(f"Training data for {train_split} is empty or invalid after cleaning.")
            return {'error': 'Training data invalid after cleaning'}
        df_train_cleaned, y_train = cleaned_data
        if df_train_cleaned.empty:
            logger.error(f"Training data for {train_split} is empty after cleaning labels.")
            return {'error': 'Training data empty after cleaning'}

        logger.info(f"Training data loaded and cleaned: {df_train_cleaned.shape[0]} samples.")

        # 3. Extract Features (fit=True for training)
        logger.info("Extracting features for training...")
        feature_extraction_start = time.time()
        try:
            # Use the extract_features method (which should call the extractor's fit_transform)
            X_train = self.extract_features(df_train_cleaned, fit=True)
        except Exception as e:
            logger.error(f"Error during feature extraction: {e}", exc_info=True)
            return {'error': f'Feature extraction failed: {e}'}
        feature_extraction_time = time.time() - feature_extraction_start
        logger.info(
            f"Feature extraction finished in {feature_extraction_time:.2f}s. Feature shape: {X_train.shape}")

        # 4. Train the Model (using the inherited internal model.fit)
        logger.info(f"Fitting internal model ({self.model.__class__.__name__})...")
        model_fit_start = time.time()
        try:
            # Call the *actual* sklearn model's fit method, not the base class train
            self.model.fit(X_train, y_train)
            self.is_trained = True  # Mark as trained
        except Exception as e:
            logger.error(f"Error during model fitting: {e}", exc_info=True)
            return {'error': f'Model fitting failed: {e}'}
        model_fit_time = time.time() - model_fit_start
        logger.info(f"Internal model fitting finished in {model_fit_time:.2f}s.")

        train_time = time.time() - train_start_time
        logger.info(f"Total training process for {self.MODEL_NAME} completed in {train_time:.2f}s")

        results = {
            'train_time': train_time,
            'feature_extraction_time': feature_extraction_time,
            'model_fit_time': model_fit_time,
            'num_train_samples': X_train.shape[0],
            'num_features': X_train.shape[1]
        }

        # (Optional) Evaluate on validation set if provided
        if val_dataset and val_split and val_suffix:
            logger.info(f"Evaluating on validation set: {val_dataset}/{val_split} ({val_suffix})")
            # Call the existing evaluate method
            val_metrics = self.evaluate(val_dataset, val_split, val_suffix)
            if val_metrics:
                results['validation_metrics'] = val_metrics

        return results  # Return training results/metrics

    # --- ADD EVALUATE METHOD ---
    def evaluate(self, dataset_name: str, split: str, suffix: str) -> Optional[Dict[str, Any]]:
        """
        Evaluates the trained model on a given dataset split.
        Implementation of the abstract evaluate method.
        """
        if not self.is_trained:
            logger.error(f"Cannot evaluate {self.MODEL_NAME}. Model is not trained.")
            return None

        logger.info(f"Evaluating {self.MODEL_NAME} on {dataset_name}/{split} ({suffix})")
        # Load test data using self.loader
        try:
            df_eval = self.loader.load_data(dataset_name, split, suffix)
        except FileNotFoundError as e:
            logger.error(f"Evaluation data not found: {e}")
            return None
        except ValueError as e: # Catch missing columns error from loader
            logger.error(f"Error loading evaluation data: {e}")
            return None

        df_eval = clean_dataset(df_eval)
        if df_eval.empty:
            logger.error(f"Evaluation data for {split} is empty after cleaning.")
            return None

        # Extract features (fit=False) - This is done inside predict
        y_true = df_eval['label'].values

        # Make predictions
        try:
            start_time = time.time()
            # Predict calls extract_features(fit=False) internally via TextBaselineModel's predict
            y_pred = self.predict(df_eval)
            eval_time = time.time() - start_time
        except Exception as e:
             logger.error(f"Error during prediction in evaluation: {e}", exc_info=True)
             return None

        # Calculate metrics using the helper from baseline_base
        eval_metrics = _evaluate_model_performance(y_true, y_pred, f"{self.MODEL_NAME} {split.capitalize()}")
        eval_metrics['eval_time'] = eval_time
        eval_metrics['eval_split'] = split # Add split info

        logger.info(f"Evaluation metrics for {split}: {eval_metrics}")
        return eval_metrics

    #  predict, save, load are inherited from TextBaselineModel

    # Ensure TextBaselineModel's implementations are suitable or override them here if needed.
    # Particularly, check if TextBaselineModel.train uses a loader correctly. If not, override train here.