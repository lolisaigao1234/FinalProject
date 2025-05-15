# models/logistic_tf_idf_baseline.py
import logging
import numpy as np
from typing import Optional, Tuple, Dict, Any  # Added Dict, Any
import pandas as pd
import joblib
import os  # Added os for path operations
import time  # Added time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Inherit from the new base class
# Ensure TextBaselineModel exists and has expected structure (like inheriting NLIModel)
from .baseline_base import TextBaselineModel, TextFeatureExtractorBase
# Import evaluation helpers and data cleaning
from .baseline_base import clean_dataset, _evaluate_model_performance, SimpleParquetLoader  # Import SimpleParquetLoader

logger = logging.getLogger(__name__)


class TFIDFExtractor(TextFeatureExtractorBase):
    """Specialized TF-IDF extractor."""

    def __init__(self, max_features: Optional[int] = 10000, ngram_range: Tuple[int, int] = (1, 2), **kwargs):
        # Pass TfidfVectorizer and relevant args to base
        super().__init__(
            TfidfVectorizer,
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            **kwargs  # Pass any other TfidfVectorizer args
        )

    def transform(self, data: pd.DataFrame, text_col1: str = 'premise_text',
                  text_col2: str = 'hypothesis_text') -> np.ndarray:
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
        features = hstack([premise_tfidf, hypothesis_tfidf])  # Keep sparse

        logger.debug(f"TF-IDF transformation complete. Feature shape: {features.shape}")
        return features  # Return sparse matrix

    # fit and fit_transform are inherited from TextFeatureExtractorBase


class LogisticTFIDFBaseline(TextBaselineModel):
    """Baseline NLI model using Logistic Regression with TF-IDF features."""
    MODEL_NAME = "Logistic_TF-IDF_Baseline"
    EXTRACTOR_CLS = TFIDFExtractor  # Needed for TextBaselineModel.load

    def __init__(self, args: Optional[object] = None, C: float = 1.0, max_iter: int = 1000,
                 max_features: Optional[int] = 10000, ngram_range: Tuple[int, int] = (1, 2), **kwargs):
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
        model_instance = LogisticRegression(C=C, max_iter=max_iter, solver='liblinear',
                                            random_state=42)  # Pass random_state if gotten from args
        super().__init__(extractor, model_instance)
        # Store specific params if needed later
        self.C = C
        self.max_iter = max_iter
        self.loader = SimpleParquetLoader()  # Instantiate loader

    # --- ADD THIS METHOD ---
    def extract_features(self, data: pd.DataFrame, fit: bool = False) -> Any:  # Return type might be sparse
        """
        Extracts TF-IDF features using the assigned extractor.
        Implementation of the abstract method from TextBaselineModel.
        Delegates to the extractor's fit_transform or transform method.
        """
        logger.debug(f"Extracting features within {self.MODEL_NAME} (fit={fit}) using {type(self.extractor).__name__}")
        if fit:
            # --- MODIFICATION START ---
            # Call fit first, then transform
            logger.info(f"Fitting extractor ({type(self.extractor).__name__})...")
            self.extractor.fit(data)
            logger.info("Transforming data after fitting...")
            features = self.extractor.transform(data)
            # --- MODIFICATION END ---
            return features
        else:
            # Use transform from the specific TFIDFExtractor
            # This only transforms the data using the already fitted vectorizer
            if not self.extractor.is_fitted:
                # Ensure the extractor was fitted during training before calling transform
                raise RuntimeError("Extractor must be fitted (usually during training) before transforming new data.")
            logger.info("Transforming data using pre-fitted extractor...")
            features = self.extractor.transform(data)
            return features

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
            df_eval_raw = self.loader.load_data(dataset_name, split, suffix)  # Use a temporary variable
        except FileNotFoundError as e:
            logger.error(f"Evaluation data not found: {e}")
            return None
        except ValueError as e:  # Catch missing columns error from loader
            logger.error(f"Error loading evaluation data: {e}")
            return None

        # --- FIX START ---
        # Clean the data and handle the tuple return value
        cleaned_data_result = clean_dataset(df_eval_raw)
        if cleaned_data_result is None:
            logger.error(f"Evaluation data for {split} is invalid or None after cleaning.")
            return None

        # Unpack the tuple
        df_eval_cleaned, y_true = cleaned_data_result

        # Check if the *cleaned DataFrame* is empty
        if df_eval_cleaned.empty:
            logger.error(f"Evaluation data for {split} is empty after cleaning labels.")
            return None
        # --- FIX END ---

        # --- FIX: Update evaluation call ---
        logger.info("Extracting features for evaluation...")
        try:
            X_eval = self.extract_features(df_eval_cleaned, fit=False)  # Extract features from cleaned data
        except Exception as e:
            logger.error(f"Error extracting features during evaluation: {e}", exc_info=True)
            return None

        logger.info("Calculating evaluation metrics...")
        eval_start_time = time.time()
        # Pass the internal sklearn model instance, extracted features, and true labels
        _, eval_metrics = _evaluate_model_performance(self.model, X_eval, y_true)  # Call the helper correctly
        eval_calc_time = time.time() - eval_start_time
        # --- END FIX ---

        # Combine timings if needed, or just use the metric calculation time
        eval_metrics['eval_time'] = eval_time  # Keep prediction time, or combine?
        eval_metrics['eval_split'] = split  # Add split info

        logger.info(f"Evaluation metrics calculation finished in {eval_calc_time:.2f}s")
        logger.info(f"Evaluation metrics for {split}: {eval_metrics}")
        return eval_metrics

    #  predict, save, load are inherited from TextBaselineModel

    # Ensure TextBaselineModel's implementations are suitable or override them here if needed.
    # Particularly, check if TextBaselineModel.train uses a loader correctly. If not, override train here.
