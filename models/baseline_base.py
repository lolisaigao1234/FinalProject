# models/baseline_base.py
import logging
import os
from abc import abstractmethod

import joblib
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any # Added Dict, Any
import time

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Use the existing NLIModel ABC from common.py as the ultimate base
from utils.common import NLIModel
from utils.database import DatabaseHandler
from config import MODELS_DIR

logger = logging.getLogger(__name__)

# Common Helper Functions (Moved from various trainers/model files)

def prepare_labels(labels, label_map=None):
    """Convert string labels to integers with a consistent mapping"""
    if label_map is None:
        label_map = {'entailment': 0, 'contradiction': 1, 'neutral': 2}

    if labels.dtype == object:
        # Handle potential NaNs introduced before conversion
        labels = labels.fillna('unknown') # Or dropna() depending on desired behavior
        return np.array([label_map.get(label, -1) for label in labels])
    elif pd.api.types.is_numeric_dtype(labels):
         # Ensure numeric labels are integers and handle potential floats/NaNs
         return pd.to_numeric(labels, errors='coerce').fillna(-1).astype(int).values
    return labels # Assume it's already in the correct numeric format


def get_label_column(df: pd.DataFrame) -> Tuple[str, np.ndarray]:
    """Extract label column name and values from dataframe"""
    if 'gold_label' in df.columns:
        return 'gold_label', df['gold_label'].values
    elif 'label' in df.columns:
        return 'label', df['label'].values
    else:
        raise ValueError("No label column (gold_label or label) found in data")


def clean_dataset(df: pd.DataFrame) -> Optional[Tuple[pd.DataFrame, np.ndarray]]:
    """Cleans dataset by handling labels and removing invalid entries. Returns None if result is empty."""
    if df is None or df.empty:
        logger.warning("Input DataFrame for cleaning is None or empty.")
        return None

    try:
        label_col, labels = get_label_column(df)
    except ValueError as e:
        logger.error(f"Error getting label column: {e}")
        return None # Cannot proceed without labels

    # Convert labels to integers and handle unknowns (-1)
    int_labels = prepare_labels(labels)

    # Filter out invalid labels BEFORE trying to index the DataFrame
    valid_mask = int_labels != -1
    num_invalid = len(valid_mask) - np.sum(valid_mask)

    if num_invalid > 0:
        logger.info(f"Removing {num_invalid} rows with invalid/missing labels.")
        # Apply mask to both DataFrame and labels *if* filtering is needed
        if not np.all(valid_mask):
            df_cleaned = df.loc[valid_mask].reset_index(drop=True)
            int_labels_cleaned = int_labels[valid_mask]
        else: # Should not happen if num_invalid > 0, but safe check
             df_cleaned = df
             int_labels_cleaned = int_labels
    else:
        # No invalid labels found
        df_cleaned = df
        int_labels_cleaned = int_labels

    if df_cleaned.empty:
         logger.warning("DataFrame is empty after cleaning labels.")
         return None

    return df_cleaned, int_labels_cleaned


def _evaluate_model_performance(model: NLIModel, X_val: np.ndarray, y_val: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """Evaluate model performance on validation/test data. (Internal helper)"""
    # Ensure y_val is not empty
    if y_val is None or len(y_val) == 0:
        logger.warning("Cannot evaluate model: y_val is empty or None.")
        return 0.0, {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    logger.info(f"Evaluating model on {X_val.shape[0]} samples...")
    start_time = time.time()
    try:
        y_pred = model.predict(X_val)
    except Exception as e:
        logger.error(f"Error during model prediction: {e}", exc_info=True)
        return 0.0, {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    eval_time = time.time() - start_time
    logger.info(f"Prediction finished in {eval_time:.2f} seconds.")

    # Ensure y_pred has the same length as y_val
    if len(y_pred) != len(y_val):
         logger.error(f"Prediction length mismatch: y_pred ({len(y_pred)}) vs y_val ({len(y_val)})")
         return eval_time, {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}


    accuracy = accuracy_score(y_val, y_pred)
    # Use zero_division=0 to handle cases with no predicted/true samples for a class
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_val, y_pred, average='weighted', zero_division=0
    )

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    logger.info(f"Evaluation Metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    return eval_time, metrics


class TextFeatureExtractorBase:
    """Base class for TF-IDF and BoW extractors."""
    def __init__(self, vectorizer_class, **kwargs):
        self.vectorizer = vectorizer_class(**kwargs)
        self.is_fitted = False

    def fit(self, data: pd.DataFrame, text_col1: str = 'premise_text', text_col2: str = 'hypothesis_text'):
        """Fits the vectorizer on the combined text."""
        if text_col1 not in data.columns or text_col2 not in data.columns:
            raise ValueError(f"DataFrame must contain '{text_col1}' and '{text_col2}' columns.")

        # Combine texts for fitting vocabulary
        combined_texts = data[text_col1].fillna('') + " " + data[text_col2].fillna('')
        logger.info(f"Fitting {self.vectorizer.__class__.__name__} on {len(combined_texts)} combined texts...")
        self.vectorizer.fit(combined_texts)
        self.is_fitted = True
        vocab_size = len(self.vectorizer.vocabulary_) if hasattr(self.vectorizer, 'vocabulary_') else 'N/A'
        logger.info(f"{self.vectorizer.__class__.__name__} fitted. Vocabulary size: {vocab_size}")

    def transform(self, data: pd.DataFrame, text_col1: str = 'premise_text', text_col2: str = 'hypothesis_text') -> np.ndarray:
        """Transforms text into features (implementation specific to subclass)."""
        raise NotImplementedError

    def save(self, filepath: str):
        """Saves the fitted vectorizer."""
        joblib.dump(self.vectorizer, filepath)
        logger.info(f"{self.vectorizer.__class__.__name__} vectorizer saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'TextFeatureExtractorBase':
        """Loads a fitted vectorizer."""
        vectorizer = joblib.load(filepath)
        # We need to know the original vectorizer class to instantiate correctly.
        # This might require storing the class name or passing it.
        # For now, assume the loading logic knows the class or handle outside.
        instance = cls.__new__(cls) # Create instance without calling __init__
        instance.vectorizer = vectorizer
        instance.is_fitted = True
        logger.info(f"{vectorizer.__class__.__name__} vectorizer loaded from {filepath}")
        return instance

# Abstract Base Class for Text-Based NLI Baselines (TF-IDF, BoW)
class TextBaselineModel(NLIModel):
    """
    Abstract Base Class for NLI models working directly on text (TF-IDF, BoW).
    Requires subclasses to implement model training and feature extraction details.
    """
    def __init__(self, extractor: TextFeatureExtractorBase, model_instance: Any):
        self.extractor = extractor
        self.model = model_instance
        self.is_trained = False

    @abstractmethod
    def extract_features(self, data: pd.DataFrame) -> Any: # Return type can be sparse or dense
        """Extracts features using the assigned extractor."""
        pass

    def train(self, X: Any, y: np.ndarray) -> None:
        """Trains the internal sklearn model."""
        logger.info(f"Training {self.model.__class__.__name__} with {X.shape[0]} samples...")
        self.model.fit(X, y)
        self.is_trained = True
        logger.info(f"{self.model.__class__.__name__} training complete.")

    def predict(self, X: Any) -> np.ndarray:
        """Makes predictions using the trained model."""
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet.")
        logger.debug(f"Predicting on {X.shape[0]} samples...")
        predictions = self.model.predict(X)
        logger.debug("Prediction finished.")
        return predictions

    def save(self, directory: str, model_name: str) -> None:
        """Saves the trained model and the feature extractor."""
        os.makedirs(directory, exist_ok=True)
        model_path = os.path.join(directory, f"{model_name}_model.joblib")
        extractor_path = os.path.join(directory, f"{model_name}_extractor.joblib") # Save extractor

        joblib.dump(self.model, model_path)
        joblib.dump(self.extractor, extractor_path) # Save the extractor object
        logger.info(f"{self.model.__class__.__name__} model saved to {model_path}")
        logger.info(f"{self.extractor.vectorizer.__class__.__name__} extractor saved to {extractor_path}")

    @classmethod
    def load(cls, directory: str, model_name: str) -> 'TextBaselineModel':
        """Loads the model and extractor."""
        model_path = os.path.join(directory, f"{model_name}_model.joblib")
        extractor_path = os.path.join(directory, f"{model_name}_extractor.joblib")

        if not os.path.exists(model_path) or not os.path.exists(extractor_path):
            raise FileNotFoundError(f"Model or extractor file not found in directory: {directory}")

        loaded_model = joblib.load(model_path)
        loaded_extractor = joblib.load(extractor_path) # Load the extractor object

        # Create instance - requires knowing the specific subclass,
        # or modifying load to be instance method, or passing subclass type
        # This static method approach is problematic for varying subclasses.
        # We might need to move load logic to the trainer or specific model files.
        # For now, let's assume this is called from the specific subclass context
        # or the trainer handles instantiation based on model_name.
        instance = cls.__new__(cls) # Create instance without calling __init__
        instance.model = loaded_model
        instance.extractor = loaded_extractor
        instance.is_trained = True
        logger.info(f"{cls.__name__} loaded from {directory}")
        return instance

    @staticmethod
    def load_raw_text_data(dataset_name: str, split: str, suffix: str, db_handler: DatabaseHandler) -> Optional[pd.DataFrame]:
        """
        Loads raw text data by merging intermediate 'pairs' and 'sentences' files.
        (Copied from logistic_tf_idf_baseline.py - needs import)
        """
        from data.data_loader import DatasetLoader # [cite: 1] - Import locally if needed
        logger.info(f"Attempting to load intermediate raw text data: {dataset_name}/{split}/{suffix}")
        pairs_table = f"pairs_{suffix}"
        sentences_table = f"sentences_{suffix}"
        merged_df = None

        try:
            pairs_df = db_handler.load_dataframe(dataset_name, split, pairs_table)
            sentences_df = db_handler.load_dataframe(dataset_name, split, sentences_table)

            if not pairs_df.empty and not sentences_df.empty:
                logger.info("Merging intermediate pairs and sentences data...")
                sentences_premise = sentences_df[['id', 'text']].rename(columns={'text': 'premise_text', 'id': 'p_id'})
                sentences_hypothesis = sentences_df[['id', 'text']].rename(columns={'text': 'hypothesis_text', 'id': 'h_id'})
                pairs_essential = pairs_df[['id', 'premise_id', 'hypothesis_id', 'label']].rename(columns={'id': 'pair_id'})

                merged_df = pd.merge(pairs_essential, sentences_premise, left_on='premise_id', right_on='p_id', how='left')
                merged_df = pd.merge(merged_df, sentences_hypothesis, left_on='hypothesis_id', right_on='h_id', how='left')

                final_cols = ['pair_id', 'premise_text', 'hypothesis_text', 'label']
                if not all(col in merged_df.columns for col in final_cols):
                    missing = [col for col in final_cols if col not in merged_df.columns]
                    logger.error(f"Columns missing after merge: {missing}.")
                    merged_df = None
                else:
                    merged_df = merged_df[final_cols].fillna('') # Fill NaNs
                    logger.info(f"Successfully loaded and merged intermediate data. Shape: {merged_df.shape}")
            else:
                 logger.warning(f"Intermediate data ({pairs_table} or {sentences_table}) not found or empty. Falling back.")
                 merged_df = None

        except Exception as e:
            logger.error(f"Error loading/merging intermediate data: {e}. Falling back.")
            merged_df = None

        if merged_df is None:
            logger.warning("Executing fallback: Loading original raw data using DatasetLoader.")
            try:
                loader = DatasetLoader(db_handler)
                raw_df = loader.load_dataset(dataset_name, split=split)
                if raw_df.empty: return None

                # Standardize columns (assuming loader does basic conversion)
                raw_df = raw_df.rename(columns={'id': 'pair_id'}, errors='ignore')
                required_cols = ['pair_id', 'premise_text', 'hypothesis_text', 'label']
                if all(col in raw_df.columns for col in required_cols):
                     raw_df['premise_text'] = raw_df['premise_text'].fillna('')
                     raw_df['hypothesis_text'] = raw_df['hypothesis_text'].fillna('')
                     return raw_df[required_cols]
                else:
                     logger.error(f"Fallback raw data missing required columns: {[c for c in required_cols if c not in raw_df.columns]}")
                     return None
            except Exception as e:
                logger.error(f"Error during fallback data loading: {e}", exc_info=True)
                return None

        return merged_df