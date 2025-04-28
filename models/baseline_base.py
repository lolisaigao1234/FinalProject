# Modify: IS567FP/models/baseline_base.py
import logging
import os
from abc import abstractmethod

import joblib
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any, List # Added Dict, Any, List
import time

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Use the existing NLIModel ABC from common.py as the ultimate base
from utils.common import NLIModel
from utils.database import DatabaseHandler
from config import MODELS_DIR

logger = logging.getLogger(__name__)

# Common Helper Functions (Moved from various trainers/model files)

# <<< ADDED _handle_nan_values function >>>
def _handle_nan_values(df: pd.DataFrame, context: str = "processing") -> pd.DataFrame:
    """
    Checks for and handles NaN values in numeric columns of a DataFrame, typically by filling with 0.

    Args:
        df (pd.DataFrame): The input DataFrame.
        context (str): A string describing the context (e.g., 'training', 'features') for logging.

    Returns:
        pd.DataFrame: The DataFrame with NaNs handled in numeric columns.
    """
    if df is None or df.empty:
        logger.warning(f"DataFrame is None or empty in _handle_nan_values ({context}).")
        return df # Return as is if empty or None

    # Select only numeric columns for NaN check/fill
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if not numeric_cols:
        logger.debug(f"No numeric columns found to check for NaNs ({context}).")
        return df # Return original if no numeric columns

    nan_check = df[numeric_cols].isnull().sum()
    total_nans = nan_check.sum()

    if total_nans > 0:
        logger.warning(f"Found {total_nans} NaN values in numeric columns during {context}. Filling with 0.")
        logger.debug(f"NaN counts per numeric column:\n{nan_check[nan_check > 0]}")
        # Fill NaNs only in numeric columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        # Verify NaNs are filled
        if df[numeric_cols].isnull().sum().sum() > 0:
            logger.error(f"NaN values still present after attempting fillna(0) during {context}!")
    else:
        logger.debug(f"No NaN values found in numeric columns during {context}.")

    return df
# <<< END of added function >>>


def prepare_labels(labels, label_map=None):
    """Convert string labels to integers with a consistent mapping"""
    if label_map is None:
        label_map = {'entailment': 0, 'contradiction': 1, 'neutral': 2}

    # Ensure labels is a pandas Series for consistent handling
    if not isinstance(labels, pd.Series):
        labels = pd.Series(labels)


    if labels.dtype == object:
        # Handle potential NaNs introduced before conversion
        labels = labels.fillna('unknown') # Or dropna() depending on desired behavior
        return np.array([label_map.get(label, -1) for label in labels])
    elif pd.api.types.is_numeric_dtype(labels):
        # Ensure numeric labels are integers and handle potential floats/NaNs
        # Use pd.to_numeric which handles various numeric types and converts to float first
        # Then fillna and convert to int
        return pd.to_numeric(labels, errors='coerce').fillna(-1).astype(int).values
    # Fallback if it's already numeric but not float/int (e.g., bool)
    try:
        return labels.astype(int).values
    except Exception as e:
        logger.error(f"Could not convert labels to integer array: {e}. Returning original.")
        return labels # Return original array if conversion fails


def get_label_column(df: pd.DataFrame) -> Tuple[str, pd.Series]: # Return Series for consistency
    """Extract label column name and values from dataframe"""
    if 'gold_label' in df.columns:
        return 'gold_label', df['gold_label']
    elif 'label' in df.columns:
        return 'label', df['label']
    else:
        raise ValueError("No label column (gold_label or label) found in data")


def clean_dataset(df: pd.DataFrame) -> Optional[Tuple[pd.DataFrame, np.ndarray]]:
    """Cleans dataset by handling labels and removing invalid entries. Returns None if result is empty."""
    if df is None or df.empty:
        logger.warning("Input DataFrame for cleaning is None or empty.")
        return None

    try:
        label_col, labels_series = get_label_column(df)
    except ValueError as e:
        logger.error(f"Error getting label column: {e}")
        return None # Cannot proceed without labels

    # Convert labels to integers and handle unknowns (-1)
    int_labels = prepare_labels(labels_series) # Pass the Series

    # Filter out invalid labels BEFORE trying to index the DataFrame
    valid_mask = int_labels != -1
    num_invalid = len(valid_mask) - np.sum(valid_mask)

    if num_invalid > 0:
        logger.info(f"Removing {num_invalid} rows with invalid/missing labels.")
        # Apply mask to both DataFrame and labels *if* filtering is needed
        if not np.all(valid_mask):
            # Ensure mask aligns with DataFrame index if it's non-standard
            if not df.index.equals(pd.RangeIndex(start=0, stop=len(df), step=1)):
                logger.debug("DataFrame index is non-standard. Aligning mask.")
                valid_mask_aligned = pd.Series(valid_mask, index=df.index)
                df_cleaned = df.loc[valid_mask_aligned] # Use boolean Series on original index
            else:
                df_cleaned = df[valid_mask].reset_index(drop=True) # Direct boolean mask ok
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
    # Check if X_val has samples
    if X_val is None or X_val.shape[0] == 0:
        logger.warning("Cannot evaluate model: X_val is empty or None.")
        return 0.0, {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    # Check for consistent number of samples
    if X_val.shape[0] != len(y_val):
         logger.error(f"Feature/Label mismatch for evaluation: X_val has {X_val.shape[0]} samples, y_val has {len(y_val)} labels.")
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
    if y_pred is None or len(y_pred) != len(y_val):
        pred_len = len(y_pred) if y_pred is not None else 'None'
        logger.error(f"Prediction length mismatch: y_pred ({pred_len}) vs y_val ({len(y_val)})")
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
        # Save the extractor instance, not just the vectorizer
        joblib.dump(self.extractor, extractor_path)
        logger.info(f"{self.model.__class__.__name__} model saved to {model_path}")
        logger.info(f"Feature extractor ({self.extractor.__class__.__name__}) saved to {extractor_path}")


    @classmethod
    def load(cls, directory: str, model_name: str) -> 'TextBaselineModel':
        """Loads the model and extractor."""
        model_path = os.path.join(directory, f"{model_name}_model.joblib")
        extractor_path = os.path.join(directory, f"{model_name}_extractor.joblib")

        if not os.path.exists(model_path) or not os.path.exists(extractor_path):
            raise FileNotFoundError(f"Model or extractor file not found in directory: {directory} for base name {model_name}")

        loaded_model = joblib.load(model_path)
        loaded_extractor = joblib.load(extractor_path) # Load the extractor object

        # Create instance using the loaded components
        # Assumes the subclass calling load() passes the correct extractor and model
        # This is typically handled by the BaselineTrainer which knows the class type
        instance = cls(extractor=loaded_extractor, model_instance=loaded_model)
        instance.is_trained = True # Assume loaded model is trained
        logger.info(f"{cls.__name__} loaded from {directory} using base name {model_name}")
        return instance

    @staticmethod
    def load_raw_text_data(dataset_name: str, split: str, suffix: str, db_handler: DatabaseHandler) -> Optional[pd.DataFrame]:
        """
        Loads raw text data by merging intermediate 'pairs' and 'sentences' files.
        Falls back to original data loader if intermediates are not found.
        """
        from data.data_loader import DatasetLoader # Import locally if needed
        logger.info(f"Attempting to load intermediate raw text data: {dataset_name}/{split}/{suffix}")
        pairs_table = f"pairs_{suffix}"
        sentences_table = f"sentences_{suffix}"
        merged_df = None

        try:
            pairs_df = db_handler.load_dataframe(dataset_name, split, pairs_table)
            sentences_df = db_handler.load_dataframe(dataset_name, split, sentences_table)

            if not pairs_df.empty and not sentences_df.empty:
                logger.info("Merging intermediate pairs and sentences data...")
                # Ensure IDs are suitable for merging (e.g., string or int)
                # If they were saved as objects, convert them back
                pairs_df['id'] = pairs_df['id'].astype(str)
                pairs_df['premise_id'] = pairs_df['premise_id'].astype(str)
                pairs_df['hypothesis_id'] = pairs_df['hypothesis_id'].astype(str)
                sentences_df['id'] = sentences_df['id'].astype(str)


                sentences_premise = sentences_df[['id', 'text']].rename(columns={'text': 'premise_text', 'id': 'p_id'})
                sentences_hypothesis = sentences_df[['id', 'text']].rename(columns={'text': 'hypothesis_text', 'id': 'h_id'})
                pairs_essential = pairs_df[['id', 'premise_id', 'hypothesis_id', 'label']].rename(columns={'id': 'pair_id'})

                # Perform merges
                merged_df_temp1 = pd.merge(pairs_essential, sentences_premise, left_on='premise_id', right_on='p_id', how='left')
                merged_df = pd.merge(merged_df_temp1, sentences_hypothesis, left_on='hypothesis_id', right_on='h_id', how='left')


                # Select and rename final columns, handle potential missing text after merge
                final_cols = ['pair_id', 'premise_text', 'hypothesis_text', 'label']
                missing_cols = [col for col in final_cols if col not in merged_df.columns]
                if missing_cols:
                    logger.error(f"Columns missing after merge: {missing_cols}. Merge failed.")
                    merged_df = None
                else:
                    merged_df = merged_df[final_cols]
                    # Check for NaNs specifically in text columns after merge (important!)
                    premise_nan_count = merged_df['premise_text'].isnull().sum()
                    hyp_nan_count = merged_df['hypothesis_text'].isnull().sum()
                    if premise_nan_count > 0 or hyp_nan_count > 0:
                         logger.warning(f"Found {premise_nan_count} NaN premises and {hyp_nan_count} NaN hypotheses after merge. Filling with empty string.")
                         merged_df['premise_text'] = merged_df['premise_text'].fillna('')
                         merged_df['hypothesis_text'] = merged_df['hypothesis_text'].fillna('')

                    # Drop rows where label is still NaN (should be handled by clean_dataset later, but good check)
                    label_nan_count = merged_df['label'].isnull().sum()
                    if label_nan_count > 0:
                         logger.warning(f"Found {label_nan_count} NaN labels after merge. Dropping these rows.")
                         merged_df.dropna(subset=['label'], inplace=True)

                    logger.info(f"Successfully loaded and merged intermediate data. Shape after merge: {merged_df.shape}")

            else:
                logger.warning(f"Intermediate data ({pairs_table} or {sentences_table}) not found or empty. Falling back to original data loader.")
                merged_df = None

        except Exception as e:
            logger.error(f"Error loading/merging intermediate data: {e}. Falling back.", exc_info=True)
            merged_df = None

        # Fallback mechanism
        if merged_df is None or merged_df.empty:
            logger.warning("Executing fallback: Loading original raw data using DatasetLoader.")
            try:
                loader = DatasetLoader(db_handler)
                # Request raw text columns explicitly
                raw_df = loader.load_dataset(dataset_name, split=split, columns=['premise', 'hypothesis', 'label'])
                if raw_df.empty:
                    logger.error("Fallback data loading returned empty DataFrame.")
                    return None

                # Rename standard columns expected by TextBaseline models
                raw_df = raw_df.rename(columns={'premise': 'premise_text', 'hypothesis': 'hypothesis_text'}, errors='ignore')
                # Add a pair_id if missing (using index)
                if 'pair_id' not in raw_df.columns:
                     raw_df.reset_index(inplace=True)
                     raw_df = raw_df.rename(columns={'index': 'pair_id'}, errors='ignore') # Rename index if it becomes a column

                # Final check for required columns after fallback
                required_cols = ['pair_id', 'premise_text', 'hypothesis_text', 'label']
                if all(col in raw_df.columns for col in required_cols):
                    raw_df['premise_text'] = raw_df['premise_text'].fillna('')
                    raw_df['hypothesis_text'] = raw_df['hypothesis_text'].fillna('')
                    logger.info(f"Fallback data loaded successfully. Shape: {raw_df.shape}")
                    return raw_df[required_cols]
                else:
                    missing = [c for c in required_cols if c not in raw_df.columns]
                    logger.error(f"Fallback raw data missing required columns after processing: {missing}")
                    return None
            except Exception as e:
                logger.error(f"Error during fallback data loading: {e}", exc_info=True)
                return None

        return merged_df