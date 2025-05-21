# Modify: models/baseline_base.py
import logging
import os
from abc import abstractmethod, ABC
import joblib
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any, List  # Added Dict, Any, List
import time
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Use the existing NLIModel ABC from common.py as the ultimate base
from utils.common import NLIModel
from utils.database import DatabaseHandler
from config import MODELS_DIR, DATA_DIR

logger = logging.getLogger(__name__)


# --- Helper Functions (Keep existing _handle_nan_values, prepare_labels, etc.) ---

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
        return df  # Return as is if empty or None

    # Select only numeric columns for NaN check/fill
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if not numeric_cols:
        logger.debug(f"No numeric columns found to check for NaNs ({context}).")
        return df  # Return original if no numeric columns

    nan_check = df[numeric_cols].isnull().sum()
    total_nans = nan_check.sum()

    if total_nans > 0:
        logger.warning(f"Found {total_nans} NaN values in numeric columns during {context}. Filling with 0.")
        logger.debug(f"NaN counts per numeric column:\n{nan_check[nan_check > 0]}")
        # Fill NaNs only in numeric columns
        # Use .loc to ensure modification happens on the original DataFrame slice
        df.loc[:, numeric_cols] = df[numeric_cols].fillna(0)
        # Verify NaNs are filled
        if df[numeric_cols].isnull().sum().sum() > 0:
            logger.error(f"NaN values still present after attempting fillna(0) during {context}!")
    else:
        logger.debug(f"No NaN values found in numeric columns during {context}.")

    return df


def prepare_labels(labels, label_map=None):
    """Convert string labels to integers with a consistent mapping"""
    if label_map is None:
        label_map = {'entailment': 0, 'contradiction': 1, 'neutral': 2}

    # Ensure labels is a pandas Series for consistent handling
    if not isinstance(labels, pd.Series):
        labels = pd.Series(labels)

    if labels.dtype == object:
        # Handle potential NaNs introduced before conversion
        labels = labels.fillna('unknown')  # Or dropna() depending on desired behavior
        # Use numpy's vectorize for potentially faster mapping on large arrays
        vectorized_map = np.vectorize(lambda label: label_map.get(label, -1))
        return vectorized_map(labels.astype(str))  # Ensure string type before mapping
        # return np.array([label_map.get(label, -1) for label in labels]) # Original list comprehension
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
        return labels  # Return original array if conversion fails


def get_label_column(df: pd.DataFrame) -> Tuple[str, pd.Series]:  # Return Series for consistency
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
        return None  # Cannot proceed without labels

    # Convert labels to integers and handle unknowns (-1)
    int_labels = prepare_labels(labels_series)  # Pass the Series

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
                df_cleaned = df.loc[valid_mask_aligned]  # Use boolean Series on original index
            else:
                df_cleaned = df[valid_mask].reset_index(drop=True)  # Direct boolean mask ok
            int_labels_cleaned = int_labels[valid_mask]
        else:  # Should not happen if num_invalid > 0, but safe check
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


def _evaluate_model_performance(model: NLIModel, X_val: np.ndarray, y_val: np.ndarray) -> Tuple[
    float, Dict[str, float]]:
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
        logger.error(
            f"Feature/Label mismatch for evaluation: X_val has {X_val.shape[0]} samples, y_val has {len(y_val)} labels.")
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
    logger.info(
        f"Evaluation Metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    return eval_time, metrics


class TextFeatureExtractorBase:
    """Base class for TF-IDF and BoW extractors."""

    def __init__(self, vectorizer_class, **kwargs):
        self.vocabulary_ = None
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

    def transform(self, data: pd.DataFrame, text_col1: str = 'premise_text', text_col2: str = 'hypothesis_text') -> Any:
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
        instance = cls.__new__(cls)  # Create instance without calling __init__
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
    def extract_features(self, data: pd.DataFrame) -> Any:  # Return type can be sparse or dense
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

    def save(self, filepath: str, model_name: str) -> None:  # Adjusted signature slightly if needed based on caller
        """Saves the trained model and the feature extractor."""
        # --- MODIFICATION START ---
        # Use 'filepath' which is passed as the directory argument
        directory = filepath  # Assign filepath to directory variable for clarity or replace directly
        os.makedirs(directory, exist_ok=True)
        model_path = os.path.join(directory, f"{model_name}_model.joblib")
        extractor_path = os.path.join(directory, f"{model_name}_extractor.joblib")  # Save extractor
        # --- MODIFICATION END ---

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
            raise FileNotFoundError(
                f"Model or extractor file not found in directory: {directory} for base name {model_name}")

        loaded_model = joblib.load(model_path)
        loaded_extractor = joblib.load(extractor_path)  # Load the extractor object

        # --- MODIFICATION START ---
        # Instantiate the class (using defaults or loaded metadata if available later)
        # For now, assume default constructor works for subclasses like MultinomialNaiveBayesBaseline
        # TODO: Enhance this by saving/loading hyperparameters (e.g., alpha, max_features)
        #       and passing them to cls() here.
        try:
            instance = cls()  # Instantiate the specific class (e.g., MultinomialNaiveBayesBaseline)
        except Exception as e:
            logger.error(
                f"Failed to instantiate {cls.__name__} with default arguments during load: {e}. Check its __init__ method.",
                exc_info=True)
            raise TypeError(
                f"Cannot auto-instantiate {cls.__name__} during load. Requires specific hyperparameters.") from e

        # Assign the loaded components
        instance.model = loaded_model
        instance.extractor = loaded_extractor
        instance.is_trained = True  # Assume loaded model is trained
        # --- MODIFICATION END ---

        logger.info(f"{cls.__name__} loaded from {directory} using base name {model_name}")
        return instance

    @staticmethod
    def load_raw_text_data(dataset_name: str, split: str, suffix: str, db_handler: DatabaseHandler) -> Optional[
        pd.DataFrame]:
        """
        Loads raw text data by merging intermediate 'pairs' and 'sentences' files.
        Falls back to original data loader if intermediates are not found.
        """
        from data.data_loader import DatasetLoader  # Import locally if needed
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
                # Check if columns exist before trying to convert type
                if 'id' in pairs_df.columns: pairs_df['id'] = pairs_df['id'].astype(str)
                if 'premise_id' in pairs_df.columns: pairs_df['premise_id'] = pairs_df['premise_id'].astype(str)
                if 'hypothesis_id' in pairs_df.columns: pairs_df['hypothesis_id'] = pairs_df['hypothesis_id'].astype(
                    str)
                if 'id' in sentences_df.columns: sentences_df['id'] = sentences_df['id'].astype(str)

                sentences_premise = sentences_df[['id', 'text']].rename(columns={'text': 'premise_text', 'id': 'p_id'})
                sentences_hypothesis = sentences_df[['id', 'text']].rename(
                    columns={'text': 'hypothesis_text', 'id': 'h_id'})
                pairs_essential = pairs_df[['id', 'premise_id', 'hypothesis_id', 'label']].rename(
                    columns={'id': 'pair_id'})

                # Perform merges
                merged_df_temp1 = pd.merge(pairs_essential, sentences_premise, left_on='premise_id', right_on='p_id',
                                           how='left')
                merged_df = pd.merge(merged_df_temp1, sentences_hypothesis, left_on='hypothesis_id', right_on='h_id',
                                     how='left')

                # Select and rename final columns, handle potential missing text after merge
                final_cols = ['pair_id', 'premise_text', 'hypothesis_text', 'label']
                # Check if all required columns exist after potential merges
                required_cols_check = ['pair_id', 'premise_text', 'hypothesis_text', 'label']
                if not all(col in merged_df.columns for col in required_cols_check):
                    missing = [col for col in required_cols_check if col not in merged_df.columns]
                    logger.error(f"Columns missing after merge: {missing}. Merge failed.")
                    merged_df = None
                else:
                    merged_df = merged_df[final_cols]
                    # Check for NaNs specifically in text columns after merge (important!)
                    premise_nan_count = merged_df['premise_text'].isnull().sum()
                    hyp_nan_count = merged_df['hypothesis_text'].isnull().sum()
                    if premise_nan_count > 0 or hyp_nan_count > 0:
                        logger.warning(
                            f"Found {premise_nan_count} NaN premises and {hyp_nan_count} NaN hypotheses after merge. Filling with empty string.")
                        merged_df['premise_text'] = merged_df['premise_text'].fillna('')
                        merged_df['hypothesis_text'] = merged_df['hypothesis_text'].fillna('')

                    # Drop rows where label is still NaN (should be handled by clean_dataset later, but good check)
                    label_nan_count = merged_df['label'].isnull().sum()
                    if label_nan_count > 0:
                        logger.warning(f"Found {label_nan_count} NaN labels after merge. Dropping these rows.")
                        merged_df.dropna(subset=['label'], inplace=True)

                    logger.info(
                        f"Successfully loaded and merged intermediate data. Shape after merge: {merged_df.shape}")

            else:
                logger.warning(
                    f"Intermediate data ({pairs_table} or {sentences_table}) not found or empty. Falling back to original data loader.")
                merged_df = None

        except Exception as e:
            logger.error(f"Error loading/merging intermediate data: {e}. Falling back.", exc_info=True)
            merged_df = None

        # Fallback mechanism
        if merged_df is None or merged_df.empty:
            logger.warning("Executing fallback: Loading original raw data using DatasetLoader.")
            try:
                loader = DatasetLoader(db_handler)
                # Assume load_dataset returns the required columns directly or adapt here
                raw_df = loader.load_dataset(dataset_name, split=split)  # Load specific split
                if raw_df.empty:
                    logger.error("Fallback data loading returned empty DataFrame.")
                    return None

                # Ensure standard column names expected by TextBaseline models
                rename_map = {}
                if "premise" in raw_df.columns: rename_map["premise"] = "premise_text"
                if "hypothesis" in raw_df.columns: rename_map["hypothesis"] = "hypothesis_text"
                if rename_map:
                    raw_df = raw_df.rename(columns=rename_map, errors='ignore')

                # Add a pair_id if missing (using index, ensure it's unique)
                if 'pair_id' not in raw_df.columns:
                    if 'id' in raw_df.columns:  # Use existing 'id' if available
                        raw_df = raw_df.rename(columns={'id': 'pair_id'}, errors='ignore')
                    else:  # Create from index
                        raw_df.reset_index(inplace=True)
                        raw_df = raw_df.rename(columns={'index': 'pair_id'},
                                               errors='ignore')  # Rename index if it becomes a column

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


# --- START: Code moved from svm_bow_baseline.py ---

class FeatureExtractor(ABC):  # Changed to ABC for clarity
    """Base class for feature extraction from feature DataFrames."""

    @abstractmethod
    def extract(self, data: pd.DataFrame, feature_cols: List[str] = None) -> np.ndarray:
        """Extracts features into a numpy array."""
        raise NotImplementedError

    @abstractmethod
    def get_feature_columns(self, data: pd.DataFrame) -> List[str]:
        """Gets the list of feature column names from the data."""
        raise NotImplementedError


def _feature_return_helper(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Helper to select feature columns and essential ID/label columns."""
    cols_to_keep = feature_cols[:]  # Create a copy
    if 'label' in df.columns:
        cols_to_keep.append('label')
    elif 'gold_label' in df.columns:
        cols_to_keep.append('gold_label')
    if 'pair_id' in df.columns: cols_to_keep.append('pair_id')
    # Ensure only existing columns are selected
    existing_cols_to_keep = [col for col in cols_to_keep if col in df.columns]
    # logger.debug(f"Selecting columns for feature set: {existing_cols_to_keep}")
    return df[existing_cols_to_keep]


def filter_syntactic_features(df: pd.DataFrame) -> List[str]:
    """Return list of syntactic feature column names."""
    # Define prefixes or patterns that identify syntactic features
    # Adjust these based on your actual feature naming convention
    syntax_cols = [col for col in df.columns if any(prefix in col for prefix in
                                                    ['_const_', '_dep_', 'diff_const_', 'diff_dep_', 'deprel_',
                                                     'pos_'])]
    # logger.debug(f"Identified {len(syntax_cols)} syntactic columns.")
    return syntax_cols


def filter_lexical_features(df: pd.DataFrame) -> List[str]:
    """Return list of lexical/statistical feature column names."""
    # Define prefixes or patterns that identify lexical/statistical features
    # Adjust these based on your actual feature naming convention
    # Example: '_bert_' might be embeddings, '_length' simple stats, 'overlap' specific lexical metric
    lexical_cols = [col for col in df.columns if any(prefix in col for prefix in
                                                     ['_bert_', '_length', 'length_', 'overlap', '_tfidf_',
                                                      '_bow_'])]  # Added tfidf/bow common patterns
    # logger.debug(f"Identified {len(lexical_cols)} lexical/stat columns.")
    return lexical_cols


# --- Feature Extractors using precomputed/existing features ---
# These assume features are already calculated and present in the DataFrame

class LexicalFeatureExtractor(FeatureExtractor):
    """Extracts precomputed lexical/statistical features from a DataFrame."""

    def get_feature_columns(self, data: pd.DataFrame) -> List[str]:
        return filter_lexical_features(data)

    def extract(self, data: pd.DataFrame, feature_cols: List[str] = None) -> np.ndarray:
        target_cols = feature_cols or self.get_feature_columns(data)
        # Ensure all target columns exist, fill missing with 0 if necessary during prediction
        missing_cols = set(target_cols) - set(data.columns)
        if missing_cols:
            logger.warning(f"Lexical extractor: Missing columns {missing_cols}. Filling with 0.")
            # Create a copy to avoid modifying the original DataFrame slice
            data_copy = data.copy()
            for col in missing_cols: data_copy[col] = 0
            return data_copy[target_cols].values
        return data[target_cols].values


class SyntacticFeatureExtractor(FeatureExtractor):
    """Extracts precomputed syntactic features from a DataFrame."""

    def get_feature_columns(self, data: pd.DataFrame) -> List[str]:
        return filter_syntactic_features(data)

    def extract(self, data: pd.DataFrame, feature_cols: List[str] = None) -> np.ndarray:
        target_cols = feature_cols or self.get_feature_columns(data)
        missing_cols = set(target_cols) - set(data.columns)
        if missing_cols:
            logger.warning(f"Syntactic extractor: Missing columns {missing_cols}. Filling with 0.")
            data_copy = data.copy()
            for col in missing_cols: data_copy[col] = 0
            return data_copy[target_cols].values
        return data[target_cols].values


class CombinedFeatureExtractor(FeatureExtractor):
    """Extracts all precomputed features (lexical + syntactic) from a DataFrame."""

    def get_feature_columns(self, data: pd.DataFrame) -> List[str]:
        lexical = filter_lexical_features(data)
        syntactic = filter_syntactic_features(data)
        # Combine, ensuring no duplicates and sort for consistency
        return sorted(list(set(lexical + syntactic)))

    def extract(self, data: pd.DataFrame, feature_cols: List[str] = None) -> np.ndarray:
        target_cols = feature_cols or self.get_feature_columns(data)
        missing_cols = set(target_cols) - set(data.columns)
        if missing_cols:
            logger.warning(f"Combined extractor: Missing columns {missing_cols}. Filling with 0.")
            data_copy = data.copy()
            for col in missing_cols: data_copy[col] = 0
            return data_copy[target_cols].values
        return data[target_cols].values


# --- END: Code moved from svm_bow_baseline.py ---

# ... (rest of the existing code in baseline_base.py) ...

# Example of how a FeatureBasedBaselineModel might look (if you don't have one already)
class FeatureBasedBaselineModel(NLIModel):
    """Abstract Base Class for models that operate on pre-extracted features."""

    def __init__(self, feature_extractor: FeatureExtractor, **kwargs):
        if feature_extractor is None:
            raise ValueError(f"{self.__class__.__name__} requires a FeatureExtractor instance.")
        self.feature_extractor = feature_extractor
        self.model = None  # The actual sklearn or similar model
        self.is_trained = False
        self.feature_cols = None  # Stores feature names used during training

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the underlying model."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make pred ictions using the trained model."""
        raise NotImplementedError

    def extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extracts features using the assigned extractor, handling train/predict differences."""
        if not self.is_trained:
            # Training: discover and store feature column names
            self.feature_cols = self.feature_extractor.get_feature_columns(data)
            logger.info(f"Storing {len(self.feature_cols)} feature columns used for training.")
            # Extract using discovered columns
            return self.feature_extractor.extract(data, self.feature_cols)
        else:
            # Prediction: use the stored feature column names
            if self.feature_cols is None:
                raise RuntimeError("Model is marked trained but feature_cols is not set.")
            logger.debug(f"Extracting features for prediction using stored {len(self.feature_cols)} columns.")
            # Pass stored columns to extractor
            return self.feature_extractor.extract(data, self.feature_cols)

    @abstractmethod
    def save(self, filepath: str) -> None:
        """Save the model state."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, filepath: str, feature_extractor: FeatureExtractor = None) -> 'FeatureBasedBaselineModel':
        """Load the model state."""
        raise NotImplementedError


# Inside models/baseline_base.py

class SimpleParquetLoader:
    @staticmethod
    def load_data(dataset_name: str, split: str, suffix: str) -> Optional[pd.DataFrame]:
        # Construct base cache directory
        cache_dir = os.path.join('cache', 'parquet', dataset_name, split)
        logger.info(f"SimpleParquetLoader: Searching for features in {cache_dir} for {dataset_name}/{split}/{suffix}")

        # Define potential feature file patterns by preference
        # Prioritize more comprehensive feature sets
        preferred_feature_filenames = [
            f"{dataset_name}_{split}_features_stats_syntactic_{suffix}.parquet",  # Original pattern - prioritize this
            f"{dataset_name}_{split}_features_lexical_syntactic_{suffix}.parquet",  # Fallback to this
            f"{dataset_name}_{split}_features_all_{suffix}.parquet",  # Another common comprehensive name
            f"{dataset_name}_{split}_features_stats_syntactic_full.parquet",  # Fallback to full if sample not found
            f"{dataset_name}_{split}_features_lexical_syntactic_full.parquet",  # Fallback to full if sample not found
        ]

        # Fallback patterns (less likely to contain all needed features for experiments)
        fallback_filenames = [
            f"features_combined_{suffix}.parquet",  # A generic combined name
            f"raw_data_{suffix}.parquet",  # Original fallback
            f"raw_data_full.parquet"  # Final original fallback
        ]

        found_filepath = None
        required_cols = ['pair_id', 'premise_text', 'hypothesis_text']  # Essential columns

        # Try preferred filenames first
        for fname in preferred_feature_filenames:
            filepath = os.path.join(cache_dir, fname)
            logger.debug(f"SimpleParquetLoader: Attempting to load preferred file: {filepath}")
            if os.path.exists(filepath):
                # Check if file has required columns before selecting it
                try:
                    df = pd.read_parquet(filepath)
                    if all(col in df.columns for col in required_cols):
                        found_filepath = filepath
                        break
                    else:
                        missing = [col for col in required_cols if col not in df.columns]
                        logger.warning(f"File {filepath} exists but is missing required columns: {missing}")
                except Exception as e:
                    logger.warning(f"Error checking file {filepath}: {e}")
        
        # If not found, try fallback filenames
        if not found_filepath:
            logger.warning(f"SimpleParquetLoader: Preferred feature files not found. Trying fallbacks...")
            for fname in fallback_filenames:
                filepath = os.path.join(cache_dir, fname)
                logger.debug(f"SimpleParquetLoader: Attempting to load fallback file: {filepath}")
                if os.path.exists(filepath):
                    try:
                        df = pd.read_parquet(filepath)
                        if all(col in df.columns for col in required_cols):
                            found_filepath = filepath
                            break
                        else:
                            missing = [col for col in required_cols if col not in df.columns]
                            logger.warning(f"File {filepath} exists but is missing required columns: {missing}")
                    except Exception as e:
                        logger.warning(f"Error checking file {filepath}: {e}")
        
        if not found_filepath:
            logger.error(f"SimpleParquetLoader: Could not find any suitable feature parquet file in {cache_dir} with required columns: {required_cols}")
            raise FileNotFoundError(
                f"SimpleParquetLoader: No feature parquet data found in {cache_dir} for {dataset_name}/{split}/{suffix} with required columns: {required_cols}"
            )

        logger.info(f"SimpleParquetLoader: Loading final features from: {found_filepath}")
        df = pd.read_parquet(found_filepath)
        logger.info(f"SimpleParquetLoader: Loaded {len(df)} rows from {found_filepath}")

        # Final check for required columns
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"SimpleParquetLoader: Loaded DataFrame from {found_filepath} is missing essential columns: {missing_cols}")
            raise ValueError(f"Missing essential columns ({missing_cols}) in loaded file {found_filepath}")
        
        return df
