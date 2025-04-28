# models/svm_bow_baseline.py
import os
import logging
import time
import glob
import numpy as np
import pandas as pd
import joblib
from typing import Tuple, List, Optional, Any

from sklearn.svm import SVC

# Keep NLIModel ABC for SVMModel inheritance (could change to a FeatureBasedBaselineModel if needed)
from utils.common import NLIModel
from config import MODELS_DIR
# Import helpers from the new base trainer/model files if they are defined there
# Or keep them here if specific to SVM's precomputed feature loading
from .baseline_base import clean_dataset, _evaluate_model_performance

logger = logging.getLogger(__name__)

# --- Data loading specific to SVM (precomputed features) ---
def load_parquet_data(dataset_name: str, split: str = 'train',
                      feature_type: str = None, # e.g., 'features_lexical_syntactic_full'
                      sample_size: Optional[int] = None, # sample_size is now part of feature_type name usually
                      cache_dir: Optional[str] = None) -> pd.DataFrame:
    """Loads precomputed SVM features from parquet files."""
    # Use MODELS_DIR or a dedicated feature cache dir
    cache_dir = cache_dir or os.path.join(MODELS_DIR, '..', 'cache', 'parquet') # Example path

    # Construct filename directly using the new format used by FeatureExtractor
    # feature_type here should be the full base name like 'SNLI_train_features_lexical_syntactic_full'
    # The split info is already in the feature_type name now.
    # We need to adjust how feature_type is passed or reconstruct the pattern.
    # Let's assume feature_type passed is like 'features_lexical_syntactic_full'
    # And we need to combine with dataset_name and split.

    # Revised pattern logic based on how BaselineTrainer saves features:
    # File name is like: {dataset_name}_{split}_{feature_type}.parquet
    filename_pattern = f"{dataset_name}_{split}_{feature_type}.parquet"
    full_pattern = os.path.join(cache_dir, filename_pattern)

    logger.info(f"SVM: Attempting to load features from: {full_pattern}")

    parquet_files = glob.glob(full_pattern)
    if not parquet_files:
        # Fallback or error if specific file not found
        logger.warning(f"No feature file found matching: {full_pattern}")
        # You might add fallbacks here if naming conventions vary
        raise FileNotFoundError(f"No SVM feature parquet files found for {dataset_name} {split} with type {feature_type}")

    # Load the first matching file (assuming one per dataset/split/feature_type)
    logger.info(f"Loading features from {parquet_files[0]}")
    result = pd.read_parquet(parquet_files[0])

    # Sampling is usually done *before* feature extraction now.
    # If sample_size needs to be applied here, uncomment below.
    # if sample_size and len(result) > sample_size:
    #     logger.info(f"Sampling {sample_size} examples from loaded features.")
    #     result = result.sample(sample_size, random_state=42)

    logger.info(f"Loaded {len(result)} examples.")
    return result


def _handle_nan_values(df, dataset_name):
    """Check for and handle NaN values in dataframe"""
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        logger.warning(f"Found {nan_count} NaN values in SVM {dataset_name} data, filling with 0")
        # It's crucial *how* NaNs are filled. 0 is common, but verify appropriateness.
        return df.fillna(0)
    return df


# --- SVM Feature Extractors (Remain largely unchanged) ---
class FeatureExtractor:
    """Base class for SVM feature extraction from precomputed feature DataFrames."""
    def extract(self, data: pd.DataFrame, feature_cols: List[str] = None) -> np.ndarray:
        raise NotImplementedError
    def get_feature_columns(self, data: pd.DataFrame) -> List[str]:
        raise NotImplementedError

def _feature_return_helper(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Helper to select feature columns and essential ID/label columns."""
    cols_to_keep = feature_cols[:] # Create a copy
    if 'label' in df.columns: cols_to_keep.append('label')
    elif 'gold_label' in df.columns: cols_to_keep.append('gold_label')
    if 'pair_id' in df.columns: cols_to_keep.append('pair_id')
    # Ensure only existing columns are selected
    existing_cols_to_keep = [col for col in cols_to_keep if col in df.columns]
    # logger.debug(f"Selecting columns for feature set: {existing_cols_to_keep}")
    return df[existing_cols_to_keep]

def filter_syntactic_features(df: pd.DataFrame) -> List[str]:
    """Return list of syntactic feature column names."""
    syntax_cols = [col for col in df.columns if any(prefix in col for prefix in
                                                    ['_const_', '_dep_', 'diff_const_', 'diff_dep_', 'deprel_', 'pos_'])]
    # logger.debug(f"Identified {len(syntax_cols)} syntactic columns.")
    return syntax_cols

def filter_lexical_features(df: pd.DataFrame) -> List[str]:
    """Return list of lexical/statistical feature column names."""
    lexical_cols = [col for col in df.columns if any(prefix in col for prefix in
                                                     ['_bert_', '_length', 'length_', 'overlap'])]
    # logger.debug(f"Identified {len(lexical_cols)} lexical/stat columns.")
    return lexical_cols

class LexicalFeatureExtractor(FeatureExtractor):
    """Extracts precomputed lexical/statistical features."""
    def get_feature_columns(self, data: pd.DataFrame) -> List[str]:
        return filter_lexical_features(data)
    def extract(self, data: pd.DataFrame, feature_cols: List[str] = None) -> np.ndarray:
        target_cols = feature_cols or self.get_feature_columns(data)
        # Ensure all target columns exist, fill missing with 0 if necessary during prediction
        missing_cols = set(target_cols) - set(data.columns)
        if missing_cols:
            logger.warning(f"Lexical extractor: Missing columns {missing_cols}. Filling with 0.")
            for col in missing_cols: data[col] = 0
        return data[target_cols].values

class SyntacticFeatureExtractor(FeatureExtractor):
    """Extracts precomputed syntactic features."""
    def get_feature_columns(self, data: pd.DataFrame) -> List[str]:
        return filter_syntactic_features(data)
    def extract(self, data: pd.DataFrame, feature_cols: List[str] = None) -> np.ndarray:
        target_cols = feature_cols or self.get_feature_columns(data)
        missing_cols = set(target_cols) - set(data.columns)
        if missing_cols:
            logger.warning(f"Syntactic extractor: Missing columns {missing_cols}. Filling with 0.")
            for col in missing_cols: data[col] = 0
        return data[target_cols].values

class CombinedFeatureExtractor(FeatureExtractor):
    """Extracts all precomputed features (lexical + syntactic)."""
    def get_feature_columns(self, data: pd.DataFrame) -> List[str]:
        lexical = filter_lexical_features(data)
        syntactic = filter_syntactic_features(data)
        # Combine, ensuring no duplicates
        return sorted(list(set(lexical + syntactic)))
    def extract(self, data: pd.DataFrame, feature_cols: List[str] = None) -> np.ndarray:
        target_cols = feature_cols or self.get_feature_columns(data)
        missing_cols = set(target_cols) - set(data.columns)
        if missing_cols:
            logger.warning(f"Combined extractor: Missing columns {missing_cols}. Filling with 0.")
            for col in missing_cols: data[col] = 0
        return data[target_cols].values

# --- SVM Model Classes (Inherit NLIModel, use specific extractors) ---
class SVMModel(NLIModel):
    """Base class for SVM models using precomputed features."""
    def __init__(self, feature_extractor: FeatureExtractor, kernel: str = 'linear', C: float = 1.0):
        if feature_extractor is None:
             raise ValueError("SVMModel requires a FeatureExtractor instance.")
        self.feature_extractor = feature_extractor
        self.kernel = kernel
        self.C = C
        self.svm = SVC(kernel=kernel, C=C, probability=True, random_state=42)
        self.is_trained = False
        self.feature_cols = None # Stores the names of columns used for training

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

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the SVM model."""
        if X is None or X.size == 0 or y is None or y.size == 0:
             logger.error("Cannot train SVM with empty features or labels.")
             return
        logger.info(f"Training SVM ({self.kernel}, C={self.C}) with {X.shape[0]} samples, {X.shape[1]} features")
        self.svm.fit(X, y)
        self.is_trained = True
        logger.info("SVM training complete.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the trained SVM model."""
        if not self.is_trained: raise RuntimeError("SVM Model has not been trained yet.")
        if self.svm is None: raise RuntimeError("SVM internal model is not initialized.")
        if X is None or X.shape[1] != len(self.feature_cols):
             expected = len(self.feature_cols) if self.feature_cols else 'N/A'
             actual = X.shape[1] if X is not None else 'None'
             raise ValueError(f"Feature mismatch for prediction. Expected {expected}, got {actual}.")
        return self.svm.predict(X)

    def save(self, filepath: str) -> None:
        """Save the SVM model state, including used feature columns."""
        if not self.is_trained or self.feature_cols is None:
             logger.warning("Attempting to save an untrained SVM model or model with missing feature columns.")
        model_data = {
            'svm_state': self.svm, # Save the sklearn SVC object
            'kernel': self.kernel,
            'C': self.C,
            'is_trained': self.is_trained,
            'feature_cols': self.feature_cols # Crucial for consistent prediction
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Saved SVM model state to {filepath}")

    @classmethod
    def load(cls, filepath: str, feature_extractor: FeatureExtractor = None) -> 'SVMModel':
        """Load SVM model state. Feature extractor type needs to be known."""
        if feature_extractor is None:
             # Cannot determine the correct extractor type from saved file alone.
             # This highlights a design challenge. The loader (trainer) needs to know
             # which extractor to associate with the loaded model.
             raise ValueError("FeatureExtractor instance must be provided when loading SVMModel.")

        logger.info(f"Loading SVM model state from {filepath}")
        model_data = joblib.load(filepath)

        # Create instance with the correct extractor and loaded parameters
        instance = cls(feature_extractor, kernel=model_data['kernel'], C=model_data['C'])
        instance.svm = model_data['svm_state']
        instance.is_trained = model_data['is_trained']
        instance.feature_cols = model_data['feature_cols'] # Load the feature column names

        if instance.feature_cols is None and instance.is_trained:
             logger.warning(f"Loaded trained SVM model from {filepath} but feature_cols list is missing.")

        logger.info(f"Loaded SVM model ({instance.kernel}, C={instance.C}). Trained: {instance.is_trained}")
        return instance


class SVMWithBagOfWords(SVMModel):
    """SVM model using only precomputed lexical/statistical features."""
    def __init__(self, kernel: str = 'linear', C: float = 1.0):
        super().__init__(LexicalFeatureExtractor(), kernel, C)

    @classmethod
    def load(cls, filepath: str, feature_extractor: Optional[FeatureExtractor] = None) -> 'SVMModel':
         # Ensure correct extractor is passed during load
         if feature_extractor is None: feature_extractor = LexicalFeatureExtractor()
         return super().load(filepath, feature_extractor)

class SVMWithSyntax(SVMModel):
    """SVM model using only precomputed syntactic features."""
    def __init__(self, kernel: str = 'linear', C: float = 1.0):
        super().__init__(SyntacticFeatureExtractor(), kernel, C)

    @classmethod
    def load(cls, filepath: str, feature_extractor: Optional[FeatureExtractor] = None) -> 'SVMModel':
        if feature_extractor is None: feature_extractor = SyntacticFeatureExtractor()
        return super().load(filepath, feature_extractor)

class SVMWithBothFeatures(SVMModel):
    """SVM model using both precomputed lexical and syntactic features."""
    def __init__(self, kernel: str = 'linear', C: float = 1.0):
        super().__init__(CombinedFeatureExtractor(), kernel, C)

    @classmethod
    def load(cls, filepath: str, feature_extractor: Optional[FeatureExtractor] = None) -> 'SVMModel':
        if feature_extractor is None: feature_extractor = CombinedFeatureExtractor()
        return super().load(filepath, feature_extractor)

# Removed SVMTrainer class
# Removed _load_datasets, _train_all_models, train_model helpers (moved to BaselineTrainer)
# Kept SVM-specific feature extractors and model subclasses
# Kept SVM-specific data loading (load_parquet_data) and NaN handling (_handle_nan_values) for now