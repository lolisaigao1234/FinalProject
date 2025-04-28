# Create file: IS567FP/models/random_forest_bow_syntactic_experiment_5.py
# --- models/random_forest_bow_syntactic_experiment_5.py ---
import logging
import os
import numpy as np
import pandas as pd
import joblib
from typing import List, Optional, Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted

# Import base class and feature loading/filtering utilities
from utils.common import NLIModel
from .baseline_base import _handle_nan_values, _evaluate_model_performance, clean_dataset # Basic helpers
# Need CombinedFeatureExtractor logic or similar to select correct columns
from .svm_bow_baseline import FeatureExtractor, filter_lexical_features, filter_syntactic_features # Reuse filtering logic

logger = logging.getLogger(__name__)

class CombinedBowSyntacticExtractor(FeatureExtractor):
    """
    Selects pre-computed BoW (lexical) and Syntactic features.
    Similar to CombinedFeatureExtractor in svm_bow_baseline.py,
    but defined here for clarity or minor adaptation if needed.
    """
    def get_feature_columns(self, data: pd.DataFrame) -> List[str]:
        """Get all relevant feature columns (BoW/lexical + syntactic)."""
        lexical_cols = filter_lexical_features(data)
        syntactic_cols = filter_syntactic_features(data)
        # Combine and remove duplicates, ensure sorting for consistency
        combined_cols = sorted(list(set(lexical_cols + syntactic_cols)))
        logger.debug(f"Combined BoW+Syntactic extractor identified {len(combined_cols)} columns.")
        # Exclude label/id columns if they were included by the filters
        final_cols = [col for col in combined_cols if col not in ['label', 'gold_label', 'pair_id']]
        return final_cols

    def extract(self, data: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
        """Extracts the specified columns, handling potential missing ones."""
        # Ensure all required columns exist in the dataframe
        missing_cols = set(feature_cols) - set(data.columns)
        if missing_cols:
            logger.warning(f"Combined BoW+Syntactic extractor: Missing columns {missing_cols} during extraction. Filling with 0.")
            data_copy = data.copy()
            for col in missing_cols:
                data_copy[col] = 0
            # Ensure correct column order
            return data_copy[feature_cols].values
        else:
            # Ensure correct column order
            return data[feature_cols].values


class RandomForestBowSyntacticExperiment5(NLIModel):
    """
    Experiment 5: Random Forest using combined BoW (lexical) and hand-crafted syntactic features.
    Inherits from NLIModel. Assumes features are pre-computed.
    """
    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None, random_state: int = 42, **kwargs):
        """
        Initialize the Experiment 5 Random Forest model.

        Args:
            n_estimators (int): Number of trees in the forest.
            max_depth (Optional[int]): Maximum depth of the trees.
            random_state (int): Controls randomness for reproducibility.
            **kwargs: Additional arguments for RandomForestClassifier.
        """
        logger.info(f"Initializing Experiment 5 Random Forest (n_estimators={n_estimators}, max_depth={max_depth})")
        self.feature_extractor = CombinedBowSyntacticExtractor()
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1, # Use all available cores
            **kwargs
        )
        self.is_trained = False
        self.feature_cols = None # Stores feature names used during training

        # Store hyperparameters
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model_kwargs = kwargs

    def extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Extracts combined BoW and syntactic features using the assigned extractor.
        Handles discovery of columns during training and uses stored columns during prediction.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame.")

        if not self.is_trained or self.feature_cols is None:
            # Training phase: discover and store feature columns
            logger.debug("Experiment 5: Discovering feature columns during training.")
            self.feature_cols = self.feature_extractor.get_feature_columns(data)
            if not self.feature_cols:
                 raise ValueError("Experiment 5: No features extracted during training phase.")
            logger.info(f"Experiment 5: Storing {len(self.feature_cols)} feature columns for training.")
            # Extract using discovered columns
            features = self.feature_extractor.extract(data, self.feature_cols)
        else:
            # Prediction phase: use stored feature columns
            logger.debug(f"Experiment 5: Extracting features for prediction using stored {len(self.feature_cols)} columns.")
            features = self.feature_extractor.extract(data, self.feature_cols)

        # Ensure features are numeric and handle potential NaNs introduced by merging/extraction steps
        if features.dtype == 'object' or pd.DataFrame(features).isna().any().any():
            logger.warning("NaNs or non-numeric types detected after feature extraction. Applying fillna(0).")
            features = pd.DataFrame(features, columns=self.feature_cols).fillna(0).values

        return features.astype(np.float32) # Ensure float type for RF


    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Trains the RandomForestClassifier model."""
        if X is None or X.size == 0 or y is None or y.size == 0:
             logger.error("Cannot train Random Forest with empty features or labels.")
             return
        # Check for NaN/Infinity in features before training
        if not np.all(np.isfinite(X)):
            logger.error("Non-finite values (NaN or Infinity) found in training features (X). Cannot train Random Forest.")
            # Add more diagnostics if needed: np.isnan(X).sum(), np.isinf(X).sum()
            return
        logger.info(f"Training Random Forest ({self.n_estimators} estimators) with {X.shape[0]} samples, {X.shape[1]} features")
        self.model.fit(X, y)
        self.is_trained = True
        logger.info("Random Forest training complete.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Makes predictions using the trained RandomForestClassifier."""
        if not self.is_trained:
            raise RuntimeError("Random Forest Model has not been trained yet.")
        check_is_fitted(self.model) # Use sklearn's check

        if X is None:
             raise ValueError("Input features (X) for prediction cannot be None.")
        if self.feature_cols is None:
             raise RuntimeError("Feature columns were not set during training.")
        if X.shape[1] != len(self.feature_cols):
             raise ValueError(f"Feature mismatch for prediction. Expected {len(self.feature_cols)} features, got {X.shape[1]}.")
        # Check for NaN/Infinity in prediction features
        if not np.all(np.isfinite(X)):
            logger.warning("Non-finite values (NaN or Infinity) found in prediction features (X). Attempting to predict anyway after filling with 0.")
            X = np.nan_to_num(X, nan=0.0, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)


        logger.debug(f"Predicting with trained Random Forest on {X.shape[0]} samples...")
        predictions = self.model.predict(X)
        logger.debug("Prediction finished.")
        return predictions

    def save(self, filepath: str) -> None:
        """Saves the trained RandomForest model state and feature columns."""
        # Filepath is expected to be the base path (e.g., /path/to/dir/model_name_suffix)
        model_path = f"{filepath}.joblib" # Save directly as joblib
        logger.info(f"Saving Experiment 5 Random Forest model state to {model_path}")

        if not self.is_trained:
            logger.warning("Attempting to save an untrained Random Forest model.")
        if self.feature_cols is None:
             logger.warning("Feature columns not set. Saving model without feature column list.")

        model_data = {
            'model_state': self.model,
            'is_trained': self.is_trained,
            'feature_cols': self.feature_cols,
            'hyperparameters': { # Store hyperparameters for reproducibility
                 'n_estimators': self.n_estimators,
                 'max_depth': self.max_depth,
                 'random_state': self.random_state,
                 **self.model_kwargs # Include any extra kwargs
            }
        }
        try:
            joblib.dump(model_data, model_path)
            logger.info(f"Successfully saved model data to {model_path}")
        except Exception as e:
             logger.error(f"Error saving Random Forest model data to {model_path}: {e}", exc_info=True)


    @classmethod
    def load(cls, filepath: str, feature_extractor: Optional[Any] = None) -> 'RandomForestBowSyntacticExperiment5':
        """Loads the RandomForest model state."""
        # Filepath is expected to be the base path (e.g., /path/to/dir/model_name_suffix)
        model_path = f"{filepath}.joblib"
        logger.info(f"Loading Experiment 5 Random Forest model state from {model_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        try:
            model_data = joblib.load(model_path)
        except Exception as e:
            logger.error(f"Error loading Random Forest model data from {model_path}: {e}", exc_info=True)
            raise

        # Re-instantiate the class with saved hyperparameters
        hyperparams = model_data.get('hyperparameters', {})
        instance = cls(
            n_estimators=hyperparams.get('n_estimators', 100),
            max_depth=hyperparams.get('max_depth', None),
            random_state=hyperparams.get('random_state', 42),
            **{k: v for k, v in hyperparams.items() if k not in ['n_estimators', 'max_depth', 'random_state']} # Pass extra kwargs
        )
        instance.model = model_data['model_state']
        instance.is_trained = model_data['is_trained']
        instance.feature_cols = model_data['feature_cols']

        if instance.feature_cols is None and instance.is_trained:
             logger.warning(f"Loaded trained Random Forest model from {model_path} but feature_cols list is missing.")

        logger.info(f"Successfully loaded Experiment 5 Random Forest model. Trained: {instance.is_trained}")
        return instance
