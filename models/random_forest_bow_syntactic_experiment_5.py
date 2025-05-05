# models/random_forest_bow_syntactic_experiment_5.py
import logging
import os

import joblib
import numpy as np
import pandas as pd
import time
from typing import List, Optional, Any # Ensure necessary imports

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack, csr_matrix  # To combine sparse matrices

# --- Updated Import ---
# Import FeatureExtractor from its new location
# Ensure baseline_base contains FeatureBasedBaselineModel, FeatureExtractor, SyntacticFeatureExtractor
from .baseline_base import FeatureBasedBaselineModel, FeatureExtractor, SyntacticFeatureExtractor
# --- End Updated Import ---

logger = logging.getLogger(__name__)

# --- Feature Extractor for this specific experiment ---
# This class now correctly inherits from the FeatureExtractor defined in baseline_base.py
class CombinedBowSyntacticExtractor(FeatureExtractor):
    """Combines Bag-of-Words features with precomputed Syntactic features."""
    def __init__(self, max_features: Optional[int] = 5000, ngram_range: tuple = (1, 1)):
        self.bow_vectorizer = CountVectorizer(
            # preprocessor=preprocess_text, # Use your preprocessing function if needed
            max_features=max_features,
            ngram_range=ngram_range,
            lowercase=True
        )
        # Use the SyntacticFeatureExtractor moved to baseline_base
        self.syntactic_extractor = SyntacticFeatureExtractor()
        self.bow_feature_names_ = None # To store BOW feature names after fitting
        self.syntactic_feature_names_ = None # To store Syntactic feature names

    def fit(self, data: pd.DataFrame, y: Optional[Any] = None):
        """Fit the BoW vectorizer and identify syntactic feature names."""
        logger.info("Fitting BoW vectorizer...")
        # Combine premise and hypothesis for BoW vocabulary fitting if they exist
        if 'premise' in data.columns and 'hypothesis' in data.columns:
             combined_text = data['premise'] + " " + data['hypothesis']
             self.bow_vectorizer.fit(combined_text)
             self.bow_feature_names_ = self.bow_vectorizer.get_feature_names_out()
             logger.info(f"BoW vectorizer fitted with {len(self.bow_feature_names_)} features.")
        else:
             logger.warning("Missing 'premise' or 'hypothesis' text columns in data. BoW vectorizer not fitted.")
             self.bow_feature_names_ = []


        # Also get syntactic feature names (requires the dataframe structure)
        self.syntactic_feature_names_ = self.syntactic_extractor.get_feature_columns(data)
        logger.info(f"Identified {len(self.syntactic_feature_names_)} syntactic features.")
        return self

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """Transform data into combined BoW and Syntactic features."""
        # BoW Transformation
        if self.bow_feature_names_ is not None and len(self.bow_feature_names_) > 0:
             logger.info("Transforming data with BoW vectorizer...")
             premise_bow = self.bow_vectorizer.transform(data['premise'])
             hypothesis_bow = self.bow_vectorizer.transform(data['hypothesis'])
             # Combine BoW features (e.g., concatenate, difference, product - concatenating here)
             bow_features = hstack([premise_bow, hypothesis_bow]) # sparse matrix
             logger.info(f"BoW features shape: {bow_features.shape}")
        else:
            logger.info("BoW vectorizer not fitted or has no features. Skipping BoW transformation.")
            # Create an empty sparse matrix with the correct number of rows
            bow_features = hstack([csr_matrix((data.shape[0], 0)), csr_matrix((data.shape[0], 0))])


        # Syntactic Transformation
        if self.syntactic_feature_names_ is not None and len(self.syntactic_feature_names_) > 0:
             logger.info("Extracting syntactic features...")
             # Extract syntactic features using the dedicated extractor
             syntactic_features_np = self.syntactic_extractor.extract(data, self.syntactic_feature_names_)
             logger.info(f"Syntactic features shape: {syntactic_features_np.shape}")
             # Combine sparse BoW features with dense Syntactic features
             # Need to handle potential NaN in syntactic features if not already done
             syntactic_features_np = np.nan_to_num(syntactic_features_np) # Simple NaN handling
             # Convert dense syntactic features to sparse for hstack compatibility
             syntactic_features_sparse = csr_matrix(syntactic_features_np)
        else:
            logger.info("Syntactic extractor not fitted or has no features. Skipping syntactic transformation.")
            # Create an empty sparse matrix with the correct number of rows
            syntactic_features_sparse = csr_matrix((data.shape[0], 0))


        # Combine BoW and Syntactic (ensure at least one has features)
        if bow_features.shape[1] > 0 and syntactic_features_sparse.shape[1] > 0:
            combined_features = hstack([bow_features, syntactic_features_sparse]).tocsr() # combine as sparse
        elif bow_features.shape[1] > 0:
            combined_features = bow_features.tocsr()
        elif syntactic_features_sparse.shape[1] > 0:
            combined_features = syntactic_features_sparse.tocsr()
        else:
            logger.warning("Both BoW and Syntactic features are empty. Returning empty feature matrix.")
            combined_features = csr_matrix((data.shape[0], 0)) # Empty sparse matrix

        logger.info(f"Combined features shape: {combined_features.shape}")
        return combined_features

    def fit_transform(self, data: pd.DataFrame, y: Optional[Any] = None) -> np.ndarray:
        """Fit the BoW vectorizer and then transform the data."""
        self.fit(data, y)
        return self.transform(data)

    # --- Implementing abstract methods from FeatureExtractor ---
    def extract(self, data: pd.DataFrame, feature_cols: List[str] = None) -> np.ndarray:
        """ Extracts features (implements the base class method).
            Note: For vectorizers, usually fit/transform is used.
            This implementation assumes fit has already been called.
            'feature_cols' are implicitly handled by the fitted vectorizer + syntactic extractor.
        """
        # Check if fit seems to have been called (at least one feature name list exists)
        if self.bow_feature_names_ is None and self.syntactic_feature_names_ is None:
             raise RuntimeError("Extractor must be fitted before calling extract/transform.")
        # Re-use the transform logic
        return self.transform(data)


    def get_feature_columns(self, data: pd.DataFrame = None) -> List[str]:
        """ Gets the combined list of feature names. Requires fitting first.
            'data' parameter is optional here as names are stored after fit.
        """
        # Check if fit seems to have been called
        if self.bow_feature_names_ is None and self.syntactic_feature_names_ is None:
             logger.warning("Extractor not fitted. Returning empty feature list.")
             return []

        # Prepare BoW feature names
        premise_bow_names = []
        hypothesis_bow_names = []
        if self.bow_feature_names_ is not None:
             premise_bow_names = [f"premise_{name}" for name in self.bow_feature_names_]
             hypothesis_bow_names = [f"hypothesis_{name}" for name in self.bow_feature_names_]

        # Prepare syntactic feature names (ensure it's a list)
        syntactic_names = self.syntactic_feature_names_ if self.syntactic_feature_names_ is not None else []

        # Return combined list
        return premise_bow_names + hypothesis_bow_names + syntactic_names


# --- Random Forest Model using the Combined Extractor ---
class RandomForestBowSyntacticExperiment5(FeatureBasedBaselineModel): # Inherit from base
    """Random Forest model using Bag-of-Words and Syntactic features."""
    MODEL_NAME = "RandomForest_BoW_Syntactic_Exp5"

    # ++++++++++++++ MODIFIED __init__ ++++++++++++++
    def __init__(self,
                 args=None, # Accept args from trainer
                 max_features: Optional[int] = 5000, # BoW param
                 ngram_range: tuple = (1, 1),       # BoW param
                 n_estimators: int = 100,           # RF param (default)
                 max_depth: Optional[int] = None,   # RF param (default)
                 random_state: int = 42,            # RF param (default)
                 min_samples_split: int = 2,        # RF param (default)
                 min_samples_leaf: int = 1,         # RF param (default)
                 # **kwargs # Removed - DO NOT blindly pass kwargs
                 ):

        # Initialize the specific feature extractor for this model
        extractor = CombinedBowSyntacticExtractor(
            max_features=max_features,
            ngram_range=ngram_range
        )
        # Initialize the base class with this extractor
        super().__init__(feature_extractor=extractor) # Pass extractor to base

        # --- Extract RF Hyperparameters ---
        # Use values passed directly, or override from 'args' if provided.
        rf_n_estimators = getattr(args, 'n_estimators', n_estimators) if args else n_estimators
        rf_max_depth = getattr(args, 'max_depth', max_depth) if args else max_depth
        rf_random_state = getattr(args, 'random_state', random_state) if args else random_state
        # Extract other RF params from args if they exist, otherwise use defaults
        rf_min_samples_split = getattr(args, 'min_samples_split', min_samples_split) if args else min_samples_split
        rf_min_samples_leaf = getattr(args, 'min_samples_leaf', min_samples_leaf) if args else min_samples_leaf
        # Add extraction for other RF params like criterion, class_weight if needed from args

        # Log the parameters being used
        logger.info(f"Initializing RandomForestClassifier with: n_estimators={rf_n_estimators}, max_depth={rf_max_depth}, "
                    f"min_samples_split={rf_min_samples_split}, min_samples_leaf={rf_min_samples_leaf}, random_state={rf_random_state}")

        # Initialize the actual RF model with *only* the accepted parameters
        self.model = RandomForestClassifier(
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            min_samples_split=rf_min_samples_split,
            min_samples_leaf=rf_min_samples_leaf,
            random_state=rf_random_state,
            n_jobs=-1, # Use all available cores
            # DO NOT ADD **kwargs HERE
        )
        # Store the init params for saving/loading if needed (optional)
        self._init_params = {
             'max_features': max_features,
             'ngram_range': ngram_range,
             'n_estimators': rf_n_estimators,
             'max_depth': rf_max_depth,
             'random_state': rf_random_state,
             'min_samples_split': rf_min_samples_split,
             'min_samples_leaf': rf_min_samples_leaf
        }
        self.is_trained = False
        # feature_cols will be set by the base class extract_features during training
    # ++++++++++++++ END MODIFIED __init__ ++++++++++++++

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the Random Forest model."""
        if X is None or X.shape[0] == 0 or y is None or y.shape[0] == 0:
            logger.error("Cannot train Random Forest with empty features or labels.")
            return
        if X.shape[1] == 0: # Check if features are empty
             logger.error("Cannot train Random Forest with zero features.")
             return
        logger.info(f"Training {self.MODEL_NAME} with {X.shape[0]} samples, {X.shape[1]} features...")
        start_time = time.time()
        self.model.fit(X, y)
        self.is_trained = True # Mark as trained AFTER fitting
        end_time = time.time()
        logger.info(f"Training complete. Time taken: {end_time - start_time:.2f} seconds.")


    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the trained Random Forest model."""
        if not self.is_trained or self.model is None:
            raise RuntimeError(f"{self.MODEL_NAME} has not been trained yet.")
        # Dimension check - ensure prediction features match training features
        if hasattr(self.model, 'n_features_in_') and X.shape[1] != self.model.n_features_in_:
            raise ValueError(f"Prediction input feature count ({X.shape[1]}) does not match "
                             f"training feature count ({self.model.n_features_in_}).")

        logger.info(f"Predicting with {self.MODEL_NAME} on {X.shape[0]} samples...")
        predictions = self.model.predict(X)
        logger.info("Prediction complete.")
        return predictions

    # --- save/load need adjustment if using FeatureBasedBaselineModel ---
    # FeatureBasedBaselineModel handles saving/loading the extractor and feature_cols
    # We only need to save/load the sklearn model itself and potentially init params

    def save(self, directory: str, model_name_base: str) -> None:
        """Saves the trained RF model, the feature extractor, and feature columns."""
        if not self.is_trained:
            logger.warning(f"Attempting to save an untrained {self.MODEL_NAME} model.")
        # Let the base class handle saving common components (extractor, feature_cols)
        # super().save(directory, model_name_base) # If base class has save

        # Save the specific model components
        model_path = os.path.join(directory, f"{model_name_base}_rf_model.joblib")
        extractor_path = os.path.join(directory, f"{model_name_base}_extractor.joblib")
        metadata_path = os.path.join(directory, f"{model_name_base}_metadata.joblib")

        # Prepare metadata
        metadata = {
            'feature_cols': self.feature_cols, # Get from base class if it exists
            'is_trained': self.is_trained,
            'init_params': self._init_params # Save init params used
        }

        try:
             os.makedirs(directory, exist_ok=True)
             joblib.dump(self.model, model_path) # Save sklearn model
             joblib.dump(self.feature_extractor, extractor_path) # Save extractor instance
             joblib.dump(metadata, metadata_path) # Save metadata
             logger.info(f"Saved {self.MODEL_NAME} model state to {model_path}")
             logger.info(f"Saved {self.MODEL_NAME} extractor state to {extractor_path}")
             logger.info(f"Saved {self.MODEL_NAME} metadata to {metadata_path}")
        except Exception as e:
             logger.error(f"Error saving {self.MODEL_NAME} state: {e}", exc_info=True)
             # Clean up partial files
             if os.path.exists(model_path): os.remove(model_path)
             if os.path.exists(extractor_path): os.remove(extractor_path)
             if os.path.exists(metadata_path): os.remove(metadata_path)
             raise

    @classmethod
    def load(cls, directory: str, model_name_base: str) -> 'RandomForestBowSyntacticExperiment5':
        """Loads the RF model, feature extractor, and metadata."""
        model_path = os.path.join(directory, f"{model_name_base}_rf_model.joblib")
        extractor_path = os.path.join(directory, f"{model_name_base}_extractor.joblib")
        metadata_path = os.path.join(directory, f"{model_name_base}_metadata.joblib")

        logger.info(f"Loading {cls.MODEL_NAME} state from {directory} with base name {model_name_base}")
        if not all(os.path.exists(p) for p in [model_path, extractor_path, metadata_path]):
            raise FileNotFoundError(f"Required model files not found in {directory} for base name {model_name_base}")

        try:
             loaded_rf_model = joblib.load(model_path)
             loaded_extractor = joblib.load(extractor_path)
             loaded_metadata = joblib.load(metadata_path)
        except Exception as e:
             logger.error(f"Error loading files for {cls.MODEL_NAME}: {e}", exc_info=True)
             raise

        # Re-create the instance using saved init params
        init_params = loaded_metadata.get('init_params', {})
        # Need to handle potential missing keys in init_params if older model saved
        instance = cls(
            args=None, # Pass None for args during load
            max_features=init_params.get('max_features', 5000),
            ngram_range=init_params.get('ngram_range', (1, 1)),
            n_estimators=init_params.get('n_estimators', 100),
            max_depth=init_params.get('max_depth', None),
            random_state=init_params.get('random_state', 42),
            min_samples_split=init_params.get('min_samples_split', 2),
            min_samples_leaf=init_params.get('min_samples_leaf', 1)
        )

        # Assign loaded components
        instance.model = loaded_rf_model
        instance.feature_extractor = loaded_extractor
        instance.feature_cols = loaded_metadata.get('feature_cols') # Restore feature names
        instance.is_trained = loaded_metadata.get('is_trained', False) # Restore trained status

        # Post-load validation
        if not isinstance(instance.feature_extractor, CombinedBowSyntacticExtractor):
             logger.warning("Loaded extractor type mismatch.")
        if instance.feature_cols is None and instance.is_trained:
             logger.warning(f"Loaded trained {cls.MODEL_NAME} model but feature_cols list is missing.")
        if not isinstance(instance.model, RandomForestClassifier):
             logger.warning(f"Loaded model type mismatch: Expected RandomForestClassifier, got {type(instance.model)}")


        logger.info(f"Loaded {cls.MODEL_NAME}. Trained: {instance.is_trained}")
        return instance

# --- Evaluate method (if needed, likely handled by ExperimentTrainer calling base class evaluate) ---
# def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
#    """Evaluates the model on the test set."""
#    # Implementation using _evaluate_model_performance from baseline_base