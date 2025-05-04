# models/random_forest_bow_syntactic_experiment_5.py
import logging
import numpy as np
import pandas as pd
import time
from typing import List, Optional, Any # Ensure necessary imports

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack # To combine sparse matrices

# --- Updated Import ---
# Import FeatureExtractor from its new location
from .baseline_base import FeatureExtractor, SyntacticFeatureExtractor, FeatureBasedBaselineModel
# You might also need other base components depending on your structure
# from .baseline_base import FeatureBasedBaselineModel # If using the example base model
# --- End Updated Import ---

from utils.common import NLIModel # Keep this if RF model inherits directly from NLIModel
# from data.preprocessor import preprocess_text # Assuming you need preprocessing

logger = logging.getLogger(__name__)

# --- Feature Extractor for this specific experiment ---
# This class now correctly inherits from the FeatureExtractor defined in baseline_base.py
class CombinedBowSyntacticExtractor(FeatureExtractor):
    """Combines Bag-of-Words features with precomputed Syntactic features."""
    def __init__(self, max_features: Optional[int] = 5000, ngram_range: tuple = (1, 1)):
        self.bow_vectorizer = CountVectorizer(
            # preprocessor=preprocess_text, # Use your preprocessing function
            max_features=max_features,
            ngram_range=ngram_range,
            lowercase=True
        )
        # Use the SyntacticFeatureExtractor moved to baseline_base
        self.syntactic_extractor = SyntacticFeatureExtractor()
        self.bow_feature_names_ = None # To store BOW feature names after fitting
        self.syntactic_feature_names_ = None # To store Syntactic feature names

    def fit(self, data: pd.DataFrame, y: Optional[Any] = None):
        """Fit the BoW vectorizer."""
        logger.info("Fitting BoW vectorizer...")
        # Combine premise and hypothesis for BoW vocabulary fitting
        combined_text = data['premise'] + " " + data['hypothesis']
        self.bow_vectorizer.fit(combined_text)
        self.bow_feature_names_ = self.bow_vectorizer.get_feature_names_out()
        logger.info(f"BoW vectorizer fitted with {len(self.bow_feature_names_)} features.")
        # Also get syntactic feature names (requires the dataframe structure)
        self.syntactic_feature_names_ = self.syntactic_extractor.get_feature_columns(data)
        logger.info(f"Identified {len(self.syntactic_feature_names_)} syntactic features.")
        return self

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """Transform data into combined BoW and Syntactic features."""
        logger.info("Transforming data with BoW vectorizer...")
        premise_bow = self.bow_vectorizer.transform(data['premise'])
        hypothesis_bow = self.bow_vectorizer.transform(data['hypothesis'])
        # Combine BoW features (e.g., concatenate, difference, product - concatenating here)
        bow_features = hstack([premise_bow, hypothesis_bow]) # sparse matrix
        logger.info(f"BoW features shape: {bow_features.shape}")

        logger.info("Extracting syntactic features...")
        # Extract syntactic features using the dedicated extractor
        # Ensure feature_cols used here match those stored if predicting after load
        syntactic_features_np = self.syntactic_extractor.extract(data, self.syntactic_feature_names_)
        logger.info(f"Syntactic features shape: {syntactic_features_np.shape}")

        # Combine sparse BoW features with dense Syntactic features
        # Need to handle potential NaN in syntactic features if not already done
        syntactic_features_np = np.nan_to_num(syntactic_features_np) # Simple NaN handling
        combined_features = hstack([bow_features, syntactic_features_np]).tocsr() # combine as sparse
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
        if self.bow_feature_names_ is None or self.syntactic_feature_names_ is None:
             raise RuntimeError("Extractor must be fitted before calling extract/transform.")
        # Re-use the transform logic
        return self.transform(data)


    def get_feature_columns(self, data: pd.DataFrame = None) -> List[str]:
        """ Gets the combined list of feature names. Requires fitting first.
            'data' parameter is optional here as names are stored after fit.
        """
        if self.bow_feature_names_ is None or self.syntactic_feature_names_ is None:
            # If not fitted, maybe return expected prefixes? Or raise error?
            # raise RuntimeError("Extractor must be fitted before getting feature columns.")
             logger.warning("Extractor not fitted. Returning empty feature list.")
             return []

        # Prefix BoW features to distinguish premise/hypothesis if needed, or just combine
        premise_bow_names = [f"premise_{name}" for name in self.bow_feature_names_]
        hypothesis_bow_names = [f"hypothesis_{name}" for name in self.bow_feature_names_]
        # Return combined list
        return premise_bow_names + hypothesis_bow_names + self.syntactic_feature_names_


# --- Random Forest Model using the Combined Extractor ---
# Option 1: Inherit from NLIModel directly
# class RandomForestBowSyntacticExperiment5(NLIModel):
# Option 2: Inherit from a new FeatureBasedBaselineModel (Recommended if structure fits)
class RandomForestBowSyntacticExperiment5(FeatureBasedBaselineModel): # Inherit from base
    """Random Forest model using Bag-of-Words and Syntactic features."""
    MODEL_NAME = "RandomForest_BoW_Syntactic_Exp5"

    # Pass the specific extractor instance to the base class
    def __init__(self,
                 max_features: Optional[int] = 5000, # BoW max features
                 ngram_range: tuple = (1, 1), # BoW ngram range
                 n_estimators: int = 100,     # RF parameter
                 max_depth: Optional[int] = None, # RF parameter
                 random_state: int = 42,
                 **kwargs): # RF parameter
        # Initialize the specific feature extractor for this model
        extractor = CombinedBowSyntacticExtractor(
            max_features=max_features,
            ngram_range=ngram_range
        )
        # Initialize the base class with this extractor
        super().__init__(feature_extractor=extractor) # Pass extractor to base

        # Initialize the actual RF model
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1, # Use all available cores
            **kwargs
        )
        self.is_trained = False
        # feature_cols will be set by the base class extract_features during training

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the Random Forest model."""
        if X is None or X.shape[0] == 0 or y is None or y.shape[0] == 0:
            logger.error("Cannot train Random Forest with empty features or labels.")
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
        # Dimension check might be needed if features can vary, but handled by base class extract_features
        logger.info(f"Predicting with {self.MODEL_NAME} on {X.shape[0]} samples...")
        predictions = self.model.predict(X)
        logger.info("Prediction complete.")
        return predictions

    def save(self, filepath: str) -> None:
        """Save the model and extractor state."""
        if not self.is_trained:
            logger.warning(f"Attempting to save an untrained {self.MODEL_NAME} model.")
        # Save necessary components: sklearn model, extractor state, feature columns
        state = {
            'model_state': self.model,
            'extractor_state': self.feature_extractor, # Save the fitted extractor
            'feature_cols': self.feature_cols, # Save feature names used for training
            'is_trained': self.is_trained,
            # Add any other parameters needed for re-initialization if necessary
            # 'init_params': {'n_estimators': self.model.n_estimators, ...}
        }
        # Consider using joblib for efficient saving of sklearn objects
        import joblib
        joblib.dump(state, filepath)
        logger.info(f"Saved {self.MODEL_NAME} state to {filepath}")

    @classmethod
    def load(cls, filepath: str, feature_extractor: Optional[FeatureExtractor] = None) -> 'RandomForestBowSyntacticExperiment5':
        """Load the model and extractor state."""
        import joblib
        logger.info(f"Loading {cls.MODEL_NAME} state from {filepath}")
        state = joblib.load(filepath)

        # Re-create the instance - need init params if not saved in state
        # This part is tricky - how were max_features, ngram_range etc stored?
        # Assuming they are part of the saved extractor state or defaults are okay.
        instance = cls() # Use default init params, will be overwritten

        instance.model = state['model_state']
        instance.feature_extractor = state['extractor_state']
        instance.feature_cols = state['feature_cols']
        instance.is_trained = state['is_trained']

        # Verify the loaded extractor type if needed
        if not isinstance(instance.feature_extractor, CombinedBowSyntacticExtractor):
             logger.warning("Loaded extractor type does not match expected CombinedBowSyntacticExtractor.")

        if instance.feature_cols is None and instance.is_trained:
             logger.warning(f"Loaded trained {cls.MODEL_NAME} model from {filepath} but feature_cols list is missing.")

        logger.info(f"Loaded {cls.MODEL_NAME}. Trained: {instance.is_trained}")
        return instance

# --- (Keep or adapt evaluate function as needed) ---
# Example evaluation function (similar to _evaluate_model_performance in base)
# def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
#    """Evaluates the model on the test set."""
#    if not self.is_trained:
#        raise RuntimeError("Model must be trained before evaluation.")
#    y_pred = self.predict(X_test)
#    return _evaluate_model_performance(y_test, y_pred, self.MODEL_NAME) # Use the base evaluator