# File: models/knn_bow_hand_crafted_syntactic_features_experiment_2.py
import time
import os
import joblib
import logging
from typing import Dict, List, Optional, Any # Added Any

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from abc import ABC # Import ABC

# Assuming config is importable and defines necessary constants
try:
    import config
except ModuleNotFoundError:
    print("Warning: config.py not found. Using placeholder values.")
    class ConfigPlaceholder:
        RANDOM_SEED = 42 # KNN doesn't use random_state, but other parts might
        KNN_N_NEIGHBORS = 5
        KNN_WEIGHTS = 'uniform'
        KNN_METRIC = 'minkowski'
        # Removed BOW related config placeholders
        # SYNTACTIC_FEATURE_COLUMNS are now detected dynamically
    config = ConfigPlaceholder()

# Make sure necessary components can be imported from baseline_base and utils
try:
    # Import necessary components similar to Experiment 1
    from .baseline_base import (
        SyntacticFeatureExtractor, clean_dataset, _evaluate_model_performance,
        SimpleParquetLoader, _handle_nan_values
    )
    from utils.common import NLIModel # NLIModel is the key interface
except (ModuleNotFoundError, ImportError) as e:
    print(f"Warning: Failed to import necessary components: {e}. Defining dummies.")
    class NLIModel: pass # Placeholder if utils.common is not found
    class SyntacticFeatureExtractor:
        def get_feature_columns(self, data): return []
        def extract(self, data, feature_cols=None): return np.array([])
    def clean_dataset(df): return df, None
    def _evaluate_model_performance(model, X, y): return 0.0, {}
    class SimpleParquetLoader:
        def load_data(self, *args): return None
    def _handle_nan_values(df, context): return df


logger = logging.getLogger(__name__)

class KnnBowSyntacticExperiment2(NLIModel, ABC): # Inherit from NLIModel and ABC
    """
    Experiment 2: k-Nearest Neighbors + Hand-crafted Syntactic Features ONLY
    Uses pre-computed hand-crafted syntactic features loaded from parquet files
    and feeds them into a KNN classifier after scaling.
    (Removed BoW component to align with ExperimentTrainer structure).
    """
    def __init__(self, args=None, **kwargs): # Accept args like Experiment 1
        self.args = args # Store args if needed
        self.name = "Experiment 2: KNN + Syntactic Features"
        self.description = "KNN classifier using only hand-crafted syntactic features."
        self.feature_type = 'syntactic' # This model now ONLY uses syntactic features
        self.is_trained = False
        self.feature_cols: Optional[List[str]] = None  # To store feature names used during training

        # Initialize the feature extractor needed
        self.feature_extractor = SyntacticFeatureExtractor()

        # Define hyperparameters for KNN
        self.params = {
            'n_neighbors': getattr(args, 'n_neighbors', config.KNN_N_NEIGHBORS),
            'weights': getattr(args, 'weights', config.KNN_WEIGHTS),
            'metric': getattr(args, 'metric', config.KNN_METRIC)
            # Add other KNN params like 'p' for minkowski if needed from args or config
        }
        # Store the actual config used
        self.model_config = self.params.copy()

        # Define the simpler pipeline: Scale features -> Classify
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()), # Scaling is crucial for KNN
            ('classifier', KNeighborsClassifier(**self.params))
        ])

    def extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Extracts pre-computed syntactic features using the SyntacticFeatureExtractor.
        Stores feature column names during the first call (training).
        Uses stored column names for subsequent calls (prediction/evaluation).
        Handles NaN values in the extracted features.
        """
        if not isinstance(data, pd.DataFrame):
             raise TypeError(f"Input data must be a pandas DataFrame, got {type(data)}")

        if not self.is_trained or self.feature_cols is None:
            # Training phase: Discover and store feature columns
            self.feature_cols = self.feature_extractor.get_feature_columns(data)
            if not self.feature_cols:
                logger.warning(f"No syntactic feature columns identified by the extractor for model {self.name}. Returning empty array.")
                # Return shape (num_samples, 0)
                return np.empty((len(data), 0))
            logger.info(f"Training {self.name}: Storing {len(self.feature_cols)} feature columns.")
            # Extract using discovered columns
            features = self.feature_extractor.extract(data, self.feature_cols)
        else:
            # Prediction/Evaluation phase: Use stored feature columns
            if self.feature_cols is None: # Should not happen if trained, but safety check
                 raise RuntimeError("Model is marked as trained, but feature_cols is not set.")
            logger.debug(f"Predict/Evaluate {self.name}: Using stored {len(self.feature_cols)} feature columns.")
            # Extract using stored columns. Extractor should handle missing cols by filling with 0.
            features = self.feature_extractor.extract(data, self.feature_cols)

        # Ensure features is a 2D numpy array
        if features is None: features = np.empty((len(data), 0))
        if features.ndim == 1: features = features.reshape(-1, 1) # Reshape if only one feature column

        # Handle NaN values in the extracted numeric features
        if features.size > 0 and np.isnan(features).any():
            # Use helper function, needs DataFrame input
            # Create temporary DataFrame for NaN handling
            temp_df = pd.DataFrame(features, columns=self.feature_cols, index=data.index[:len(features)]) # Align index
            temp_df_filled = _handle_nan_values(temp_df, context=f"extract_features ({'train' if not self.is_trained else 'predict'})")
            features = temp_df_filled.values
            logger.debug(f"NaN values handled in extracted features for {self.name}.")
        elif features.size == 0:
             logger.warning(f"{self.name}: No features extracted, skipping NaN handling.")
             # Ensure correct shape (n_samples, 0)
             features = np.empty((len(data), 0))


        # Final shape check
        if features.shape[0] != len(data):
             logger.warning(f"Row count mismatch after feature extraction/NaN handling: Expected {len(data)}, Got {features.shape[0]}. This might indicate data loss.")
             # Attempt to create array with correct number of rows if features are empty
             if features.shape[1] == 0:
                 features = np.empty((len(data), 0))


        return features

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Trains the KNN pipeline (Scaler + Classifier)."""
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
             raise TypeError("Input X and y must be NumPy arrays.")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Feature shape {X.shape} and label shape {y.shape} mismatch.")
        # Handle case with zero features (important check!)
        if X.shape[1] == 0:
             logger.warning(f"Attempting to train {self.name} with zero features. Training skipped. Model will predict based on neighbors in 0-dim space (may error or give default).")
             # Mark as trained conceptually, though the classifier won't learn much.
             self.is_trained = True
             # Ensure feature_cols is empty list if shape[1] is 0
             if self.feature_cols is None: self.feature_cols = []
             return

        logger.info(f"Training {self.name} with {X.shape[0]} samples and {X.shape[1]} features...")
        start_time = time.time()
        # Ensure feature_cols are set (should be done by extract_features call before train)
        if self.feature_cols is None:
            logger.error("Feature columns (feature_cols) were not set before training. This should not happen.")
            # As fallback, try to infer from X, though this is risky
            self.feature_cols = [f"feature_{i}" for i in range(X.shape[1])]
            # raise RuntimeError("feature_cols not set before training call.") # Stricter alternative

        try:
            self.pipeline.fit(X, y)
            self.is_trained = True
            train_time = time.time() - start_time
            logger.info(f"{self.name} training complete in {train_time:.2f} seconds.")
        except Exception as e:
            logger.error(f"Error during pipeline fitting for {self.name}: {e}", exc_info=True)
            self.is_trained = False # Mark as not trained if error occurs
            raise # Re-raise the exception

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Makes predictions using the trained KNN pipeline."""
        if not self.is_trained:
            raise RuntimeError(f"Model {self.name} has not been trained yet.")
        if not isinstance(X, np.ndarray):
             raise TypeError("Input X must be a NumPy array.")

        # Validate input feature count against training feature count
        expected_features = len(self.feature_cols) if self.feature_cols is not None else -1
        if expected_features == -1:
             raise RuntimeError("Model is trained, but feature_cols is not set.")

        # Handle the case where the model was trained on zero features
        if expected_features == 0:
            if X.shape[1] != 0:
                raise ValueError("Model was trained with 0 features, but prediction input has features.")
            logger.warning(f"Predicting with {self.name} which was trained on 0 features. Returning empty array or default prediction.")
            # KNN might error here, or predict based on the training 'y' distribution if possible,
            # or we can return a default. Returning empty array might be safest if downstream handles it.
            return np.array([], dtype=int) # Return empty array of appropriate type

        # Normal case: Check feature count match
        if X.shape[1] != expected_features:
            raise ValueError(
                f"Prediction input feature count ({X.shape[1]}) does not match "
                f"training feature count ({expected_features})."
            )

        logger.debug(f"Predicting with {self.name} on {X.shape[0]} samples...")
        try:
            predictions = self.pipeline.predict(X)
            logger.debug("Prediction finished.")
            return predictions
        except Exception as e:
            logger.error(f"Error during pipeline prediction for {self.name}: {e}", exc_info=True)
            raise # Re-raise the exception


    def evaluate(self, dataset_name: str, split: str, suffix: str) -> Dict[str, Any]:
        """Evaluates the trained model on a given dataset split using pre-computed features."""
        if not self.is_trained:
            logger.error(f"Cannot evaluate model {self.name}: Model is not trained.")
            return {} # Return empty dict to indicate failure

        logger.info(f"Evaluating {self.name} on {dataset_name}/{split}/{suffix}...")
        data_loader = SimpleParquetLoader()
        eval_df = None
        try:
            # Load the pre-computed features for the evaluation split
            eval_df = data_loader.load_data(self, dataset_name, split, suffix) # Pass self if loader needs it
            if eval_df is None or eval_df.empty:
                raise ValueError(f"Failed to load evaluation feature data for {dataset_name}/{split}/{suffix}")
            logger.info(f"Loaded {len(eval_df)} rows for evaluation from Parquet.")
        except FileNotFoundError:
            logger.error(f"Evaluation feature file not found for {dataset_name}/{split}/{suffix}. Cannot evaluate.")
            return {}
        except Exception as e:
            logger.error(f"Error loading evaluation data: {e}", exc_info=True)
            return {}

        # Clean data (primarily handles labels)
        cleaned_data = clean_dataset(eval_df)
        if cleaned_data is None:
            logger.error("Evaluation data became empty or invalid after cleaning.")
            return {}
        df_cleaned, y_true_numeric = cleaned_data

        if df_cleaned.empty or y_true_numeric is None or len(y_true_numeric) == 0:
            logger.warning("No valid evaluation samples left after cleaning.")
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'eval_time': 0.0} # Return zero metrics

        # Extract features using the stored feature columns from training
        X_eval = None
        try:
            logger.debug(f"Extracting evaluation features using stored columns: {self.feature_cols}")
            X_eval = self.extract_features(df_cleaned) # Use the instance's extract_features

            # --- Alignment Check ---
            # Ensure y_true_numeric aligns with X_eval if extract_features implicitly dropped rows (e.g., via NaN handling if it modified df_cleaned index)
            # This is less likely if _handle_nan_values just fills, but essential if rows could be dropped.
            if X_eval.shape[0] != len(y_true_numeric):
                # Check if df_cleaned index was somehow modified and use it to realign y_true_numeric
                if X_eval.shape[0] == len(df_cleaned):
                    # If extract_features returned features corresponding to the *final* state of df_cleaned
                    # (after potential modifications), then y_true_numeric should also correspond to that.
                    # This assumes clean_dataset returns labels aligned with its output df.
                     logger.warning(f"Shape mismatch: X_eval ({X_eval.shape[0]}) vs y_true ({len(y_true_numeric)}). Assuming alignment with df_cleaned ({len(df_cleaned)}).")
                     # No explicit realignment needed if y_true_numeric already matches df_cleaned output from clean_dataset.

                else:
                    # If counts don't match even df_cleaned, something is wrong.
                    raise ValueError(
                        f"Evaluation feature/label count mismatch after extraction: "
                        f"X_eval={X_eval.shape[0]}, y_true={len(y_true_numeric)}, df_cleaned={len(df_cleaned)}"
                    )

            # Handle case where model expects 0 features
            if X_eval.shape[1] == 0 and len(self.feature_cols) == 0:
                 logger.warning("Evaluating model trained on 0 features with 0 features.")
                 # Proceed to evaluation, performance will likely be poor or based on majority class if predict handles it.
            elif X_eval.shape[1] != len(self.feature_cols):
                 # This check should ideally be inside predict, but double-check here
                 raise ValueError(f"Evaluation feature dimension ({X_eval.shape[1]}) != training dimension ({len(self.feature_cols)})")


        except Exception as e:
            logger.error(f"Error extracting features during evaluation: {e}", exc_info=True)
            return {}

        # Perform evaluation using the helper function
        # Ensure y_true_numeric is a numpy array
        y_true_np = y_true_numeric if isinstance(y_true_numeric, np.ndarray) else np.array(y_true_numeric)

        # Use self (the loaded & trained model instance) for evaluation
        eval_time, metrics = _evaluate_model_performance(self, X_eval, y_true_np)
        metrics['eval_time'] = eval_time # Add timing info

        logger.info(f"Evaluation complete for {self.name} on {split}. Metrics: {metrics}")
        return metrics

    def get_pipeline(self):
        """Returns the scikit-learn pipeline object."""
        return self.pipeline

    def get_params(self) -> Dict[str, Any]:
        """Returns the parameters for the model (pipeline parameters)."""
        if not self.pipeline:
             return {}
        # Get parameters specifically for the classifier step
        clf_params = {f'classifier__{k}': v for k, v in self.params.items()}
        # Get parameters for the scaler step (optional, usually defaults are fine)
        scaler_params = {k: v for k, v in self.pipeline.named_steps['scaler'].get_params().items()}
        scaler_params_prefixed = {f'scaler__{k}': v for k, v in scaler_params.items()}

        # Combine parameters
        all_params = {**scaler_params_prefixed, **clf_params}
        return all_params

    # --- ADDED save AND load METHODS (adapted from Experiment 1) ---
    def save(self, directory: str, model_name: str) -> None:
        """Saves the trained pipeline and feature column names."""
        if not self.is_trained:
            logger.warning(f"Model {self.name} is not trained. Saving may result in an untrained model state.")
            # Decide if you want to prevent saving untrained models
            # raise RuntimeError("Cannot save an untrained model.")

        os.makedirs(directory, exist_ok=True)
        pipeline_path = os.path.join(directory, f"{model_name}_pipeline.joblib")
        feature_cols_path = os.path.join(directory, f"{model_name}_feature_cols.joblib")

        try:
            joblib.dump(self.pipeline, pipeline_path)
            joblib.dump(self.feature_cols, feature_cols_path)  # Save the list of feature names
            logger.info(f"{self.name} pipeline saved to {pipeline_path}")
            logger.info(f"{self.name} feature columns saved to {feature_cols_path}")
        except Exception as e:
            logger.error(f"Error saving model {self.name} to {directory}: {e}", exc_info=True)
            # Optional: clean up partially saved files if error occurs
            if os.path.exists(pipeline_path): os.remove(pipeline_path)
            if os.path.exists(feature_cols_path): os.remove(feature_cols_path)
            raise  # Re-raise the exception

    @classmethod
    def load(cls, directory: str, model_name: str) -> 'KnnBowSyntacticExperiment2':
        """Loads the pipeline and feature columns."""
        pipeline_path = os.path.join(directory, f"{model_name}_pipeline.joblib")
        feature_cols_path = os.path.join(directory, f"{model_name}_feature_cols.joblib")

        if not os.path.exists(pipeline_path) or not os.path.exists(feature_cols_path):
            raise FileNotFoundError(
                f"Pipeline ({pipeline_path}) or feature columns ({feature_cols_path}) file not found in directory: {directory}"
            )

        try:
            loaded_pipeline = joblib.load(pipeline_path)
            loaded_feature_cols = joblib.load(feature_cols_path)

            # Create a new instance of the class.
            # Pass None for args, as hyperparameters are loaded with the pipeline.
            instance = cls(args=None)

            # Assign the loaded components
            instance.pipeline = loaded_pipeline
            instance.feature_cols = loaded_feature_cols
            instance.is_trained = True  # Assume loaded model is trained

            # Optional: Restore hyperparameters from the loaded pipeline if needed elsewhere
            loaded_clf_params = loaded_pipeline.named_steps['classifier'].get_params()
            instance.params.update({k: v for k, v in loaded_clf_params.items() if k in instance.params})
            instance.model_config = instance.params.copy()


            logger.info(f"{cls.__name__} loaded successfully from {directory} using base name {model_name}")
            return instance
        except Exception as e:
            logger.error(f"Error loading model {cls.__name__} from {directory}: {e}", exc_info=True)
            raise  # Re-raise the exception
    # --- End save/load ---