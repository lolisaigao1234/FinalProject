# Modify in IS567FP/models/random_forest_bow_syntactic_experiment_5.py

import logging
import os
from typing import Tuple, Optional, Any, Dict, List # Added Dict, Tuple, Optional, List
import pandas as pd
import numpy as np
import joblib
import time # Added time

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted

# Import necessary components
from utils.common import NLIModel
from utils.database import DatabaseHandler # Added
from .baseline_base import clean_dataset, _evaluate_model_performance # Added

logger = logging.getLogger(__name__)

class CombinedBowSyntacticExtractor(FeatureExtractor):
    # ... (keep existing implementation) ...
    def get_feature_columns(self, data: pd.DataFrame) -> List[str]:
        """Get all relevant feature columns (BoW/lexical + syntactic)."""
        lexical_cols = filter_lexical_features(data)
        syntactic_cols = filter_syntactic_features(data)
        combined_cols = sorted(list(set(lexical_cols + syntactic_cols)))
        logger.debug(f"Combined BoW+Syntactic extractor identified {len(combined_cols)} columns.")
        final_cols = [col for col in combined_cols if col not in ['label', 'gold_label', 'pair_id']]
        return final_cols

    def extract(self, data: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
        """Extracts the specified columns, handling potential missing ones."""
        missing_cols = set(feature_cols) - set(data.columns)
        if missing_cols:
            logger.warning(f"Combined BoW+Syntactic extractor: Missing columns {missing_cols} during extraction. Filling with 0.")
            data_copy = data.copy()
            for col in missing_cols:
                data_copy[col] = 0
            return data_copy[feature_cols].values
        else:
            return data[feature_cols].values


class RandomForestBowSyntacticExperiment5(NLIModel):
    # ... (keep __init__ and other methods as they are) ...
    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None, random_state: int = 42, **kwargs):
        # ... (existing init code) ...
        self.db_handler = DatabaseHandler() # Add db_handler instance
        # ... (rest of init) ...
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


    def _load_and_prepare_data(self, dataset: str, split: str, suffix: str) -> Optional[Tuple[pd.DataFrame, np.ndarray]]:
        """Loads precomputed features, handles NaNs, cleans, and returns features+labels."""
        logger.info(f"Exp5: Loading features for {dataset}/{split}/{suffix}")
        feature_type_base = f"features_lexical_syntactic_{suffix}" # Expecting combined features
        feature_table_name = f"{dataset}_{split}_{feature_type_base}"

        try:
            features_df = self.db_handler.load_dataframe(dataset, split, feature_table_name)
            if features_df.empty:
                logger.error(f"Loaded empty features DataFrame for {feature_table_name}.")
                return None
            features_df = _handle_nan_values(features_df, f"{dataset}/{split} features")
        except Exception as e:
            logger.error(f"Failed to load features from {feature_table_name}: {e}", exc_info=True)
            return None

        clean_result = clean_dataset(features_df)
        if not clean_result:
            logger.error(f"Feature data for {split} is invalid after cleaning.")
            return None
        features_df_clean, y_labels = clean_result

        # Return the cleaned DataFrame (needed for feature extraction step) and labels
        return features_df_clean, y_labels

    # --- MODIFIED train METHOD ---
    def train(self, train_dataset: str, train_split: str, train_suffix: str,
              val_dataset: Optional[str] = None, val_split: Optional[str] = None, val_suffix: Optional[str] = None) -> Dict[str, Any]:
        """
        Loads data, extracts features, trains the RandomForestClassifier, and optionally evaluates.
        Signature now matches the call from BaselineTrainer.
        """
        logger.info(f"Starting Exp5 training for {train_dataset}/{train_split}/{train_suffix}")
        start_time = time.time()

        # 1. Load and prepare training data
        train_prep_result = self._load_and_prepare_data(train_dataset, train_split, train_suffix)
        if not train_prep_result:
            return {'error': 'Training data loading/preparation failed.'}
        X_train_df, y_train = train_prep_result

        # 2. Extract features (this also sets self.feature_cols)
        logger.info("Extracting features from training data...")
        try:
            # Use the instance's extract_features method, which will discover/store columns
            X_train_transformed = self.extract_features(X_train_df)
            if X_train_transformed is None or X_train_transformed.shape[0] == 0:
                 raise ValueError("Feature extraction returned empty array.")
            logger.info(f"Training features extracted. Shape: {X_train_transformed.shape}")
        except Exception as e:
            logger.error(f"Error extracting training features: {e}", exc_info=True)
            return {'error': 'Training feature extraction failed.'}

        # 3. Train the RandomForest model (using the original internal train logic)
        # Check for NaN/Infinity in features before training
        if not np.all(np.isfinite(X_train_transformed)):
            logger.error("Non-finite values (NaN or Infinity) found in training features (X_train_transformed). Cannot train Random Forest.")
            return {'error': 'Non-finite values in training features.'}

        logger.info("Training Random Forest model...")
        self.model.fit(X_train_transformed, y_train)
        self.is_trained = True # Mark as trained *after* successful fit
        train_time = time.time() - start_time
        logger.info(f"Model training completed in {train_time:.2f}s.")

        # 4. Evaluate on validation data (if provided)
        eval_metrics = {}
        eval_time = 0.0
        if val_dataset and val_split and val_suffix:
            logger.info(f"Loading and evaluating on validation data: {val_dataset}/{val_split}/{val_suffix}")
            val_prep_result = self._load_and_prepare_data(val_dataset, val_split, val_suffix)
            if val_prep_result:
                X_val_df, y_val = val_prep_result
                try:
                    logger.info("Extracting features from validation data...")
                    # Use extract_features again, it will use stored self.feature_cols now
                    X_val_transformed = self.extract_features(X_val_df)
                    logger.info("Evaluating model on validation data...")
                    # Pass self directly to the evaluator as it implements predict
                    eval_time, metrics = _evaluate_model_performance(self, X_val_transformed, y_val)
                    eval_metrics = metrics
                    eval_metrics['eval_time'] = eval_time
                except Exception as e:
                    logger.error(f"Error during validation evaluation: {e}", exc_info=True)
                    eval_metrics = {'error': 'Validation evaluation failed.'}
            else:
                logger.warning("Failed to load/prepare validation data. Skipping validation.")
                eval_metrics = {'warning': 'Validation data failed to load.'}
        else:
            logger.info("No validation data provided. Skipping validation.")

        results = {
            'train_time': train_time,
            **eval_metrics
        }
        return results # Return evaluation results

    # --- extract_features, predict, save, load methods remain mostly unchanged ---
    # Ensure extract_features sets self.feature_cols correctly during the first call (training)
    def extract_features(self, data: pd.DataFrame) -> np.ndarray:
        # ... (Keep existing implementation from random_forest_bow_syntactic_experiment_5.py) ...
        # This implementation correctly discovers columns on first call (train)
        # and reuses them on subsequent calls (predict/eval).
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame.")

        if not self.is_trained or self.feature_cols is None:
            logger.debug("Experiment 5: Discovering feature columns during training/first extraction.")
            self.feature_cols = self.feature_extractor.get_feature_columns(data)
            if not self.feature_cols:
                 raise ValueError("Experiment 5: No features extracted during training phase.")
            logger.info(f"Experiment 5: Storing {len(self.feature_cols)} feature columns for training.")
            features = self.feature_extractor.extract(data, self.feature_cols)
        else:
            logger.debug(f"Experiment 5: Extracting features for prediction using stored {len(self.feature_cols)} columns.")
            features = self.feature_extractor.extract(data, self.feature_cols)

        if features.dtype == 'object' or pd.DataFrame(features).isna().any().any():
            logger.warning("NaNs or non-numeric types detected after feature extraction. Applying fillna(0).")
            features = pd.DataFrame(features, columns=self.feature_cols).fillna(0).values

        return features.astype(np.float32) # Ensure float type for RF


    def predict(self, X: np.ndarray) -> np.ndarray:
        # ... (Keep existing implementation) ...
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
        # ... (Keep existing implementation) ...
        model_path = f"{filepath}.joblib" # Save directly as joblib
        logger.info(f"Saving Experiment 5 Random Forest model state to {model_path}")

        if not self.is_trained:
            logger.warning("Attempting to save an untrained Random Forest model.")
        # Ensure feature_cols is set before saving metadata if trained
        if self.is_trained and self.feature_cols is None:
             logger.error("Model is marked trained but feature_cols is None. Cannot save feature list.")
             # Decide how to handle: save anyway, raise error, or try re-extracting?
             # Saving without feature_cols might break loading/prediction later.
             # For now, we'll log the error and save what we have.
             # A better approach might be to ensure feature_cols is always set after training.

        model_data = {
            'model_state': self.model,
            'is_trained': self.is_trained,
            'feature_cols': self.feature_cols, # Save the list of columns used
            'hyperparameters': { # Store hyperparameters
                 'n_estimators': self.n_estimators,
                 'max_depth': self.max_depth,
                 'random_state': self.random_state,
                 **self.model_kwargs
            }
        }
        try:
            joblib.dump(model_data, model_path)
            logger.info(f"Successfully saved model data to {model_path}")
        except Exception as e:
             logger.error(f"Error saving Random Forest model data to {model_path}: {e}", exc_info=True)


    @classmethod
    def load(cls, filepath: str, feature_extractor: Optional[Any] = None) -> 'RandomForestBowSyntacticExperiment5':
        # ... (Keep existing implementation) ...
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
        instance.feature_cols = model_data['feature_cols'] # Load feature columns

        if instance.feature_cols is None and instance.is_trained:
             logger.warning(f"Loaded trained Random Forest model from {model_path} but feature_cols list is missing.")

        logger.info(f"Successfully loaded Experiment 5 Random Forest model. Trained: {instance.is_trained}")
        return instance