# File: IS567FP/models/cross_eval_syntactic_experiment_7.py
import logging
import os
import time
from typing import Dict, Any, Optional, Tuple, List

import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

from utils.common import get_split_sizes # Import the new utility
from .baseline_base import clean_dataset, _evaluate_model_performance, _handle_nan_values, SimpleParquetLoader, filter_syntactic_features

logger = logging.getLogger(__name__)

# --- Feature Filtering (from previous correct version) ---
def filter_dependency_features(df: pd.DataFrame) -> List[str]:
    if not isinstance(df, pd.DataFrame):
        logger.error("filter_dependency_features expects a Pandas DataFrame.")
        return []
    dep_cols = [col for col in df.columns if any(p in col for p in ["_dep_", "deprel_"])]
    logger.debug(f"Exp7: Identified {len(dep_cols)} dependency columns.")
    return dep_cols

def filter_constituency_features(df: pd.DataFrame) -> List[str]:
    if not isinstance(df, pd.DataFrame):
        logger.error("filter_constituency_features expects a Pandas DataFrame.")
        return []
    const_cols = [col for col in df.columns if any(p in col for p in ["_const_"])]
    logger.debug(f"Exp7: Identified {len(const_cols)} constituency columns.")
    return const_cols


class CrossEvalSyntacticExperiment7:
    def __init__(self, args: object):
        self.args = args
        self.dataset_name = args.dataset
        self.total_sample_size = getattr(args, 'sample_size', None)

        # Get actual split sizes using the utility function
        self.train_actual_size, self.val_actual_size, self.test_actual_size = get_split_sizes(self.total_sample_size)
        
        # Suffix for saving models/overall config identification should use total_sample_size
        self.config_identifier_suffix = f"total_sample{self.total_sample_size}" if self.total_sample_size is not None else "full"
        
        self.save_dir = self._get_save_directory()
        os.makedirs(self.save_dir, exist_ok=True)
        self.loader = SimpleParquetLoader()
        self.results: Dict[str, Any] = {}

        # Replace SVM parameters with DT and KNN parameters
        self.dt_max_depth = getattr(args, 'max_depth', 10)
        self.dt_min_samples_split = getattr(args, 'min_samples_split', 2)
        self.knn_n_neighbors = getattr(args, 'n_neighbors', 5)
        self.knn_weights = getattr(args, 'weights', 'uniform')
        self.lr_C = getattr(args, 'C', 1.0)
        self.lr_max_iter = getattr(args, 'max_iter', 1000)
        self.random_state = 42

    def _get_save_directory(self) -> str:
        base_dir = os.path.join(
            os.path.dirname(__file__),
            '..', 'saved_models', 'experiment_7',
            self.dataset_name,
            self.config_identifier_suffix # Use the config identifier
        )
        return base_dir

    def _get_file_suffix_for_split_load(self, split: str) -> str:
        """Determines the file suffix (e.g., 'sample10') for loading a specific split's features."""
        if self.total_sample_size is None: # Not sampled, use 'full'
            return "full"
        
        actual_size_for_split: Optional[int] = None
        if split == "train":
            actual_size_for_split = self.train_actual_size
        elif split == "validation":
            actual_size_for_split = self.val_actual_size
        elif split == "test":
            actual_size_for_split = self.test_actual_size
        else:
            logger.error(f"Unknown split '{split}' for suffix determination. Defaulting to 'full'.")
            return "full"
        
        if actual_size_for_split is None or actual_size_for_split <= 0:
             logger.warning(f"Actual size for split '{split}' is {actual_size_for_split}. Files for this split might be 'full' or not exist if size is 0. Defaulting to 'full' for safety.")
             # If a split legitimately has 0 samples, its feature file shouldn't exist.
             # The calling code should handle None return from _load_and_prepare_features.
             # Returning "full" here is a fallback if actual_size is unexpectedly None/0.
             # A more robust approach might be to return a suffix that guarantees FileNotFoundError
             # if the size implies no file should exist (e.g. if actual_size_for_split == 0).
             # For now, this relies on the feature extractor not creating files for 0-sample splits.
             return "full" 
        return f"sample{actual_size_for_split}"

    # --- Start of _load_and_prepare_features (incorporating previous Attribute Error fix) ---
    def _load_and_prepare_features(self, split: str) -> Optional[Tuple[pd.DataFrame, np.ndarray]]:
        current_split_file_suffix = self._get_file_suffix_for_split_load(split)
        
        logger.info(f"Exp7: Attempting to load features for {self.dataset_name}/{split} with file suffix '{current_split_file_suffix}'")
        
        features_df = None
        try:
            features_df = self.loader.load_data(self, self.dataset_name, split, current_split_file_suffix) 
            
            if features_df is None or features_df.empty:
                logger.warning(f"SimpleParquetLoader returned empty or None for {self.dataset_name}/{split}/{current_split_file_suffix}.")
                return None 
            
            features_df = _handle_nan_values(features_df, f"Exp7 features for {self.dataset_name}/{split}")

        except FileNotFoundError:
            logger.error(f"Exp7: Feature file not found by SimpleParquetLoader for {self.dataset_name}/{split} with suffix '{current_split_file_suffix}'.")
            return None
        except Exception as e:
            logger.error(f"Exp7: Error loading features for {self.dataset_name}/{split} (suffix '{current_split_file_suffix}'): {e}", exc_info=True)
            return None

        clean_result = clean_dataset(features_df)
        if not clean_result:
            logger.error(f"Exp7: Feature data for {split} (suffix '{current_split_file_suffix}') invalid after cleaning.")
            return None
        features_df_clean, y_labels = clean_result

        if features_df_clean.empty:
            logger.warning(f"Exp7: No valid samples after cleaning for {split} (suffix '{current_split_file_suffix}').")
            return None

        syntactic_cols = filter_syntactic_features(features_df_clean)
        if not syntactic_cols:
            logger.error(f"Exp7: No syntactic columns in data for {split} (suffix '{current_split_file_suffix}').")
            return None 

        logger.info(f"Exp7: Using {len(syntactic_cols)} syntactic features for {split} split.")
        syntactic_features_df = features_df_clean[syntactic_cols]

        if syntactic_features_df.shape[0] != len(y_labels):
             logger.error(f"Exp7: Mismatch: syntactic DF ({syntactic_features_df.shape[0]}) vs labels ({len(y_labels)}) for {split} (suffix '{current_split_file_suffix}').")
             return None

        return syntactic_features_df, y_labels
    # --- End of _load_and_prepare_features ---

    # --- _train_and_evaluate_classifier (from previous correct version, ensure it's present) ---
    def _train_and_evaluate_classifier(self, model_instance: Any, X_train: np.ndarray, y_train: np.ndarray,
                                      X_val: Optional[np.ndarray], y_val: Optional[np.ndarray],
                                      model_desc: str) -> Dict[str, Any]:
        logger.info(f"--- Training {model_desc} ---")
        start_time = time.time()

        if not isinstance(X_train, np.ndarray): X_train = np.asarray(X_train)
        if not np.all(np.isfinite(X_train)):
            logger.warning(f"Non-finite values (NaN/inf) in {model_desc} training data X_train. Filling with 0.")
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        
        y_train = np.asarray(y_train).ravel()

        model_instance.fit(X_train, y_train)
        train_time = time.time() - start_time
        logger.info(f"Training complete in {train_time:.2f}s")

        eval_metrics = {}
        eval_time = 0.0
        if X_val is not None and y_val is not None and X_val.shape[0] > 0:
            logger.info(f"Evaluating {model_desc} on validation data ({X_val.shape[0]} samples)...")
            if not isinstance(X_val, np.ndarray): X_val = np.asarray(X_val)
            if not np.all(np.isfinite(X_val)):
                 logger.warning(f"Non-finite values (NaN/inf) in {model_desc} validation data X_val. Filling with 0.")
                 X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
            
            y_val = np.asarray(y_val).ravel()

            class PredictWrapper:
                def __init__(self, model): self.model = model
                def predict(self, X): return self.model.predict(X)

            eval_time, metrics = _evaluate_model_performance(PredictWrapper(model_instance), X_val, y_val)
            eval_metrics = metrics
            eval_metrics['eval_time'] = eval_time
            logger.info(f"Validation Metrics for {model_desc}: {metrics}")
        else:
            logger.info(f"Skipping validation for {model_desc} (X_val or y_val is None or empty / X_val has 0 samples).")

        model_filename = f"{model_desc.replace(' ', '_').lower()}.joblib"
        model_path = os.path.join(self.save_dir, model_filename)
        try:
            joblib.dump(model_instance, model_path)
            logger.info(f"Saved {model_desc} model to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model {model_desc} to {model_path}: {e}", exc_info=True)

        return {**eval_metrics, 'train_time': train_time}

    # --- run_experiment (incorporating previous Attribute Error fix) ---
    def run_experiment(self) -> Dict[str, Any]:
        logger.info(f"===== Starting Experiment 7 for {self.dataset_name} (Config: {self.config_identifier_suffix}) =====")

        train_prep = self._load_and_prepare_features("train")
        val_prep = self._load_and_prepare_features("validation")
        # test_prep = self._load_and_prepare_features("test") 

        if not train_prep:
            logger.error("Failed to load/prepare training data. Aborting Experiment 7.")
            return {"error": "Training data failed to load."}
        train_syntactic_df, y_train = train_prep 

        val_syntactic_df, y_val = (None, None)
        if val_prep:
            val_syntactic_df, y_val = val_prep
        else:
            logger.warning("Validation data failed to load via precomputed file. Will attempt to split from training data.")

        if val_syntactic_df is None or (isinstance(val_syntactic_df, pd.DataFrame) and val_syntactic_df.empty):
            if train_syntactic_df.shape[0] < 5: # Check rows
                logger.error("Not enough training samples to create a validation split.")
                return {"error": "Not enough data for train/validation split."}
            
            logger.info("Splitting validation set from training data (as precomputed validation data was not found/loaded).")
            y_train_1d = np.asarray(y_train).ravel()
            # Determine test_size for train_test_split (e.g. 0.1 / (0.8+0.1) of remaining data after train)
            # If train is 80% of total, and val is 10% of total, then val is 10/80 = 0.125 of train.
            # Or more simply, if val_actual_size was determined:
            relative_val_size = 0.2 # Default fallback if val_actual_size not useful here
            if self.train_actual_size and self.val_actual_size and (self.train_actual_size > 0):
                # Calculate val size relative to current train_syntactic_df to achieve desired original val proportion
                # This assumes train_syntactic_df holds the intended 'train_actual_size' proportion of data
                relative_val_size = self.val_actual_size / (self.train_actual_size + self.val_actual_size) if (self.train_actual_size + self.val_actual_size) > 0 else 0.2

            if relative_val_size <=0 or relative_val_size >=1: relative_val_size = 0.2 # safety net

            try:
                 train_syntactic_df, val_syntactic_df, y_train, y_val = train_test_split(
                     train_syntactic_df, y_train_1d, test_size=relative_val_size, random_state=self.random_state, stratify=y_train_1d
                 )
            except ValueError as e: # Stratify error
                 logger.warning(f"Stratified split failed ({e}). Attempting non-stratified split.")
                 train_syntactic_df, val_syntactic_df, y_train, y_val = train_test_split(
                     train_syntactic_df, y_train_1d, test_size=relative_val_size, random_state=self.random_state
                 )
            logger.info(f"Train size after split: {train_syntactic_df.shape[0]}, Validation size: {val_syntactic_df.shape[0]}")


        dep_cols = filter_dependency_features(train_syntactic_df)
        const_cols = filter_constituency_features(train_syntactic_df)

        if not dep_cols: logger.warning("No dependency columns in training's syntactic features!")
        if not const_cols: logger.warning("No constituency columns in training's syntactic features!")

        X_train_dep = np.array([]).reshape(train_syntactic_df.shape[0], 0)
        if dep_cols and not train_syntactic_df.empty:
            X_train_dep = train_syntactic_df[dep_cols].values.astype(np.float32)
        
        X_train_const = np.array([]).reshape(train_syntactic_df.shape[0], 0)
        if const_cols and not train_syntactic_df.empty:
            X_train_const = train_syntactic_df[const_cols].values.astype(np.float32)

        X_val_dep, X_val_const = None, None
        if val_syntactic_df is not None and not val_syntactic_df.empty and y_val is not None:
            val_df_processed_dep = val_syntactic_df.copy()
            val_df_processed_const = val_syntactic_df.copy()

            if dep_cols:
                missing_dep_in_val = set(dep_cols) - set(val_df_processed_dep.columns)
                if missing_dep_in_val:
                    logger.warning(f"Adding {len(missing_dep_in_val)} missing dep columns to val_df: {missing_dep_in_val}.")
                    for col in missing_dep_in_val: val_df_processed_dep[col] = 0 
                X_val_dep = val_df_processed_dep[dep_cols].values.astype(np.float32)
            else: X_val_dep = np.array([]).reshape(val_syntactic_df.shape[0], 0)

            if const_cols:
                missing_const_in_val = set(const_cols) - set(val_df_processed_const.columns)
                if missing_const_in_val:
                    logger.warning(f"Adding {len(missing_const_in_val)} missing const columns to val_df: {missing_const_in_val}.")
                    for col in missing_const_in_val: val_df_processed_const[col] = 0
                X_val_const = val_df_processed_const[const_cols].values.astype(np.float32)
            else: X_val_const = np.array([]).reshape(val_syntactic_df.shape[0], 0)
        
        self.results = {} 

        if X_train_dep.shape[1] > 0:
            dt_dep = DecisionTreeClassifier(
                max_depth=self.dt_max_depth,
                min_samples_split=self.dt_min_samples_split,
                random_state=self.random_state
            )
            self.results['DT_Dependency'] = self._train_and_evaluate_classifier(
                dt_dep, X_train_dep, y_train, X_val_dep, y_val, "Decision Tree (Dependency Features)"
            )
        else: 
            logger.warning("No dependency features for DT_Dependency model.")
            self.results['DT_Dependency'] = {'error': 'No dependency features found'}

        if X_train_const.shape[1] > 0:
            dt_const = DecisionTreeClassifier(
                max_depth=self.dt_max_depth,
                min_samples_split=self.dt_min_samples_split,
                random_state=self.random_state
            )
            self.results['DT_Constituency'] = self._train_and_evaluate_classifier(
                dt_const, X_train_const, y_train, X_val_const, y_val, "Decision Tree (Constituency Features)"
            )
        else:
            logger.warning("No constituency features for DT_Constituency model.")
            self.results['DT_Constituency'] = {'error': 'No constituency features found'}

        if X_train_dep.shape[1] > 0:
            knn_dep = KNeighborsClassifier(
                n_neighbors=self.knn_n_neighbors,
                weights=self.knn_weights,
                n_jobs=-1  # Use all available cores
            )
            self.results['KNN_Dependency'] = self._train_and_evaluate_classifier(
                knn_dep, X_train_dep, y_train, X_val_dep, y_val, "KNN (Dependency Features)"
            )
        else:
            logger.warning("No dependency features for KNN_Dependency model.")
            self.results['KNN_Dependency'] = {'error': 'No dependency features found'}

        if X_train_const.shape[1] > 0:
            knn_const = KNeighborsClassifier(
                n_neighbors=self.knn_n_neighbors,
                weights=self.knn_weights,
                n_jobs=-1  # Use all available cores
            )
            self.results['KNN_Constituency'] = self._train_and_evaluate_classifier(
                knn_const, X_train_const, y_train, X_val_const, y_val, "KNN (Constituency Features)"
            )
        else:
            logger.warning("No constituency features for KNN_Constituency model.")
            self.results['KNN_Constituency'] = {'error': 'No constituency features found'}
            
        logger.info(f"===== Experiment 7 Finished ({self.dataset_name}/{self.config_identifier_suffix}) =====")
        logger.info(f"Results: {self.results}")
        return self.results