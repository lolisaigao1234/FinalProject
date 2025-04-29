# Create file: IS567FP/models/cross_eval_syntactic_experiment_7.py
# --- START cross_eval_syntactic_experiment_7.py ---
import logging
import os
import time
from typing import Dict, Any, Optional, Tuple, List

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Use NLIModel as the base type for models being evaluated internally
from utils.common import NLIModel
from utils.database import DatabaseHandler
# Import helpers from baseline_base
from .baseline_base import clean_dataset, _evaluate_model_performance, _handle_nan_values, prepare_labels
# Import feature loading and filtering logic (can reuse logic from svm baseline)
from .svm_bow_baseline import load_parquet_data, filter_syntactic_features, filter_lexical_features # Reusing helpers

logger = logging.getLogger(__name__)

# --- Feature Filtering Specific to Experiment 7 ---

def filter_dependency_features(df: pd.DataFrame) -> List[str]:
    """Return list of dependency-based feature column names."""
    # Identify columns related to dependency parsing
    dep_cols = [col for col in df.columns if any(p in col for p in ["_dep_", "deprel_"])]
    logger.debug(f"Exp7: Identified {len(dep_cols)} dependency columns.")
    return dep_cols

def filter_constituency_features(df: pd.DataFrame) -> List[str]:
    """Return list of constituency-based feature column names."""
    # Identify columns related to constituency parsing
    const_cols = [col for col in df.columns if any(p in col for p in ["_const_"])]
    logger.debug(f"Exp7: Identified {len(const_cols)} constituency columns.")
    return const_cols

# --- Experiment 7 Orchestration Class ---

class CrossEvalSyntacticExperiment7:
    """
    Experiment 7: Compares dependency vs. constituency syntactic features
                  across SVM and Logistic Regression classifiers.
    This class manages the loading, feature selection, training, and evaluation
    for this specific experimental setup. It doesn't inherit from NLIModel directly
    as it orchestrates multiple underlying models.
    """
    def __init__(self, args: object):
        self.args = args
        self.dataset_name = args.dataset
        self.sample_size = getattr(args, 'sample_size', None)
        self.suffix = f"sample{self.sample_size}" if self.sample_size else "full"
        self.save_dir = self._get_save_directory()
        os.makedirs(self.save_dir, exist_ok=True)
        self.db_handler = DatabaseHandler()
        self.results: Dict[str, Any] = {} # To store results

        # Hyperparameters from args
        self.svm_kernel = getattr(args, 'kernel', 'linear')
        self.svm_C = getattr(args, 'C', 1.0)
        self.lr_C = getattr(args, 'C', 1.0)
        self.lr_max_iter = getattr(args, 'max_iter', 1000) # Add if needed
        self.random_state = 42

    def _get_save_directory(self) -> str:
        """Determines the save directory specific to this experiment."""
        # Subdirectory for Experiment 7
        base_dir = os.path.join(
            os.path.dirname(__file__), # Relative to current file's dir
            '..', # Go up one level from models/
            'saved_models', # Main model saving directory
            'experiment_7',
            self.dataset_name,
            self.suffix
        )
        return base_dir

    def _load_and_prepare_features(self, split: str) -> Optional[Tuple[pd.DataFrame, np.ndarray]]:
        """Loads precomputed features and prepares labels for a given split."""
        logger.info(f"Exp7: Loading precomputed features for {self.dataset_name}/{split}/{self.suffix}")
        # Load the combined lexical+syntactic features file
        feature_type_base = f"features_lexical_syntactic_{self.suffix}"
        feature_table_name = f"{self.dataset_name}_{split}_{feature_type_base}"

        try:
            features_df = self.db_handler.load_dataframe(self.dataset_name, split, feature_table_name)
            if features_df.empty:
                logger.error(f"Loaded empty features DataFrame for {feature_table_name}.")
                return None
            features_df = _handle_nan_values(features_df, f"{self.dataset_name}/{split} features")
        except Exception as e:
            logger.error(f"Failed to load features from {feature_table_name}: {e}", exc_info=True)
            return None

        # Clean dataset (handles labels)
        clean_result = clean_dataset(features_df)
        if not clean_result:
            logger.error(f"Feature data for {split} is invalid after cleaning.")
            return None
        features_df_clean, y_labels = clean_result

        return features_df_clean, y_labels

    def _train_and_evaluate_classifier(self, model_instance: Any, X_train: np.ndarray, y_train: np.ndarray,
                                      X_val: Optional[np.ndarray], y_val: Optional[np.ndarray],
                                      model_desc: str) -> Dict[str, Any]:
        """Trains a classifier and evaluates it."""
        logger.info(f"--- Training {model_desc} ---")
        start_time = time.time()
        if not np.all(np.isfinite(X_train)):
            logger.warning(f"Non-finite values in {model_desc} training data. Filling with 0.")
            X_train = np.nan_to_num(X_train, nan=0.0)
        model_instance.fit(X_train, y_train)
        train_time = time.time() - start_time
        logger.info(f"Training complete in {train_time:.2f}s")

        eval_metrics = {}
        eval_time = 0.0
        if X_val is not None and y_val is not None and X_val.shape[0] > 0:
            logger.info(f"Evaluating {model_desc} on validation data...")
            if not np.all(np.isfinite(X_val)):
                 logger.warning(f"Non-finite values in {model_desc} validation data. Filling with 0.")
                 X_val = np.nan_to_num(X_val, nan=0.0)
            # Create a temporary wrapper for evaluation if needed, or adapt evaluator
            # Simple approach: use the fitted model directly
            # Need a predict method for _evaluate_model_performance
            class PredictWrapper: # Simple wrapper to mimic NLIModel predict
                def __init__(self, model): self.model = model
                def predict(self, X): return self.model.predict(X)

            eval_time, metrics = _evaluate_model_performance(PredictWrapper(model_instance), X_val, y_val)
            eval_metrics = metrics
            eval_metrics['eval_time'] = eval_time
            logger.info(f"Validation Metrics: {metrics}")
        else:
            logger.info("Skipping validation evaluation for {model_desc}.")

        # Save the trained sklearn model
        model_filename = f"{model_desc.replace(' ', '_').lower()}.joblib"
        model_path = os.path.join(self.save_dir, model_filename)
        try:
            joblib.dump(model_instance, model_path)
            logger.info(f"Saved {model_desc} model to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model {model_desc} to {model_path}: {e}", exc_info=True)

        return {**eval_metrics, 'train_time': train_time}

    def run_experiment(self) -> Dict[str, Any]:
        """Runs the full Experiment 7 pipeline."""
        logger.info(f"===== Starting Experiment 7 for {self.dataset_name} ({self.suffix}) =====")

        # 1. Load data
        train_prep = self._load_and_prepare_features("train")
        val_prep = self._load_and_prepare_features("validation")
        # test_prep = self._load_and_prepare_features("test") # Load test data if final eval needed

        if not train_prep:
            logger.error("Failed to load training data. Aborting Experiment 7.")
            return {"error": "Training data failed to load."}
        train_df, y_train = train_prep
        val_df, y_val = val_prep if val_prep else (None, None)
        # test_df, y_test = test_prep if test_prep else (None, None)

        # Split validation set from training if not loaded
        if val_df is None or y_val is None:
             if len(train_df) < 5: logger.error("Not enough train data for validation split"); return {"error": "Not enough data"}
             logger.info("Splitting validation set from training data.")
             train_df, val_df, y_train, y_val = train_test_split(train_df, y_train, test_size=0.2, random_state=self.random_state, stratify=y_train)
             logger.info(f"Train size: {len(train_df)}, Validation size: {len(val_df)}")

        # Inside run_experiment method of CrossEvalSyntacticExperiment7

        # ... (previous code loading train_df, val_df, y_train, y_val) ...

        # 2. Identify feature columns BASED ON TRAINING DATA
        dep_cols = filter_dependency_features(train_df)
        const_cols = filter_constituency_features(train_df)

        if not dep_cols: logger.warning("No dependency feature columns found in training data!")
        if not const_cols: logger.warning("No constituency feature columns found in training data!")

        # Prepare feature matrices
        # Training Data (assume columns exist as they were derived from train_df)
        X_train_dep = train_df[dep_cols].values if dep_cols else np.array([]).reshape(len(train_df), 0)
        X_train_const = train_df[const_cols].values if const_cols else np.array([]).reshape(len(train_df), 0)

        # --- START FIX ---
        # Validation Data - Ensure columns match training columns before selection
        X_val_dep = None
        X_val_const = None

        if val_df is not None and y_val is not None:  # Check if validation data is loaded and valid
            val_df_processed = val_df.copy()  # Work on a copy

            # Dependency Features for Validation
            if dep_cols:
                missing_dep_in_val = set(dep_cols) - set(val_df_processed.columns)
                if missing_dep_in_val:
                    logger.warning(
                        f"Adding {len(missing_dep_in_val)} missing dependency columns to val_df: {missing_dep_in_val}")
                    for col in missing_dep_in_val:
                        val_df_processed[col] = 0  # Add missing columns with 0
                # Now select using the definitive list from training data
                X_val_dep = val_df_processed[dep_cols].values
            else:
                X_val_dep = np.array([]).reshape(len(val_df_processed), 0)

            # Constituency Features for Validation
            if const_cols:
                missing_const_in_val = set(const_cols) - set(val_df_processed.columns)
                if missing_const_in_val:
                    logger.warning(
                        f"Adding {len(missing_const_in_val)} missing constituency columns to val_df: {missing_const_in_val}")
                    for col in missing_const_in_val:
                        val_df_processed[col] = 0  # Add missing columns with 0
                # Select using the definitive list from training data
                X_val_const = val_df_processed[const_cols].values
            else:
                X_val_const = np.array([]).reshape(len(val_df_processed), 0)

        # --- END FIX ---

        # 3. Train and Evaluate Classifiers

        # --- SVM ---
        if dep_cols:
            svm_dep = SVC(kernel=self.svm_kernel, C=self.svm_C, probability=True, random_state=self.random_state)
            self.results['SVM_Dependency'] = self._train_and_evaluate_classifier(
                svm_dep, X_train_dep, y_train, X_val_dep, y_val, "SVM (Dependency Features)"
            )
        else: self.results['SVM_Dependency'] = {'error': 'No dependency features found'}

        if const_cols:
            svm_const = SVC(kernel=self.svm_kernel, C=self.svm_C, probability=True, random_state=self.random_state)
            self.results['SVM_Constituency'] = self._train_and_evaluate_classifier(
                svm_const, X_train_const, y_train, X_val_const, y_val, "SVM (Constituency Features)"
            )
        else: self.results['SVM_Constituency'] = {'error': 'No constituency features found'}

        # --- Logistic Regression ---
        if dep_cols:
            lr_dep = LogisticRegression(C=self.lr_C, max_iter=self.lr_max_iter, solver='liblinear', random_state=self.random_state)
            # LR might benefit from scaling, but train/eval helper doesn't include it. Add scaling here if needed.
            # scaler_dep = StandardScaler(with_mean=False)
            # X_train_dep_scaled = scaler_dep.fit_transform(X_train_dep)
            # X_val_dep_scaled = scaler_dep.transform(X_val_dep) if X_val_dep is not None else None
            self.results['LR_Dependency'] = self._train_and_evaluate_classifier(
                lr_dep, X_train_dep, y_train, X_val_dep, y_val, "Logistic Regression (Dependency Features)"
            )
        else: self.results['LR_Dependency'] = {'error': 'No dependency features found'}

        if const_cols:
            lr_const = LogisticRegression(C=self.lr_C, max_iter=self.lr_max_iter, solver='liblinear', random_state=self.random_state)
            # scaler_const = StandardScaler(with_mean=False)
            # X_train_const_scaled = scaler_const.fit_transform(X_train_const)
            # X_val_const_scaled = scaler_const.transform(X_val_const) if X_val_const is not None else None
            self.results['LR_Constituency'] = self._train_and_evaluate_classifier(
                lr_const, X_train_const, y_train, X_val_const, y_val, "Logistic Regression (Constituency Features)"
            )
        else: self.results['LR_Constituency'] = {'error': 'No constituency features found'}

        logger.info(f"===== Experiment 7 Finished =====")
        logger.info(f"Results: {self.results}")
        return self.results

# --- END cross_eval_syntactic_experiment_7.py ---