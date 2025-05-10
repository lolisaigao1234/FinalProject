# File: IS567FP/models/cross_eval_syntactic_experiment_7.py
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
from utils.common import NLIModel  # Assuming this is not used directly by this class but good for context
from utils.database import DatabaseHandler
# Import helpers from baseline_base
from .baseline_base import clean_dataset, _evaluate_model_performance, _handle_nan_values, SimpleParquetLoader, \
    filter_syntactic_features

# Import feature loading and filtering logic (can reuse logic from svm baseline)

logger = logging.getLogger(__name__)


# --- Feature Filtering Specific to Experiment 7 ---

def filter_dependency_features(df: pd.DataFrame) -> List[str]:
    """Return list of dependency-based feature column names."""
    if not isinstance(df, pd.DataFrame):
        logger.error("filter_dependency_features expects a Pandas DataFrame.")
        return []
    # Identify columns related to dependency parsing
    dep_cols = [col for col in df.columns if any(p in col for p in ["_dep_", "deprel_"])]
    logger.debug(f"Exp7: Identified {len(dep_cols)} dependency columns.")
    return dep_cols


def filter_constituency_features(df: pd.DataFrame) -> List[str]:
    """Return list of constituency-based feature column names."""
    if not isinstance(df, pd.DataFrame):
        logger.error("filter_constituency_features expects a Pandas DataFrame.")
        return []
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
    for this specific experimental setup.
    """

    def __init__(self, args: object):
        self.args = args
        self.dataset_name = args.dataset
        self.sample_size = getattr(args, 'sample_size', None)
        # Suffix generally refers to the training sample size marker
        self.suffix = f"sample{self.sample_size}" if self.sample_size else "full"
        self.save_dir = self._get_save_directory()
        os.makedirs(self.save_dir, exist_ok=True)
        # self.db_handler = DatabaseHandler() # Not used directly in the provided snippet for loading
        self.loader = SimpleParquetLoader()
        self.results: Dict[str, Any] = {}

        self.svm_kernel = getattr(args, 'kernel', 'linear')
        self.svm_C = getattr(args, 'C', 1.0)
        self.lr_C = getattr(args, 'C', 1.0)  # Assuming LR C is same as SVM C if not specified otherwise
        self.lr_max_iter = getattr(args, 'max_iter', 1000)
        self.random_state = 42

    def _get_save_directory(self) -> str:
        base_dir = os.path.join(
            os.path.dirname(__file__),
            '..',
            'saved_models',
            'experiment_7',
            self.dataset_name,
            self.suffix
        )
        return base_dir

    def _load_and_prepare_features(self, split: str) -> Optional[Tuple[pd.DataFrame, np.ndarray]]:
        """
        Loads precomputed features using SimpleParquetLoader, filters for syntactic ones,
        and prepares labels. Returns a DataFrame of syntactic features and a NumPy array of labels.
        """
        # The suffix used here should align with how feature files are named by the feature extraction process.
        # Typically, this 'self.suffix' (e.g., "sample80") applies to train, validation, and test files
        # related to this specific sampled run, even if validation/test files contain fewer rows.
        current_split_suffix = self.suffix
        logger.info(
            f"Exp7: Loading precomputed features for {self.dataset_name}/{split}/{current_split_suffix} using SimpleParquetLoader")

        features_df = None
        try:
            # Use SimpleParquetLoader instance, passing self as it was in original SimpleParquetLoader
            # but SimpleParquetLoader.load_data is static, so `self` isn't strictly needed for it.
            # Assuming the loader method signature is `load_data(self_or_caller_instance, dataset_name, split, suffix)`
            # If load_data is truly static like `@staticmethod def load_data(dataset_name, split, suffix):`
            # then the call would be `self.loader.load_data(self.dataset_name, split, current_split_suffix)`
            # Based on the provided baseline_base.py, it seems to be a staticmethod that takes `self` as first arg in definition, which is unusual for @staticmethod.
            # Let's assume it expects the caller instance.
            features_df = self.loader.load_data(self, self.dataset_name, split, current_split_suffix)

            if features_df is None or features_df.empty:
                logger.error(
                    f"SimpleParquetLoader returned empty or None features DataFrame for {self.dataset_name}/{split}/{current_split_suffix}.")
                return None

            # Handle NaNs on the entire loaded DataFrame first
            features_df = _handle_nan_values(features_df, f"Exp7 features for {self.dataset_name}/{split}")

        except FileNotFoundError:
            logger.error(
                f"Exp7: Feature file not found by SimpleParquetLoader for {self.dataset_name}/{split}/{current_split_suffix}.")
            # Log details of paths searched if possible, or ensure SimpleParquetLoader does
            return None  # This will trigger fallback to split from train if it's for validation set
        except Exception as e:
            logger.error(
                f"Exp7: Failed to load features using SimpleParquetLoader for {self.dataset_name}/{split}/{current_split_suffix}: {e}",
                exc_info=True)
            return None

        # Clean dataset (handles labels and removes rows with invalid labels)
        clean_result = clean_dataset(features_df)
        if not clean_result:
            logger.error(f"Exp7: Feature data for {split} is invalid or empty after cleaning.")
            return None
        features_df_clean, y_labels = clean_result  # y_labels is np.ndarray

        if features_df_clean.empty:
            logger.warning(f"Exp7: No valid samples remaining after cleaning for {split}.")
            return None

        # Filter ONLY syntactic features
        syntactic_cols = filter_syntactic_features(features_df_clean)  # Expects DataFrame
        if not syntactic_cols:
            logger.error("Exp7: No syntactic feature columns found in the data after cleaning!")
            return None

        logger.info(f"Exp7: Using {len(syntactic_cols)} syntactic features for {split} split.")
        # Create a DataFrame containing only the syntactic features
        syntactic_features_df = features_df_clean[syntactic_cols]

        # It's good practice to handle NaNs again *after* column selection if some operations might reintroduce them,
        # or if _handle_nan_values didn't cover all cases.
        # However, _handle_nan_values + fillna(0) on numeric_cols should be robust.
        # If syntactic_features_df might have non-numeric columns that became all NaN and were not converted,
        # they might cause issues later. For now, assume syntactic_cols are numeric or handled.

        if syntactic_features_df.shape[0] != len(y_labels):
            logger.error(
                f"Exp7: Mismatch between syntactic features DF ({syntactic_features_df.shape[0]}) and labels ({len(y_labels)}) for {split}.")
            return None

        return syntactic_features_df, y_labels  # Return DataFrame of features

    def _train_and_evaluate_classifier(self, model_instance: Any, X_train: np.ndarray, y_train: np.ndarray,
                                       X_val: Optional[np.ndarray], y_val: Optional[np.ndarray],
                                       model_desc: str) -> Dict[str, Any]:
        """Trains a classifier and evaluates it."""
        logger.info(f"--- Training {model_desc} ---")
        start_time = time.time()

        # Ensure X_train is finite
        if not isinstance(X_train, np.ndarray): X_train = np.asarray(X_train)  # Ensure numpy array
        if not np.all(np.isfinite(X_train)):
            logger.warning(f"Non-finite values (NaN/inf) in {model_desc} training data X_train. Filling with 0.")
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

        # Ensure y_train is correctly formatted (e.g. 1D array)
        y_train = np.asarray(y_train).ravel()

        model_instance.fit(X_train, y_train)
        train_time = time.time() - start_time
        logger.info(f"Training complete in {train_time:.2f}s")

        eval_metrics = {}
        eval_time = 0.0
        if X_val is not None and y_val is not None and X_val.shape[0] > 0:
            logger.info(f"Evaluating {model_desc} on validation data ({X_val.shape[0]} samples)...")
            if not isinstance(X_val, np.ndarray): X_val = np.asarray(X_val)  # Ensure numpy array
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
            logger.info(f"Skipping validation evaluation for {model_desc} (X_val or y_val is None or empty).")

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

        train_prep = self._load_and_prepare_features("train")
        val_prep = self._load_and_prepare_features("validation")
        # test_prep = self._load_and_prepare_features("test") # Uncomment if test eval is needed

        if not train_prep:
            logger.error("Failed to load or prepare training data. Aborting Experiment 7.")
            return {"error": "Training data failed to load."}
        # train_syntactic_df is a DataFrame of only syntactic features
        train_syntactic_df, y_train = train_prep

        val_syntactic_df, y_val = (None, None)
        if val_prep:
            val_syntactic_df, y_val = val_prep  # val_syntactic_df is a DataFrame if loaded
        else:
            logger.warning("Validation data failed to load. Will attempt to split from training data.")

        # If validation data wasn't loaded, split from training data
        if val_syntactic_df is None or (isinstance(val_syntactic_df, pd.DataFrame) and val_syntactic_df.empty):
            if train_syntactic_df.shape[0] < 5:  # Check rows in DataFrame
                logger.error("Not enough training data samples to create a validation split.")
                return {"error": "Not enough data for train/validation split."}

            logger.info(
                "Splitting validation set from training data (as precomputed validation data was not found/loaded).")
            # train_test_split works with DataFrames, will return DataFrames
            # Stratify by y_train which is np.ndarray
            # Ensure y_train is 1D array for stratify
            y_train_1d = np.asarray(y_train).ravel()
            if len(np.unique(y_train_1d)) < 2 and train_syntactic_df.shape[0] * 0.2 >= len(
                    np.unique(y_train_1d)):  # check for stratify
                logger.warning("Not enough classes in y_train for stratified split, attempting non-stratified split.")
                train_syntactic_df, val_syntactic_df, y_train, y_val = train_test_split(
                    train_syntactic_df, y_train, test_size=0.2, random_state=self.random_state
                )
            else:
                train_syntactic_df, val_syntactic_df, y_train, y_val = train_test_split(
                    train_syntactic_df, y_train, test_size=0.2, random_state=self.random_state, stratify=y_train_1d
                )
            logger.info(
                f"Train size after split: {train_syntactic_df.shape[0]}, Validation size: {val_syntactic_df.shape[0]}")

        # 2. Identify feature columns for dependency and constituency from the TRAINING syntactic DataFrame
        # These functions now correctly receive a DataFrame
        dep_cols = filter_dependency_features(train_syntactic_df)
        const_cols = filter_constituency_features(train_syntactic_df)

        if not dep_cols: logger.warning("No dependency feature columns found in training data's syntactic features!")
        if not const_cols: logger.warning(
            "No constituency feature columns found in training data's syntactic features!")

        # Prepare feature matrices (NumPy arrays) for classifiers
        # Training Data
        X_train_dep = np.array([]).reshape(train_syntactic_df.shape[0], 0)
        if dep_cols and not train_syntactic_df.empty:
            X_train_dep = train_syntactic_df[dep_cols].values.astype(np.float32)

        X_train_const = np.array([]).reshape(train_syntactic_df.shape[0], 0)
        if const_cols and not train_syntactic_df.empty:
            X_train_const = train_syntactic_df[const_cols].values.astype(np.float32)

        # Validation Data - Ensure columns match training columns before selection
        X_val_dep = None
        X_val_const = None

        if val_syntactic_df is not None and not val_syntactic_df.empty and y_val is not None:
            # Create copies to add missing columns if needed
            val_df_processed_dep = val_syntactic_df.copy()
            val_df_processed_const = val_syntactic_df.copy()

            # Dependency Features for Validation
            if dep_cols:
                missing_dep_in_val = set(dep_cols) - set(val_df_processed_dep.columns)
                if missing_dep_in_val:
                    logger.warning(
                        f"Adding {len(missing_dep_in_val)} missing dependency columns to val_df: {missing_dep_in_val}. Filling with 0.")
                    for col in missing_dep_in_val:
                        val_df_processed_dep[col] = 0
                X_val_dep = val_df_processed_dep[dep_cols].values.astype(np.float32)
            else:  # If no dep_cols from train, then X_val_dep should be empty
                X_val_dep = np.array([]).reshape(val_syntactic_df.shape[0], 0)

            # Constituency Features for Validation
            if const_cols:
                missing_const_in_val = set(const_cols) - set(val_df_processed_const.columns)
                if missing_const_in_val:
                    logger.warning(
                        f"Adding {len(missing_const_in_val)} missing constituency columns to val_df: {missing_const_in_val}. Filling with 0.")
                    for col in missing_const_in_val:
                        val_df_processed_const[col] = 0
                X_val_const = val_df_processed_const[const_cols].values.astype(np.float32)
            else:  # If no const_cols from train, then X_val_const should be empty
                X_val_const = np.array([]).reshape(val_syntactic_df.shape[0], 0)
        else:
            logger.info(
                "Validation data (val_syntactic_df or y_val) is None or empty. Skipping creation of X_val_dep and X_val_const.")

        # 3. Train and Evaluate Classifiers
        self.results = {}  # Clear previous results if any

        # --- SVM ---
        if X_train_dep.shape[1] > 0:  # Check if there are dependency features
            svm_dep = SVC(kernel=self.svm_kernel, C=self.svm_C, probability=True, random_state=self.random_state)
            self.results['SVM_Dependency'] = self._train_and_evaluate_classifier(
                svm_dep, X_train_dep, y_train, X_val_dep, y_val, "SVM (Dependency Features)"
            )
        else:
            logger.warning("No dependency features to train SVM_Dependency model.")
            self.results['SVM_Dependency'] = {'error': 'No dependency features found for training'}

        if X_train_const.shape[1] > 0:  # Check if there are constituency features
            svm_const = SVC(kernel=self.svm_kernel, C=self.svm_C, probability=True, random_state=self.random_state)
            self.results['SVM_Constituency'] = self._train_and_evaluate_classifier(
                svm_const, X_train_const, y_train, X_val_const, y_val, "SVM (Constituency Features)"
            )
        else:
            logger.warning("No constituency features to train SVM_Constituency model.")
            self.results['SVM_Constituency'] = {'error': 'No constituency features found for training'}

        # --- Logistic Regression ---
        if X_train_dep.shape[1] > 0:
            lr_dep = LogisticRegression(C=self.lr_C, max_iter=self.lr_max_iter, solver='liblinear',
                                        random_state=self.random_state)
            self.results['LR_Dependency'] = self._train_and_evaluate_classifier(
                lr_dep, X_train_dep, y_train, X_val_dep, y_val, "Logistic Regression (Dependency Features)"
            )
        else:
            logger.warning("No dependency features to train LR_Dependency model.")
            self.results['LR_Dependency'] = {'error': 'No dependency features found for training'}

        if X_train_const.shape[1] > 0:
            lr_const = LogisticRegression(C=self.lr_C, max_iter=self.lr_max_iter, solver='liblinear',
                                          random_state=self.random_state)
            self.results['LR_Constituency'] = self._train_and_evaluate_classifier(
                lr_const, X_train_const, y_train, X_val_const, y_val, "Logistic Regression (Constituency Features)"
            )
        else:
            logger.warning("No constituency features to train LR_Constituency model.")
            self.results['LR_Constituency'] = {'error': 'No constituency features found for training'}

        logger.info(f"===== Experiment 7 Finished ({self.dataset_name}/{self.suffix}) =====")
        logger.info(f"Results: {self.results}")
        return self.results