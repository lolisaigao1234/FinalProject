# Create file: IS567FP/models/cross_validate_syntactic_experiment_8.py
# --- START cross_validate_syntactic_experiment_8.py ---
import logging
import os
import time
from typing import Dict, Any, Optional, Tuple, List

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB # MNB needs non-negative features
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler # For MNB non-negativity
from sklearn.pipeline import Pipeline # For scaling within CV folds
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

# Use NLIModel as the base type for models being evaluated internally
from utils.common import NLIModel
from utils.database import DatabaseHandler
# Import helpers from baseline_base
from .baseline_base import clean_dataset, _handle_nan_values, prepare_labels, filter_syntactic_features

# Import feature loading and filtering logic (can reuse logic from svm baseline)

logger = logging.getLogger(__name__)

# --- Experiment 8 Orchestration Class ---

class CrossValidateSyntacticExperiment8:
    """
    Experiment 8: Rigorous comparison of all models using cross-validation
                  on the training dataset, utilizing only pre-trained
                  syntactic features.
    """
    def __init__(self, args: object):
        # self.save_dir = None
        self.args = args
        self.dataset_name = args.dataset
        self.sample_size = getattr(args, 'sample_size', None)
        # Use 'full' suffix if no sample size, otherwise use the sample size
        # This needs to match the suffix used when features were generated
        self.suffix = f"sample{self.sample_size}" if self.sample_size else "full"
        # self.save_dir = self._get_save_directory() # No models saved in pure CV
        # os.makedirs(self.save_dir, exist_ok=True)
        self.db_handler = DatabaseHandler()
        self.results: Dict[str, Any] = {} # To store CV results

        # Hyperparameters from args
        self.cv_folds = 5 # Number of cross-validation folds
        self.random_state = 42

        # Classifier configurations
        self.classifiers_to_test = {
            "SVM_Linear": SVC(kernel='linear', C=getattr(args, 'C', 1.0), probability=False, random_state=self.random_state),
            "SVM_RBF": SVC(kernel='rbf', C=getattr(args, 'C', 1.0), probability=False, random_state=self.random_state),
            "LogisticRegression": LogisticRegression(C=getattr(args, 'C', 1.0), max_iter=getattr(args, 'max_iter', 1000), solver='liblinear', random_state=self.random_state),
            "MultinomialNB": Pipeline([ # Needs scaling for non-negative features
                 ('scaler', MinMaxScaler()),
                 ('mnb', MultinomialNB(alpha=getattr(args, 'alpha', 1.0)))
             ]),
            "RandomForest": RandomForestClassifier(n_estimators=getattr(args, 'n_estimators', 100), max_depth=getattr(args, 'max_depth', None), random_state=self.random_state, n_jobs=-1),
            "GradientBoosting": GradientBoostingClassifier(n_estimators=getattr(args, 'n_estimators', 100), learning_rate=getattr(args, 'learning_rate', 0.1), max_depth=getattr(args, 'max_depth', 3), random_state=self.random_state)
        }

        # Scoring metrics for cross-validation
        self.scoring_metrics = {
            'accuracy': make_scorer(accuracy_score),
            'precision_weighted': make_scorer(precision_score, average='weighted', zero_division=0),
            'recall_weighted': make_scorer(recall_score, average='weighted', zero_division=0),
            'f1_weighted': make_scorer(f1_score, average='weighted', zero_division=0)
        }

    # def _get_save_directory(self) -> str:
    #     # Not saving models, but could save results here
    #     base_dir = os.path.join(
    #         os.path.dirname(__file__),
    #         '..',
    #         'saved_models', # Or a 'results' directory
    #         'experiment_8_results',
    #         self.dataset_name,
    #         self.suffix
    #     )
    #     return base_dir

    def _load_and_prepare_features(self, split: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Loads precomputed features, filters syntactic ones, and prepares labels."""
        logger.info(f"Exp8: Loading precomputed features for {self.dataset_name}/{split}/{self.suffix}")
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

        clean_result = clean_dataset(features_df)
        if not clean_result:
            logger.error(f"Feature data for {split} is invalid after cleaning.")
            return None
        features_df_clean, y_labels = clean_result

        # Filter ONLY syntactic features
        syntactic_cols = filter_syntactic_features(features_df_clean)
        if not syntactic_cols:
            logger.error("No syntactic feature columns found in the data!")
            return None

        logger.info(f"Exp8: Using {len(syntactic_cols)} syntactic features for cross-validation.")
        X_syntactic = features_df_clean[syntactic_cols].values.astype(np.float32) # Ensure numeric

        # Handle potential NaNs *after* filtering
        if np.isnan(X_syntactic).any():
            logger.warning("NaNs detected in syntactic features! Filling with 0.")
            X_syntactic = np.nan_to_num(X_syntactic, nan=0.0)
        if not np.all(np.isfinite(X_syntactic)):
            logger.error("Non-finite values remain after NaN handling. Check data.")
            return None

        return X_syntactic, y_labels

    def run_experiment(self) -> Dict[str, Any]:
        """Runs the cross-validation comparison for all specified models."""
        logger.info(f"===== Starting Experiment 8: Syntactic Feature Cross-Validation =====")
        logger.info(f"Dataset: {self.dataset_name}, Suffix: {self.suffix}, Folds: {self.cv_folds}")

        # 1. Load Training Data Features
        train_prep = self._load_and_prepare_features("train")
        if not train_prep:
            logger.error("Failed to load training data features. Aborting Experiment 8.")
            return {"error": "Training data features failed to load."}
        X_train_syntactic, y_train = train_prep

        if X_train_syntactic.shape[0] == 0:
             logger.error("Training data features are empty after preparation.")
             return {"error": "Empty training features."}

        # 2. Perform Cross-Validation for each classifier
        kfold = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        for model_name, classifier in self.classifiers_to_test.items():
            logger.info(f"--- Running Cross-Validation for: {model_name} ---")
            start_time = time.time()
            try:
                # cross_validate handles splitting and fitting/scoring for each fold
                cv_results = cross_validate(
                    estimator=classifier,
                    X=X_train_syntactic,
                    y=y_train,
                    cv=kfold,
                    scoring=self.scoring_metrics,
                    n_jobs=-1, # Use all available CPU cores
                    error_score='raise' # See errors during CV
                )
                elapsed_time = time.time() - start_time
                self.results[model_name] = {
                    'fit_time_mean': np.mean(cv_results['fit_time']),
                    'score_time_mean': np.mean(cv_results['score_time']),
                    'total_cv_time': elapsed_time,
                    'accuracy_mean': np.mean(cv_results['test_accuracy']),
                    'accuracy_std': np.std(cv_results['test_accuracy']),
                    'precision_mean': np.mean(cv_results['test_precision_weighted']),
                    'precision_std': np.std(cv_results['test_precision_weighted']),
                    'recall_mean': np.mean(cv_results['test_recall_weighted']),
                    'recall_std': np.std(cv_results['test_recall_weighted']),
                    'f1_mean': np.mean(cv_results['test_f1_weighted']),
                    'f1_std': np.std(cv_results['test_f1_weighted']),
                    'fold_f1_scores': cv_results['test_f1_weighted'].tolist() # Optional: store individual fold scores
                }
                logger.info(f"Finished CV for {model_name} in {elapsed_time:.2f}s. Avg F1: {self.results[model_name]['f1_mean']:.4f}")
            except ValueError as ve:
                 logger.error(f"ValueError during cross-validation for {model_name}: {ve}. Check feature scaling/data for MNB.", exc_info=True)
                 self.results[model_name] = {'error': f"ValueError: {ve}"}
            except Exception as e:
                logger.error(f"Error during cross-validation for {model_name}: {e}", exc_info=True)
                self.results[model_name] = {'error': str(e)}

        # 3. Log final summary
        logger.info(f"===== Experiment 8 Cross-Validation Summary ({self.dataset_name} / {self.suffix}) =====")
        for model_name, metrics in self.results.items():
            if 'error' in metrics:
                logger.info(f"  {model_name}: ERROR ({metrics['error']})")
            else:
                logger.info(f"  {model_name}:")
                logger.info(f"    Avg Fit Time: {metrics['fit_time_mean']:.3f}s")
                logger.info(f"    Avg Score Time: {metrics['score_time_mean']:.3f}s")
                logger.info(f"    Avg Accuracy: {metrics['accuracy_mean']:.4f} (+/- {metrics['accuracy_std']:.4f})")
                logger.info(f"    Avg Precision (Weighted): {metrics['precision_mean']:.4f} (+/- {metrics['precision_std']:.4f})")
                logger.info(f"    Avg Recall (Weighted): {metrics['recall_mean']:.4f} (+/- {metrics['recall_std']:.4f})")
                logger.info(f"    Avg F1 (Weighted): {metrics['f1_mean']:.4f} (+/- {metrics['f1_std']:.4f})")
        logger.info(f"=================================================================")

        # Optionally save results to a file
        # results_filename = f"experiment_8_cv_results_{self.dataset_name}_{self.suffix}.json"
        # results_path = os.path.join(self.save_dir, results_filename)
        # try:
        #     import json
        #     with open(results_path, 'w') as f:
        #         json.dump(self.results, f, indent=4)
        #     logger.info(f"Saved Experiment 8 results to {results_path}")
        # except Exception as e:
        #     logger.error(f"Failed to save results to JSON: {e}")

        return self.results

# --- END cross_validate_syntactic_experiment_8.py ---