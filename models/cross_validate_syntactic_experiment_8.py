# IS567FP/models/cross_validate_syntactic_experiment_8.py
import logging
import os
import time
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

from utils.common import NLIModel, get_split_sizes # <<<< MODIFIED: Added get_split_sizes
from utils.database import DatabaseHandler
from .baseline_base import clean_dataset, _handle_nan_values, prepare_labels, filter_syntactic_features

logger = logging.getLogger(__name__)

class CrossValidateSyntacticExperiment8:
    def __init__(self, args: object):
        self.args = args
        self.dataset_name = args.dataset
        self.sample_size = getattr(args, 'sample_size', None)
        self.n_jobs = getattr(args, 'n_jobs', -1)  # Get n_jobs from args, default to -1 (all cores)

        # Determine the suffix for feature file names
        if self.sample_size is not None and self.sample_size > 0:
            # These ratios should reflect the proportions used by preprocessor.py
            # when it determines the size of the 'train' data for the suffix.
            # get_split_sizes defaults to train_ratio=0.8, val_ratio=0.1.
            # We assume these are the relevant conceptual ratios for the train split
            # as suggested by the logic in preprocessor.py.
            train_ratio_from_preprocessor_logic = 0.8
            # val_ratio is also a parameter for get_split_sizes, influencing other splits
            val_ratio_from_preprocessor_logic = 0.1

            # Use get_split_sizes to find out how many samples the 'train' split would have
            train_samples_for_suffix, _, _ = get_split_sizes(
                total_sample_size=self.sample_size,
                train_ratio=train_ratio_from_preprocessor_logic,
                val_ratio=val_ratio_from_preprocessor_logic
            )

            if train_samples_for_suffix is not None and train_samples_for_suffix > 0:
                self.suffix = f"sample{train_samples_for_suffix}"
                logger.info(
                    f"Experiment 8 (using get_split_sizes): "
                    f"total_sample_size from args = {self.sample_size}, "
                    f"calculated train samples for suffix = {train_samples_for_suffix}. "
                    f"Expected suffix: '{self.suffix}'"
                )
            else:
                # This means get_split_sizes allocated 0 or None samples to train.
                # preprocessor.py would likely also not generate files for such a train split.
                # The experiment will fail to load data, which is an expected outcome.
                actual_suffix_val = train_samples_for_suffix if train_samples_for_suffix is not None else 0
                self.suffix = f"sample{actual_suffix_val}" # e.g., "sample0"
                logger.warning(
                    f"Experiment 8 (using get_split_sizes): "
                    f"total_sample_size from args = {self.sample_size}, "
                    f"calculated train samples for suffix = {actual_suffix_val}. "
                    f"Suffix set to '{self.suffix}'. Preprocessor likely did not generate 'train' files for this size. "
                    f"Expect data loading failure for the 'train' split."
                )
        else:
            self.suffix = "full"
            logger.info(f"Experiment 8: No valid sample_size provided. Using suffix 'full'.")

        self.db_handler = DatabaseHandler()
        self.results: Dict[str, Any] = {}
        self.cv_folds = 5
        self.random_state = 42
        # ... (rest of __init__ remains the same: self.classifiers_to_test, self.scoring_metrics) ...
        # Classifier configurations
        self.classifiers_to_test = {
            "DecisionTree": DecisionTreeClassifier(
                max_depth=getattr(args, 'max_depth', 10),
                min_samples_split=getattr(args, 'min_samples_split', 2),
                random_state=self.random_state
            ),
            "KNN": KNeighborsClassifier(
                n_neighbors=getattr(args, 'n_neighbors', 5),
                weights=getattr(args, 'weights', 'uniform'),
                n_jobs=self.n_jobs  # Use n_jobs from args
            ),
            "LogisticRegression": LogisticRegression(
                C=getattr(args, 'C', 1.0),
                max_iter=getattr(args, 'max_iter', 1000),
                solver='liblinear',
                random_state=self.random_state,
                n_jobs=1 #self.n_jobs  # Use n_jobs from args
            ),
            "MultinomialNB": Pipeline([ # Needs scaling for non-negative features
                 ('scaler', MinMaxScaler()),
                 ('mnb', MultinomialNB(alpha=getattr(args, 'alpha', 1.0)))
             ]),
            "RandomForest": RandomForestClassifier(
                n_estimators=getattr(args, 'n_estimators', 100),
                max_depth=getattr(args, 'max_depth', None),
                random_state=self.random_state,
                n_jobs=self.n_jobs  # Use n_jobs from args
            ),
            "GradientBoosting": GradientBoostingClassifier(
                n_estimators=getattr(args, 'n_estimators', 100),
                learning_rate=getattr(args, 'learning_rate', 0.1),
                max_depth=getattr(args, 'max_depth', 3),
                random_state=self.random_state,
                # Reduce memory usage for GradientBoosting
                subsample=0.8,  # Use 80% of samples for each tree
                max_features='sqrt'  # Use sqrt of features for each split
            )
        }

        # Scoring metrics for cross-validation
        self.scoring_metrics = {
            'accuracy': make_scorer(accuracy_score),
            'precision_weighted': make_scorer(precision_score, average='weighted', zero_division=0),
            'recall_weighted': make_scorer(recall_score, average='weighted', zero_division=0),
            'f1_weighted': make_scorer(f1_score, average='weighted', zero_division=0)
        }


    def _load_and_prepare_features(self, split: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Loads precomputed features, filters syntactic ones, and prepares labels."""
        # This method now uses the self.suffix calculated in __init__
        logger.info(f"Exp8: Loading precomputed features for {self.dataset_name}/{split}/{self.suffix}")
        
        # IMPORTANT: Experiment 8 is for cross-validation on the 'train' split.
        # The suffix calculation above is specifically for the 'train' split features.
        # If this function were ever used for 'dev' or 'test' splits, the suffix logic
        # would need to be adjusted for how preprocessor.py names those files.
        # For Exp8, 'split' will always be 'train'.
        
        feature_type_base = f"features_stats_syntactic_{self.suffix}"
        feature_table_name = f"{self.dataset_name}_{split}_{feature_type_base}" # e.g., SNLI_train_features_stats_syntactic_sample80

        try:
            features_df = self.db_handler.load_dataframe(self.dataset_name, split, feature_table_name)
            if features_df.empty:
                logger.error(f"Loaded empty features DataFrame for {feature_table_name}.")
                return None
            features_df = _handle_nan_values(features_df, f"{self.dataset_name}/{split} features")
        except Exception as e:
            logger.error(f"Failed to load features from {feature_table_name}: {e}", exc_info=True)
            return None

        # ... (rest of _load_and_prepare_features remains the same) ...
        clean_result = clean_dataset(features_df)
        if not clean_result:
            logger.error(f"Feature data for {split} is invalid after cleaning.")
            return None
        features_df_clean, y_labels = clean_result

        syntactic_cols = filter_syntactic_features(features_df_clean)
        if not syntactic_cols:
            logger.error("No syntactic feature columns found in the data!")
            return None

        logger.info(f"Exp8: Using {len(syntactic_cols)} syntactic features for cross-validation.")
        X_syntactic = features_df_clean[syntactic_cols].values.astype(np.float32)

        if np.isnan(X_syntactic).any():
            logger.warning("NaNs detected in syntactic features! Filling with 0.")
            X_syntactic = np.nan_to_num(X_syntactic, nan=0.0)
        if not np.all(np.isfinite(X_syntactic)):
            logger.error("Non-finite values remain after NaN handling. Check data.")
            return None

        return X_syntactic, y_labels

    def run_experiment(self) -> Dict[str, Any]:
        # ... (run_experiment method remains largely the same, it uses _load_and_prepare_features)
        logger.info(f"===== Starting Experiment 8: Syntactic Feature Cross-Validation =====")
        logger.info(f"Dataset: {self.dataset_name}, Suffix used for loading: {self.suffix}, Folds: {self.cv_folds}")

        # 1. Load Training Data Features
        # For Experiment 8, we only care about the 'train' split features
        train_prep = self._load_and_prepare_features("train")
        if not train_prep:
            logger.error("Failed to load training data features. Aborting Experiment 8.")
            return {"error": "Training data features failed to load."}
        X_train_syntactic, y_train = train_prep

        if X_train_syntactic.shape[0] == 0:
             logger.error("Training data features are empty after preparation.")
             return {"error": "Empty training features."}
        # ... (rest of run_experiment)
        kfold = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        for model_name, classifier in self.classifiers_to_test.items():
            logger.info(f"--- Running Cross-Validation for: {model_name} ---")
            start_time = time.time()
            try:
                # Adjust n_jobs for cross_validate based on model type
                cv_n_jobs = 1 if model_name == "GradientBoosting" else self.n_jobs
                
                cv_results = cross_validate(
                    estimator=classifier,
                    X=X_train_syntactic,
                    y=y_train,
                    cv=kfold,
                    scoring=self.scoring_metrics,
                    n_jobs=cv_n_jobs,  # Use adjusted n_jobs
                    error_score='raise'
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
                    'fold_f1_scores': cv_results['test_f1_weighted'].tolist()
                }
                logger.info(f"Finished CV for {model_name} in {elapsed_time:.2f}s. Avg F1: {self.results[model_name]['f1_mean']:.4f}")
            except ValueError as ve:
                logger.error(f"ValueError during cross-validation for {model_name}: {ve}. Check feature scaling/data for MNB.", exc_info=True)
                self.results[model_name] = {'error': f"ValueError: {ve}"}
            except Exception as e:
                logger.error(f"Error during cross-validation for {model_name}: {e}", exc_info=True)
                self.results[model_name] = {'error': str(e)}
        
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
        return self.results