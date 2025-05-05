# File: IS567FP/models/decision_tree_hand_crafted_syntactic_features_experiment_1.py
import time
import os  # <-- 添加 os 导入
import joblib  # <-- 添加 joblib 导入
from typing import Dict, List, Optional  # <-- 添加 List 导入

import numpy as np
import pandas as pd  # <-- 添加 pandas 导入
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --- (其他导入和配置保持不变) ---
try:
    import config
except ModuleNotFoundError:
    print("Warning: config.py not found. Using placeholder values.")


    class ConfigPlaceholder:
        RANDOM_SEED = 42
        DECISION_TREE_MAX_DEPTH = None
        DECISION_TREE_MIN_SAMPLES_SPLIT = 2
        DECISION_TREE_MIN_SAMPLES_LEAF = 1


    config = ConfigPlaceholder()

try:
    # 确保导入了 SyntacticFeatureExtractor 和 _handle_nan_values
    from .baseline_base import SyntacticFeatureExtractor, clean_dataset, _evaluate_model_performance, \
        SimpleParquetLoader, _handle_nan_values
    from utils.common import NLIModel
except ModuleNotFoundError:
    print("Warning: baseline_base.py or its components not found.")


    # ... (Placeholder definitions remain the same) ...
    class NLIModel:
        pass  # Add placeholder if utils.common is not found


    class SyntacticFeatureExtractor:
        @staticmethod
        def get_feature_columns(self, data): return []
        @staticmethod
        def extract(self, data, feature_cols=None): return np.array([])


    def clean_dataset(df):
        return df, None


    def _evaluate_model_performance(model, X, y):
        return 0.0, {}


    class SimpleParquetLoader:
        @staticmethod
        def load_data(self, *args): return None


    def _handle_nan_values(df, context):
        return df

from abc import ABC
import logging

logger = logging.getLogger(__name__)


# --- (SyntacticFeatureSelector 类保持不变) ---
class SyntacticFeatureSelector:
    """
    Placeholder transformer assuming input X is a DataFrame
    and we need to select columns corresponding to syntactic features.
    Alternatively, if X passed is already just the numeric features, this isn't needed.
    """

    def __init__(self, feature_columns):
        self.feature_columns = feature_columns

    def fit(self, X, y=None):
        return self  # No fitting needed

    def transform(self, X):
        if isinstance(X, pd.DataFrame):  # Check if it's a DataFrame
            missing_cols = [col for col in self.feature_columns if col not in X.columns]
            if missing_cols:
                raise ValueError(f"Missing syntactic feature columns in input data: {missing_cols}")
            return X[self.feature_columns].values
        elif isinstance(X, np.ndarray):  # Check if it's a NumPy array
            # If X is already a numpy array, assume it's the correct features
            # Potentially add shape validation based on expected number of features
            expected_features = len(self.feature_columns) if self.feature_columns else -1
            if expected_features != -1 and X.shape[1] != expected_features:
                logger.warning(
                    f"Input X is a NumPy array, but its shape ({X.shape}) doesn't match expected features ({expected_features}). Proceeding anyway.")
            else:
                logger.debug("Input X is a NumPy array, assuming it contains the correct syntactic features.")
            return X
        else:
            raise TypeError(f"Input X must be a pandas DataFrame or NumPy array, got {type(X)}")


class DecisionTreeSyntacticExperiment1(NLIModel, ABC):
    """
    Experiment 1: Decision Tree + Hand-crafted Syntactic Features
    Decision Tree using only hand-crafted features derived from parse trees.
    Assumes features are pre-computed and passed in the input data.
    """

    def __init__(self, args=None, **kwargs):
        self.args = args
        self.name = "Experiment 1: Decision Tree + Syntactic Features"
        self.description = "Decision Tree classifier using only hand-crafted syntactic features."
        self.feature_type = 'syntactic'
        self.is_trained = False
        self.feature_cols: Optional[List[str]] = None  # 初始化为 Optional[List[str]]

        self.feature_extractor = SyntacticFeatureExtractor()  # 保持不变

        self.params = {
            'max_depth': getattr(args, 'max_depth', config.DECISION_TREE_MAX_DEPTH),
            'min_samples_split': getattr(args, 'min_samples_split', config.DECISION_TREE_MIN_SAMPLES_SPLIT),
            'min_samples_leaf': getattr(args, 'min_samples_leaf', config.DECISION_TREE_MIN_SAMPLES_LEAF),
            'random_state': getattr(args, 'random_state', config.RANDOM_SEED)
        }
        self.model_config = self.params.copy()

        # --- 定义模型管道 (保持不变) ---
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', DecisionTreeClassifier(**self.params))
        ])

    # --- (extract_features, get_pipeline, get_params, train, predict, evaluate 方法保持不变) ---
    def extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Extracts pre-computed syntactic features using the SyntacticFeatureExtractor.
        Stores feature column names during the first call (training).
        Uses stored column names for subsequent calls (prediction/evaluation).
        """
        if not self.is_trained or self.feature_cols is None:
            # 训练阶段：获取并存储特征列名
            self.feature_cols = self.feature_extractor.get_feature_columns(data)  # type: ignore
            if not self.feature_cols:
                logger.warning(f"No syntactic feature columns identified by the extractor for model {self.name}.")
                return np.array([]).reshape(len(data), 0)
            logger.info(f"Training {self.name}: Storing {len(self.feature_cols)} feature columns.")
            features = self.feature_extractor.extract(data, self.feature_cols)  # type: ignore
        else:
            # 预测/评估阶段：使用存储的列名提取特征
            logger.debug(f"Predict/Evaluate {self.name}: Using stored {len(self.feature_cols)} feature columns.")
            features = self.feature_extractor.extract(data, self.feature_cols)  # type: ignore

        if features.size > 0:
            # Convert features back to DataFrame for NaN handling
            # Use stored feature_cols for columns; ensure index alignment
            features_df = pd.DataFrame(features, columns=self.feature_cols,
                                       index=data.index[:len(features)])  # Align index
            features_df = _handle_nan_values(features_df,
                                             context=f"extract_features ({'train' if not self.is_trained else 'predict'})")
            features = features_df.values
        else:
            logger.warning(f"{self.name}: No features extracted, skipping NaN handling.")

        return features

    def get_pipeline(self):
        return self.pipeline

    def get_params(self):
        pipeline_params = self.pipeline.get_params()
        model_params = {f'classifier__{k}': v for k, v in self.params.items()}
        all_params = model_params
        return all_params

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        if X.shape[0] != len(y):
            raise ValueError(f"Feature shape {X.shape} and label length {len(y)} mismatch.")
        if X.size == 0:
            logger.warning(f"Attempting to train {self.name} with zero features. Skipping training.")
            return

        logger.info(f"Training {self.name} with {X.shape[0]} samples and {X.shape[1]} features...")
        start_time = time.time()
        # Ensure feature_cols are set before training, extract_features should handle this
        if self.feature_cols is None and X.shape[1] > 0:
            # This case shouldn't happen if extract_features was called first, but as a safeguard:
            logger.warning("feature_cols not set before training, inferring from X shape.")
            self.feature_cols = [f"feature_{i}" for i in range(X.shape[1])]

        self.pipeline.fit(X, y)
        self.is_trained = True
        train_time = time.time() - start_time
        logger.info(f"{self.name} training complete in {train_time:.2f} seconds.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError(f"Model {self.name} has not been trained yet.")

        # Check input feature count against stored feature_cols count AFTER training
        if self.feature_cols is not None and X.shape[1] != len(self.feature_cols):
            raise ValueError(
                f"Prediction input feature count ({X.shape[1]}) does not match training feature count ({len(self.feature_cols)})."
            )
        elif self.feature_cols is None and self.is_trained:
            # If trained but somehow feature_cols is None (e.g., trained with 0 features)
            if X.shape[1] != 0:
                raise ValueError("Model was trained with 0 features, but prediction input has features.")
            # If X also has 0 features, prediction is trivial (or handle as error)
            logger.warning(f"Predicting with {self.name} which was trained on 0 features.")
            return np.array([])  # Or predict a default class, depends on desired behavior

        if X.size == 0 and (self.feature_cols is None or len(self.feature_cols) == 0):
            logger.warning(f"Predicting with {self.name} on zero features (as expected). Returning empty array.")
            return np.array([])

        logger.debug(f"Predicting with {self.name} on {X.shape[0]} samples...")
        predictions = self.pipeline.predict(X)
        logger.debug("Prediction finished.")
        return predictions

    def evaluate(self, dataset_name: str, split: str, suffix: str) -> Dict[str, float]:
        if not self.is_trained:
            logger.error(f"Cannot evaluate model {self.name}: Model is not trained.")
            return {}

        logger.info(f"Evaluating {self.name} on {dataset_name}/{split}/{suffix}...")
        data_loader = SimpleParquetLoader()
        try:
            eval_df = data_loader.load_data(dataset_name, split, suffix)  # type: ignore
            if eval_df is None or eval_df.empty:
                raise ValueError(f"Failed to load evaluation data for {dataset_name}/{split}/{suffix}")
        except FileNotFoundError:
            logger.error(f"Evaluation data file not found for {dataset_name}/{split}/{suffix}. Cannot evaluate.")
            return {}
        except Exception as e:
            logger.error(f"Error loading evaluation data: {e}", exc_info=True)
            return {}

        cleaned_data = clean_dataset(eval_df)
        if cleaned_data is None:
            logger.error("Evaluation data became empty or invalid after cleaning.")
            return {}
        df_cleaned, y_true = cleaned_data
        if len(df_cleaned) == 0:
            logger.warning("No valid evaluation samples left after cleaning.")
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'eval_time': 0.0}

        try:
            X_eval = self.extract_features(df_cleaned)
            # Ensure y_true aligns with X_eval after potential NaN handling might drop rows indirectly
            # (Though _handle_nan_values fills, safe check)
            if X_eval.shape[0] != len(y_true):
                # Re-align y_true if extract_features modified the effective row count implicitly via DataFrame index
                common_index = df_cleaned.index.intersection(
                    pd.DataFrame(X_eval, index=df_cleaned.index).index)  # Assuming extract keeps index
                if len(common_index) == X_eval.shape[0]:
                    y_true = y_true[common_index]  # Realign labels based on remaining index
                else:
                    # If index doesn't match, something is wrong
                    raise ValueError(
                        f"Evaluation feature/label count mismatch after extraction: {X_eval.shape[0]} vs {len(y_true)}, and index mismatch.")

            if X_eval.size == 0 and (self.feature_cols is None or len(self.feature_cols) > 0):
                # Check if feature_cols expected features but got none
                if self.feature_cols and len(self.feature_cols) > 0:
                    logger.error(
                        f"Evaluation feature extraction resulted in zero features, while {len(self.feature_cols)} were expected.")
                    return {}
                else:  # Model expected 0 features, and got 0 features. Evaluation depends...
                    logger.warning("Evaluating model trained on 0 features with 0 features.")
                    # Decide how to evaluate: maybe return 0 metrics or predict default class?
                    # Using _evaluate_model_performance might still work if y_true is also aligned.


        except Exception as e:
            logger.error(f"Error extracting features during evaluation: {e}", exc_info=True)
            return {}

        # Ensure y_true is numpy array for metrics calculation
        y_true_np = y_true.values if isinstance(y_true, pd.Series) else np.array(y_true)

        eval_time, metrics = _evaluate_model_performance(self, X_eval, y_true_np)  # Pass self (model instance)
        metrics['eval_time'] = eval_time

        logger.info(f"Evaluation complete for {self.name} on {split}. Metrics: {metrics}")
        return metrics

    # ++++++++++++++++ ADDED save AND load METHODS ++++++++++++++++
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
    def load(cls, directory: str, model_name: str) -> 'DecisionTreeSyntacticExperiment1':
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
            # We need 'args' to initialize, but loaded model doesn't depend on original args directly.
            # Pass None or a default args object if necessary for __init__.
            # Here, __init__ uses args primarily for hyperparams, which are part of the loaded pipeline.
            instance = cls(args=None)  # Initialize with None args, pipeline params will override.

            # Assign the loaded components
            instance.pipeline = loaded_pipeline
            instance.feature_cols = loaded_feature_cols
            instance.is_trained = True  # Assume loaded model is trained

            # Optional: Restore hyperparameters from the loaded pipeline if needed
            # loaded_params = loaded_pipeline.named_steps['classifier'].get_params()
            # instance.params.update({k: v for k, v in loaded_params.items() if k in instance.params})
            # instance.model_config = instance.params.copy()

            logger.info(f"{cls.__name__} loaded successfully from {directory} using base name {model_name}")
            return instance
        except Exception as e:
            logger.error(f"Error loading model {cls.__name__} from {directory}: {e}", exc_info=True)
            raise  # Re-raise the exception
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
