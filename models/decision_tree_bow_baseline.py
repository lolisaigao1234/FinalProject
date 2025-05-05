# models/decision_tree_bow_baseline.py
import logging
import time
import pandas as pd
import numpy as np
import joblib
import os
from typing import Optional, Dict, Any

# Scikit-learn components
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer

# Project-specific imports
from utils.common import NLIModel
from utils.database import DatabaseHandler

# Import base utilities
from .baseline_base import TextBaselineModel, clean_dataset, _evaluate_model_performance, SimpleParquetLoader, \
    TextFeatureExtractorBase  # <--- 确保 TextBaselineModel 在这里

logger = logging.getLogger(__name__)


class DecisionTreeBowBaseline(TextBaselineModel):  # Inherits NLIModel
    """Decision Tree baseline using Bag-of-Words features."""
    MODEL_NAME = "DecisionTree_BoW_Baseline"

    def __init__(self, extractor: TextFeatureExtractorBase, model_instance: Any, args: Optional[object] = None,
                 max_features: int = 10000, max_depth: Optional[int] = None, random_state: int = 42, **kwargs):
        # Handle args object if passed
        super().__init__(extractor, model_instance)
        if args:
            max_features = getattr(args, 'bow_max_features', getattr(args, 'max_features', max_features))
            max_depth = getattr(args, 'max_depth', max_depth)
            random_state = getattr(args, 'random_state', random_state)

        # Create the Scikit-learn vectorizer and model instances
        self._vectorizer = CountVectorizer(max_features=max_features, lowercase=True, ngram_range=(1, 1))
        self._model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)

        # Initialize the TextBaselineModel parent class
        # TextBaselineModel 需要 extractor 和 model_instance
        # 注意：这里的 extractor 概念可能与 DecisionTreeBowBaseline 内部的 _vectorizer 不同。
        # TextBaselineModel 的 extractor 预期是一个有 fit/transform/save/load 的对象。
        # 你可能需要创建一个简单的包装器或者调整 TextBaselineModel。
        # **一个简化的方法是，暂时不调用 super().__init__，并直接管理 self._vectorizer 和 self._model。**
        # super().__init__(extractor=???, model_instance=self._model) # <--- 如何提供 extractor?

        self.is_trained = False
        # self.loader = SimpleParquetLoader() # 这些可以在子类中保留或删除，取决于是否使用父类逻辑
        # self.db_handler = DatabaseHandler()

        # --- 重要: TextBaselineModel 期望有 self.extractor 和 self.model ---
        # 为了兼容性，可以这样设置：
        self.extractor = self._vectorizer  # 将内部 vectorizer 赋值给父类期望的属性名
        self.model = self._model  # 将内部 model 赋值给父类期望的属性名

    def _prepare_features(self, df: pd.DataFrame, fit_vectorizer: bool = False) -> Optional[np.ndarray]:
        """Prepares BoW features from premise and hypothesis."""
        # Use premise/hypothesis columns if they exist, fallback to text columns? Adjust as needed.
        premise_col = 'premise' if 'premise' in df.columns else 'premise_text'
        hypothesis_col = 'hypothesis' if 'hypothesis' in df.columns else 'hypothesis_text'

        if premise_col not in df.columns or hypothesis_col not in df.columns:
            logger.error(
                f"DataFrame missing suitable premise ({premise_col}) or hypothesis ({hypothesis_col}) columns.")
            return None

        # Combine premise and hypothesis for vectorization
        # Handle potential NaN values before concatenation
        df[premise_col] = df[premise_col].fillna('')
        df[hypothesis_col] = df[hypothesis_col].fillna('')
        combined_text = df[premise_col] + " " + df[hypothesis_col]

        try:
            if fit_vectorizer:
                logger.info(f"Fitting CountVectorizer with max_features={self.vectorizer.max_features}...")
                features = self.vectorizer.fit_transform(combined_text)
                logger.info(f"Vectorizer fitted with {features.shape[1]} features.")
            else:
                if not hasattr(self.vectorizer, 'vocabulary_') or not self.vectorizer.vocabulary_:
                    logger.error("Vectorizer has not been fitted. Call fit first or load a trained model.")
                    return None
                logger.info("Transforming text using existing CountVectorizer...")
                features = self.vectorizer.transform(combined_text)
            # Convert sparse matrix to dense for Decision Tree (can be memory intensive)
            return features.toarray()
        except Exception as e:
            logger.error(f"Error during CountVectorizer fit/transform: {e}", exc_info=True)
            return None

    # --- Method to satisfy NLIModel ABC ---
    def extract_features(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Extracts features using the assigned extractor (CountVectorizer)."""
        logger.info(f"Extracting BoW features...")
        # TextBaselineModel 的 train 方法会先调用 extract_features
        # 因此，拟合 (fit) 应该在 train 方法调用之前，在 BaselineTrainer 中完成。
        # DecisionTreeBowBaseline 的 train 方法需要调整，不再自己 fit_transform
        # 或者 BaselineTrainer 需要调整为先 fit extractor，再调用 train
        if not hasattr(self.extractor, 'vocabulary_') or not self.extractor.vocabulary_:
            # 在 BaselineTrainer 中，extractor 应该已经被 fit 过了
            logger.error("Vectorizer (extractor) has not been fitted.")
            raise RuntimeError("Vectorizer (extractor) must be fitted before calling extract_features.")

        premise_col = 'premise' if 'premise' in data.columns else 'premise_text'
        hypothesis_col = 'hypothesis' if 'hypothesis' in data.columns else 'hypothesis_text'
        # ... (与 _prepare_features 类似的文本组合和转换逻辑) ...
        combined_text = data[premise_col].fillna('') + " " + data[hypothesis_col].fillna('')
        features = self.extractor.transform(combined_text)
        return features.toarray()  # 返回适合 Decision Tree 的密集数组

    def train(self, X: Any, y: np.ndarray) -> None:
        """Trains the internal Decision Tree model."""
        if X is None or y is None:
            raise ValueError("Features (X) or labels (y) are None.")
        logger.info(f"Training Decision Tree model (max_depth={self.model.max_depth})...")
        start_time = time.time()
        self.model.fit(X, y)  # 直接使用传入的 X, y
        self.is_trained = True
        train_time = time.time() - start_time
        logger.info(f"Training finished in {train_time:.2f}s.")

    def predict(self, X: Any) -> np.ndarray:
        """Makes predictions using the trained model."""
        if not self.is_trained:
            raise RuntimeError(f"{self.MODEL_NAME} has not been trained yet.")
        if X is None:
            raise ValueError("Input features (X) for prediction are None.")
        logger.info(f"Predicting with {self.MODEL_NAME} on {X.shape[0]} samples...")
        predictions = self.model.predict(X)  # 直接使用传入的 X
        logger.debug("Prediction finished.")
        return predictions

    def save(self, directory: str, model_name: str) -> None:
        """Saves the trained model and the feature extractor (vectorizer)."""
        if not self.is_trained and not hasattr(self.extractor, 'vocabulary_'):
            logger.warning(f"Attempting to save {self.MODEL_NAME} where neither model nor vectorizer is fitted.")
            return

        os.makedirs(directory, exist_ok=True)
        model_path = os.path.join(directory, f"{model_name}_model.joblib")
        extractor_path = os.path.join(directory, f"{model_name}_extractor.joblib")

        if self.is_trained:
            try:
                logger.info(f"Saving model to {model_path}")
                joblib.dump(self.model, model_path)
            except Exception as e:
                logger.error(f"Failed to save model: {e}", exc_info=True)
        else:
            logger.warning("Model is not trained, skipping model save.")

        # 总是尝试保存 extractor (vectorizer)，因为它可能已拟合
        if hasattr(self.extractor, 'vocabulary_') and self.extractor.vocabulary_:
            try:
                logger.info(f"Saving vectorizer (extractor) to {extractor_path}")
                joblib.dump(self.extractor, extractor_path)
            except Exception as e:
                logger.error(f"Failed to save vectorizer (extractor): {e}", exc_info=True)
        else:
            logger.warning("Vectorizer (extractor) is not fitted, skipping extractor save.")

    @classmethod
    def load(cls, directory: str, model_name: str) -> 'DecisionTreeBowBaseline':
        """Loads the model and extractor."""
        model_path = os.path.join(directory, f"{model_name}_model.joblib")
        extractor_path = os.path.join(directory, f"{model_name}_extractor.joblib")

        if not os.path.exists(model_path) or not os.path.exists(extractor_path):
            raise FileNotFoundError(
                f"Model or extractor file not found in directory: {directory} for base name {model_name}")

        loaded_model = joblib.load(model_path)
        loaded_extractor = joblib.load(extractor_path)

        # 实例化类，需要传递加载的参数
        try:
            # 假设 DecisionTreeClassifier 有 max_depth 和 random_state 属性
            instance = cls(max_depth=getattr(loaded_model, 'max_depth', None),
                           random_state=getattr(loaded_model, 'random_state', 42))
        except Exception as e:
            logger.warning(f"Could not instantiate {cls.__name__} with loaded params: {e}. Trying default init.")
            instance = cls()  # 回退到默认

        instance.model = loaded_model
        instance.extractor = loaded_extractor  # 赋值给 extractor
        instance._model = loaded_model  # 也更新内部属性
        instance._vectorizer = loaded_extractor  # 也更新内部属性
        instance.is_trained = True
        logger.info(f"{cls.MODEL_NAME} loaded from {directory} using base name {model_name}")
        return instance

    # --- Method to satisfy NLIModel ABC ---
    def evaluate(self, dataset_name: str, split: str, suffix: str) -> Optional[Dict[str, Any]]:
        """
        Evaluates the trained model on a given dataset split.
        Implementation of the abstract evaluate method.
        """
        if not self.is_trained:
            logger.error(f"Cannot evaluate {self.MODEL_NAME}. Model is not trained.")
            return None

        logger.info(f"Evaluating {self.MODEL_NAME} on {dataset_name}/{split} ({suffix})")
        # Load test data
        try:
            df_eval = self.loader.load_data(dataset_name, split, suffix)
        except FileNotFoundError as e:
            logger.error(f"Evaluation data not found: {e}")
            return None
        except ValueError as e:  # Catch missing columns error from loader
            logger.error(f"Error loading evaluation data: {e}")
            return None

        df_eval = clean_dataset(df_eval)
        if df_eval.empty:
            logger.error(f"Evaluation data for {split} is empty after cleaning.")
            return None

        # Extract features (fit=False)
        # X_eval = self.extract_features(df_eval, fit=False) # Predict method does this
        y_true = df_eval['label'].values

        # Make predictions
        try:
            start_time = time.time()
            y_pred = self.predict(df_eval)  # Predict handles feature extraction
            eval_time = time.time() - start_time
        except Exception as e:
            logger.error(f"Error during prediction in evaluation: {e}", exc_info=True)
            return None

        # Calculate metrics
        eval_metrics = _evaluate_model_performance(y_true, y_pred, f"{self.MODEL_NAME} {split.capitalize()}")
        eval_metrics['eval_time'] = eval_time
        eval_metrics['eval_split'] = split  # Add split info

        logger.info(f"Evaluation metrics for {split}: {eval_metrics}")
        return eval_metrics
