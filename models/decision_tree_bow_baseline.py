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
from .baseline_base import clean_dataset, _evaluate_model_performance, SimpleParquetLoader  # Import SimpleParquetLoader

logger = logging.getLogger(__name__)


class DecisionTreeBowBaseline(NLIModel):  # Inherits NLIModel
    """Decision Tree baseline using Bag-of-Words features."""
    MODEL_NAME = "DecisionTree_BoW_Baseline"

    def __init__(self, args: Optional[object] = None, max_features: int = 10000, max_depth: Optional[int] = None,
                 random_state: int = 42, **kwargs):
        # Handle args object if passed
        if args:
            # Use bow_max_features specific arg if available, else fallback
            max_features = getattr(args, 'bow_max_features', getattr(args, 'max_features', max_features))
            max_depth = getattr(args, 'max_depth', max_depth)
            random_state = getattr(args, 'random_state', random_state)

        self.vectorizer = CountVectorizer(max_features=max_features, lowercase=True, ngram_range=(1, 1))
        self.model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
        self.is_trained = False
        # Use the simple loader for demonstration. Replace with DB Handler if appropriate
        self.loader = SimpleParquetLoader()
        self.db_handler = DatabaseHandler() # Or use DB Handler if needed

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
    def extract_features(self, data: pd.DataFrame, fit: bool = False) -> Optional[np.ndarray]:
        """
        Implementation of the abstract extract_features method.
        Delegates to _prepare_features. Handles fitting the vectorizer if fit=True.
        """
        if not isinstance(data, pd.DataFrame):
            logger.error("extract_features requires a pandas DataFrame input.")
            return None
        logger.info(f"Extracting features (fit={fit})...")
        # Determine if fitting is needed based on 'fit' flag AND if the model isn't already trained
        # This logic assumes 'fit=True' is only passed during the training phase.
        should_fit_vectorizer = fit and not self.is_trained
        features = self._prepare_features(data, fit_vectorizer=should_fit_vectorizer)
        # Crucially, if fitting occurred, mark the vectorizer part as ready (model itself isn't trained yet)
        # The main `train` method will handle setting `self.is_trained` after model.fit()
        return features

    def train(self, train_dataset: str, train_split: str, train_suffix: str,
              val_dataset: str, val_split: str, val_suffix: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Train the Decision Tree model."""
        logger.info(f"Starting training for {self.MODEL_NAME} on {train_dataset}/{train_split} ({train_suffix})")
        # Load data
        try:
            # Use self.loader to get data
            df_train = self.loader.load_data(train_dataset, train_split, train_suffix)
            # Load validation data if split/suffix specify it
            df_val = None
            if val_dataset and val_split and val_suffix:
                df_val = self.loader.load_data(val_dataset, val_split, val_suffix)
                df_val = clean_dataset(df_val)
            else:
                logger.info("No validation dataset/split/suffix provided for evaluation during training.")

        except FileNotFoundError as e:
            logger.error(f"Required data not found: {e}")
            return None
        except ValueError as e:  # Catch missing columns error from loader
            logger.error(f"Error loading data: {e}")
            return None

        df_train = clean_dataset(df_train)

        if df_train.empty:
            logger.error("Training data is empty after cleaning.")
            return None

        # Use extract_features with fit=True for training data
        X_train = self.extract_features(df_train, fit=True)
        y_train = df_train['label'].values

        if X_train is None:
            logger.error("Feature preparation failed for training data.")
            return None

        logger.info(f"Training Decision Tree model (max_depth={self.model.max_depth})...")
        start_time = time.time()
        self.model.fit(X_train, y_train)
        self.is_trained = True  # Model is now trained
        train_time = time.time() - start_time
        logger.info(f"Training finished in {train_time:.2f}s.")

        # Evaluate on validation set if available
        val_metrics = {}
        if df_val is not None and not df_val.empty:
            logger.info("Evaluating on validation set...")
            # Use extract_features with fit=False for validation data
            X_val = self.extract_features(df_val, fit=False)
            y_val = df_val['label'].values
            if X_val is not None:
                y_pred_val = self.predict(df_val)  # Use predict which calls extract_features internally
                val_metrics = _evaluate_model_performance(y_val, y_pred_val, f"{self.MODEL_NAME} Validation")
                val_metrics['validation_time'] = time.time() - (start_time + train_time)
            else:
                logger.warning("Could not prepare features for validation set.")
        elif df_val is not None and df_val.empty:
            logger.warning("Validation data was empty after cleaning.")
        else:
            logger.info("No validation data to evaluate.")

        # Return combined results
        results = {'train_time': train_time}
        results.update(val_metrics)
        return results

    def predict(self, data: Any) -> np.ndarray:
        """Make predictions (expects a DataFrame)."""
        if not self.is_trained:
            raise RuntimeError(f"{self.MODEL_NAME} has not been trained yet.")
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"Input 'data' for {self.MODEL_NAME}.predict must be a pandas DataFrame.")

        # Use extract_features with fit=False for prediction
        X = self.extract_features(data, fit=False)
        if X is None:
            raise ValueError("Failed to prepare features for prediction.")
        logger.info(f"Predicting with {self.MODEL_NAME} on {X.shape[0]} samples...")
        return self.model.predict(X)

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

    def save(self, path_prefix: str) -> None:
        """Saves the vectorizer and the trained model."""
        if not self.is_trained:
            logger.warning(f"Attempting to save an untrained {self.MODEL_NAME}.")
            # Decide if saving should be allowed even if not trained (e.g., save fitted vectorizer)
            # For safety, often better to prevent saving untrained models fully.
            # return # Uncomment to prevent saving untrained models

        # Ensure directory exists (path_prefix might contain directory path)
        save_dir = os.path.dirname(path_prefix)
        if save_dir:  # Only create if path_prefix includes a directory
            os.makedirs(save_dir, exist_ok=True)

        vectorizer_path = f"{path_prefix}_vectorizer.joblib"
        model_path = f"{path_prefix}_model.joblib"

        try:
            logger.info(f"Saving vectorizer to {vectorizer_path}")
            joblib.dump(self.vectorizer, vectorizer_path)
        except Exception as e:
            logger.error(f"Failed to save vectorizer: {e}", exc_info=True)

        try:
            logger.info(f"Saving model to {model_path}")
            joblib.dump(self.model, model_path)
        except Exception as e:
            logger.error(f"Failed to save model: {e}", exc_info=True)

    @classmethod
    def load(cls, path_prefix: str) -> 'DecisionTreeBowBaseline':
        """Loads the vectorizer and the trained model."""
        vectorizer_path = f"{path_prefix}_vectorizer.joblib"
        model_path = f"{path_prefix}_model.joblib"

        # Check if files exist before attempting load
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"Vectorizer file not found at {vectorizer_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        logger.info(f"Loading vectorizer from {vectorizer_path}")
        vectorizer = joblib.load(vectorizer_path)
        logger.info(f"Loading model from {model_path}")
        model = joblib.load(model_path)

        # Recreate instance - pass necessary params if __init__ requires them
        # Here, assuming defaults or that params are implicitly handled/not needed for load
        # If __init__ requires 'args', loading becomes complex as 'args' isn't saved.
        # Consider saving essential params (max_depth, random_state) in save()
        # Or modify __init__ to have None defaults if args isn't passed.
        try:
            instance = cls(max_depth=model.max_depth, random_state=model.random_state)  # Pass loaded params
        except Exception as e:
            logger.warning(
                f"Could not instantiate {cls.__name__} with loaded params during load: {e}. Trying default init.")
            instance = cls()  # Fallback to default init

        instance.vectorizer = vectorizer
        instance.model = model
        instance.is_trained = True  # Assume loaded model is trained
        logger.info(f"{cls.MODEL_NAME} loaded successfully.")
        return instance
