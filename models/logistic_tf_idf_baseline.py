# IS567FP/models/logistic_tf_idf_baseline.py
import logging
import os
import joblib
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict # Added Dict
import time # Added time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support # Added metrics

from utils.common import NLIModel
from models.SVMTrainer import clean_dataset # Re-use helper functions
from config import MODELS_DIR # Import necessary configs
from utils.database import DatabaseHandler # To load intermediate pairs/sentences data

logger = logging.getLogger(__name__)


def _evaluate_model(model: NLIModel, X_val: np.ndarray, y_val: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """Evaluate model performance on validation/test data."""
    logger.info(f"Evaluating model on {X_val.shape[0]} samples...")
    start_time = time.time()
    y_pred = model.predict(X_val)
    eval_time = time.time() - start_time
    logger.info(f"Prediction finished in {eval_time:.2f} seconds.")

    accuracy = accuracy_score(y_val, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_val, y_pred, average='weighted', zero_division=0
    )

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    logger.info(f"Evaluation Metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    return eval_time, metrics


class TFIDFExtractor:
    """
    Handles TF-IDF vectorization for premise and hypothesis pairs.
    Fits the vectorizer on training data and transforms train/val/test data.
    """
    def __init__(self, max_features: Optional[int] = 10000, ngram_range: Tuple[int, int] = (1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english'
        )
        self.is_fitted = False

    def fit(self, data: pd.DataFrame):
        """Fits the TF-IDF vectorizer on the combined premise and hypothesis text."""
        if 'premise_text' not in data.columns or 'hypothesis_text' not in data.columns:
            raise ValueError("DataFrame must contain 'premise_text' and 'hypothesis_text' columns for fitting.")

        combined_texts = data['premise_text'].fillna('') + " " + data['hypothesis_text'].fillna('')
        logger.info(f"Fitting TF-IDF vectorizer on {len(combined_texts)} combined texts...")
        self.vectorizer.fit(combined_texts)
        self.is_fitted = True
        logger.info(f"TF-IDF vectorizer fitted with vocabulary size: {len(self.vectorizer.vocabulary_)}")

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """Transforms premise and hypothesis text into TF-IDF features."""
        if not self.is_fitted:
            raise RuntimeError("TF-IDF vectorizer must be fitted before transforming.")
        if 'premise_text' not in data.columns or 'hypothesis_text' not in data.columns:
            raise ValueError("DataFrame must contain 'premise_text' and 'hypothesis_text' columns for transforming.")

        logger.info(f"Transforming {len(data)} samples with TF-IDF...")
        premise_tfidf = self.vectorizer.transform(data['premise_text'].fillna(''))
        hypothesis_tfidf = self.vectorizer.transform(data['hypothesis_text'].fillna(''))

        features = np.concatenate([premise_tfidf.toarray(), hypothesis_tfidf.toarray()], axis=1)
        logger.info(f"Transformation complete. Feature shape: {features.shape}")
        return features

    def save(self, filepath: str):
        """Saves the fitted vectorizer."""
        joblib.dump(self.vectorizer, filepath)
        logger.info(f"TF-IDF vectorizer saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'TFIDFExtractor':
        """Loads a fitted vectorizer."""
        vectorizer = joblib.load(filepath)
        instance = cls()
        instance.vectorizer = vectorizer
        instance.is_fitted = True
        logger.info(f"TF-IDF vectorizer loaded from {filepath}")
        return instance


class LogisticTFIDFBaseline(NLIModel):
    """Baseline NLI model using Logistic Regression with TF-IDF features."""

    def __init__(self, C: float = 1.0, max_iter: int = 1000, max_features: Optional[int] = 10000, ngram_range: Tuple[int, int] = (1, 2)):
        self.C = C
        self.max_iter = max_iter
        self.tfidf_extractor = TFIDFExtractor(max_features=max_features, ngram_range=ngram_range)
        self.model = LogisticRegression(C=C, max_iter=max_iter, solver='liblinear', random_state=42)
        self.is_trained = False

    def extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extracts TF-IDF features assuming extractor is fitted/loaded."""
        if not self.tfidf_extractor.is_fitted:
             logger.error("TF-IDF extractor is not fitted. Cannot extract features properly.")
             raise RuntimeError("TF-IDF extractor must be fitted or loaded before extracting features.")
        return self.tfidf_extractor.transform(data)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Trains the Logistic Regression model."""
        logger.info(f"Training Logistic Regression with {X.shape[0]} samples and {X.shape[1]} features...")
        self.model.fit(X, y)
        self.is_trained = True
        logger.info("Logistic Regression training complete.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Makes predictions using the trained model."""
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet.")
        logger.info(f"Predicting on {X.shape[0]} samples...")
        predictions = self.model.predict(X)
        logger.info("Prediction finished.")
        return predictions

    def save(self, directory: str, model_name: str = "logistic_tfidf_baseline") -> None:
        """Saves the trained model and the TF-IDF vectorizer."""
        os.makedirs(directory, exist_ok=True)
        model_path = os.path.join(directory, f"{model_name}_model.joblib")
        vectorizer_path = os.path.join(directory, f"{model_name}_vectorizer.joblib")

        joblib.dump(self.model, model_path)
        self.tfidf_extractor.save(vectorizer_path)
        logger.info(f"Logistic Regression model saved to {model_path}")
        logger.info(f"TF-IDF vectorizer saved to {vectorizer_path}")

    @classmethod
    def load(cls, directory: str, model_name: str = "logistic_tfidf_baseline") -> 'LogisticTFIDFBaseline':
        """Loads the model and vectorizer."""
        model_path = os.path.join(directory, f"{model_name}_model.joblib")
        vectorizer_path = os.path.join(directory, f"{model_name}_vectorizer.joblib")

        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"Model or vectorizer file not found in directory: {directory}")

        loaded_model = joblib.load(model_path)
        loaded_tfidf_extractor = TFIDFExtractor.load(vectorizer_path)

        instance = cls(C=loaded_model.C, max_iter=loaded_model.max_iter)
        instance.model = loaded_model
        instance.tfidf_extractor = loaded_tfidf_extractor
        instance.is_trained = True
        logger.info(f"Logistic TF-IDF baseline loaded from {directory}")
        return instance

    @staticmethod
    def load_raw_text_data(dataset_name: str, split: str, suffix: str, db_handler: DatabaseHandler) -> Optional[pd.DataFrame]:
        """
        Loads raw text data by merging intermediate 'pairs' and 'sentences' files.
        Falls back to using DatasetLoader if intermediate files are not found.
        """
        logger.info(f"Attempting to load intermediate raw text data for TF-IDF: {dataset_name}/{split}/{suffix}")
        pairs_table = f"pairs_{suffix}"
        sentences_table = f"sentences_{suffix}"
        merged_df = None # Initialize

        try:
            pairs_df = db_handler.load_dataframe(dataset_name, split, pairs_table)
            sentences_df = db_handler.load_dataframe(dataset_name, split, sentences_table)

            if not pairs_df.empty and not sentences_df.empty:
                logger.info("Merging intermediate pairs and sentences data...")
                # --- Primary Logic: Merge intermediate files ---
                sentences_premise = sentences_df[['id', 'text']].rename(columns={'text': 'premise_text', 'id': 'p_id'})
                sentences_hypothesis = sentences_df[['id', 'text']].rename(columns={'text': 'hypothesis_text', 'id': 'h_id'})
                pairs_essential = pairs_df[['id', 'premise_id', 'hypothesis_id', 'label']].rename(columns={'id': 'pair_id'})

                merged_df = pd.merge(pairs_essential, sentences_premise, left_on='premise_id', right_on='p_id', how='left')
                merged_df = pd.merge(merged_df, sentences_hypothesis, left_on='hypothesis_id', right_on='h_id', how='left')

                final_cols = ['pair_id', 'premise_text', 'hypothesis_text', 'label']
                if not all(col in merged_df.columns for col in final_cols):
                    missing = [col for col in final_cols if col not in merged_df.columns]
                    logger.error(f"Columns missing after merge: {missing}. Cannot proceed with merged data.")
                    merged_df = None # Invalidate merge result
                else:
                    merged_df = merged_df[final_cols]
                    merged_df['premise_text'] = merged_df['premise_text'].fillna('')
                    merged_df['hypothesis_text'] = merged_df['hypothesis_text'].fillna('')
                    logger.info(f"Successfully loaded and merged intermediate data. Shape: {merged_df.shape}")
            else:
                 logger.warning(f"Intermediate data ({pairs_table} or {sentences_table}) not found or empty for {dataset_name}/{split}/{suffix}. Will attempt fallback.")
                 merged_df = None # Explicitly set to None to trigger fallback check

        except Exception as e:
            logger.error(f"Error during primary data loading/merging: {e}. Will attempt fallback.")
            merged_df = None # Ensure fallback is attempted on error

        # --- Fallback Logic ---
        if merged_df is None:
            logger.warning("Executing fallback: Loading original raw data using DatasetLoader.")
            try:
                # *** Import DatasetLoader ONLY if fallback is needed ***
                from data.data_loader import DatasetLoader # [cite: 1]
                loader = DatasetLoader(db_handler)
                # Use the 'split' name directly (e.g., 'train', 'validation', 'test')
                raw_df = loader.load_dataset(dataset_name, split=split)

                if not raw_df.empty and 'premise_text' in raw_df.columns and 'hypothesis_text' in raw_df.columns:
                    # Ensure label column exists and is standardized
                    if 'label' not in raw_df.columns and 'gold_label' in raw_df.columns:
                         raw_df = raw_df.rename(columns={'gold_label':'label'})
                    elif 'label' not in raw_df.columns:
                         logger.error("Fallback raw data missing 'label' column.")
                         return None

                    # Add pair_id if missing (use original ID if available)
                    if 'pair_id' not in raw_df.columns and 'id' in raw_df.columns:
                         raw_df = raw_df.rename(columns={'id':'pair_id'})
                    elif 'pair_id' not in raw_df.columns:
                         # Generate IDs if absolutely necessary, but prefer original IDs
                         logger.warning("Fallback data missing 'id' or 'pair_id'. Generating fallback IDs.")
                         raw_df['pair_id'] = [f"fallback_{i}" for i in range(len(raw_df))]

                    # Select necessary columns
                    required_cols = ['pair_id', 'premise_text', 'hypothesis_text', 'label']
                    if all(col in raw_df.columns for col in required_cols):
                         logger.info(f"Successfully loaded raw data via fallback. Shape: {raw_df.shape}")
                         # Fill NaNs in text columns just in case
                         raw_df['premise_text'] = raw_df['premise_text'].fillna('')
                         raw_df['hypothesis_text'] = raw_df['hypothesis_text'].fillna('')
                         return raw_df[required_cols]
                    else:
                         missing = [col for col in required_cols if col not in raw_df.columns]
                         logger.error(f"Fallback raw data is missing required columns after processing: {missing}")
                         return None
                else:
                    logger.error("Fallback loading via DatasetLoader failed or returned incomplete data.")
                    return None
            except ImportError:
                 logger.error("Fallback failed: Could not import DatasetLoader from data.data_loader.")
                 return None
            except Exception as e:
                logger.error(f"Error during fallback data loading: {e}", exc_info=True) # Log traceback for fallback errors
                return None

        # Return the successfully merged dataframe from the primary path if it exists
        return merged_df


class LogisticRegressionTrainer:
    def __init__(self, save_dir: str = os.path.join(MODELS_DIR, 'logistic_tfidf_baseline')):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.db_handler = DatabaseHandler()

    def run_training(self, args):
        """Runs the training and evaluation for the Logistic Regression TF-IDF model."""
        logger.info(f"Starting Logistic Regression TF-IDF training for dataset: {args.dataset}")

        suffix = f"sample{args.sample_size}" if args.sample_size else "full"
        logger.info(f"Using data suffix: {suffix}")

        # Load data using the potentially fallback-enabled method
        train_data = LogisticTFIDFBaseline.load_raw_text_data(args.dataset, 'train', suffix, self.db_handler)
        val_data = LogisticTFIDFBaseline.load_raw_text_data(args.dataset, 'validation', suffix, self.db_handler)
        test_data = LogisticTFIDFBaseline.load_raw_text_data(args.dataset, 'test', suffix, self.db_handler)

        if train_data is None or train_data.empty:
             logger.error(f"Failed to load training data for suffix '{suffix}'. Aborting.")
             return None
        if val_data is None or val_data.empty:
             logger.warning(f"Failed to load validation data for suffix '{suffix}'. Validation will be skipped.")

        # Clean and prepare labels
        train_data, y_train = clean_dataset(train_data)
        y_val = None
        if val_data is not None and not val_data.empty:
            val_data, y_val = clean_dataset(val_data)
        else:
            val_data = None # Ensure consistency

        # Check if train_data is still valid after cleaning
        if train_data is None:
             logger.error("Training data became invalid after cleaning. Aborting.")
             return None

        # Initialize Model
        logger.info("Initializing LogisticTFIDFBaseline model...")
        model = LogisticTFIDFBaseline(
            C=args.C,
            max_features=args.max_features,
        )

        # Fit TF-IDF Extractor and Transform Train Data
        logger.info("Fitting TF-IDF on training data and transforming...")
        model.tfidf_extractor.fit(train_data)
        x_train = model.tfidf_extractor.transform(train_data)

        # Train Model
        start_time = time.time()
        model.train(x_train, y_train)
        train_time = time.time() - start_time
        logger.info(f"Training completed in {train_time:.2f} seconds.")

        # Evaluate on Validation Set
        eval_results = {}
        if val_data is not None and y_val is not None:
            logger.info("Evaluating on validation data...")
            x_val = model.extract_features(val_data) # Use extract_features method
            eval_time, metrics = _evaluate_model(model, x_val, y_val)
            eval_results = {**metrics, 'eval_time': eval_time}
            logger.info(f"Validation Results - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, Eval Time: {eval_time:.2f}s")
        else:
             logger.info("Skipping validation as validation data is not available.")

        # Save Model
        model.save(self.save_dir)

        # Evaluate on Test Set
        test_metrics = {}
        if test_data is not None and not test_data.empty:
             logger.info("Evaluating on test data...")
             test_data, y_test = clean_dataset(test_data)
             if test_data is not None and y_test is not None: # Check after cleaning
                 x_test = model.extract_features(test_data) # Use extract_features method
                 test_eval_time, test_metrics = _evaluate_model(model, x_test, y_test)
                 logger.info(f"Test Results - Accuracy: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1']:.4f}, Eval Time: {test_eval_time:.2f}s")
             else:
                  logger.warning("Test data became invalid after cleaning, skipping test evaluation.")
        else:
             logger.info("Skipping test evaluation as test data is not available.")

        # Return combined results (validation metrics + train time, potentially add test metrics too)
        final_results = {**eval_results, 'train_time': train_time}
        # Optionally add test results under a specific key
        if test_metrics:
             final_results['test_metrics'] = test_metrics

        return final_results