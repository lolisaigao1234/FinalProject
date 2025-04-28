# IS567FP/models/logistic_tf_idf_baseline.py
import logging
import os
import joblib
import pandas as pd
import numpy as np
from typing import Tuple, Optional, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import time

from utils.common import NLIModel
from models.SVMTrainer import prepare_labels, get_label_column, clean_dataset, _handle_nan_values # Re-use helper functions
from config import MODELS_DIR, DATASETS # Import necessary configs
from utils.database import DatabaseHandler # To load intermediate pairs/sentences data

logger = logging.getLogger(__name__)

class TFIDFExtractor:
    """
    Handles TF-IDF vectorization for premise and hypothesis pairs.
    Fits the vectorizer on training data and transforms train/val/test data.
    """
    def __init__(self, max_features: Optional[int] = 10000, ngram_range: Tuple[int, int] = (1, 2)):
        # Combine premise and hypothesis for TF-IDF vocabulary building
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

        # Combine texts for fitting
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

        premise_tfidf = self.vectorizer.transform(data['premise_text'].fillna(''))
        hypothesis_tfidf = self.vectorizer.transform(data['hypothesis_text'].fillna(''))

        # Simple concatenation of features (other strategies like diff/prod could be used)
        # Convert sparse matrices to dense numpy arrays for concatenation if needed by LogisticRegression
        # Note: This can be memory-intensive for large datasets/vocabularies.
        # Consider using sparse matrices directly if the classifier supports it.
        features = np.concatenate([premise_tfidf.toarray(), hypothesis_tfidf.toarray()], axis=1)
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
        self.tfidf_extractor = TFIDFExtractor(max_features=max_features, ngram_range=ngram_range)
        # Increased max_iter for potential convergence issues with large feature sets
        self.model = LogisticRegression(C=C, max_iter=max_iter, solver='liblinear', random_state=42) # liblinear is often good for high-dim sparse data
        self.is_trained = False

    def extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extracts TF-IDF features. Fits the extractor if not already trained."""
        if not self.tfidf_extractor.is_fitted:
             logger.info("Fitting TF-IDF extractor during feature extraction (should ideally be done on training data only before splitting)")
             # This fit should ideally happen *only* on the training set before splitting for validation/testing
             # If called during predict on new data, it should *only* transform
             self.tfidf_extractor.fit(data)

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
        return self.model.predict(X)

    def save(self, directory: str, model_name: str = "logistic_tfidf_baseline") -> None:
        """Saves the trained Logistic Regression model and the TF-IDF vectorizer."""
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

        model = joblib.load(model_path)
        tfidf_extractor = TFIDFExtractor.load(vectorizer_path)

        # Recreate the instance - parameters like C, max_iter are stored in the saved model
        # We don't need to pass them explicitly unless we want to override
        instance = cls()
        instance.model = model
        instance.tfidf_extractor = tfidf_extractor
        instance.is_trained = True # Assume loaded model is trained
        logger.info(f"Logistic TF-IDF baseline loaded from {directory}")
        return instance

    @staticmethod
    def load_raw_text_data(dataset_name: str, split: str, suffix: str, db_handler: DatabaseHandler) -> Optional[pd.DataFrame]:
        """Loads the pairs and sentences dataframes and merges them to get raw text."""
        logger.info(f"Loading raw text data for TF-IDF: {dataset_name}/{split}/{suffix}")
        pairs_table = f"pairs_{suffix}"
        sentences_table = f"sentences_{suffix}"

        pairs_df = db_handler.load_dataframe(dataset_name, split, pairs_table)
        sentences_df = db_handler.load_dataframe(dataset_name, split, sentences_table)

        if pairs_df.empty or sentences_df.empty:
            logger.error(f"Could not load required intermediate data ({pairs_table}, {sentences_table}) for {dataset_name}/{split}.")
            return None

        # Merge to get text
        try:
             # Select necessary columns and rename for clarity before merge
             sentences_premise = sentences_df[['id', 'text']].rename(columns={'text': 'premise_text', 'id': 'p_id'})
             sentences_hypothesis = sentences_df[['id', 'text']].rename(columns={'text': 'hypothesis_text', 'id': 'h_id'})
             pairs_essential = pairs_df[['id', 'premise_id', 'hypothesis_id', 'label']].rename(columns={'id': 'pair_id'})

             # Merge premise
             merged_df = pd.merge(pairs_essential, sentences_premise, left_on='premise_id', right_on='p_id', how='left')
             # Merge hypothesis
             merged_df = pd.merge(merged_df, sentences_hypothesis, left_on='hypothesis_id', right_on='h_id', how='left')

             # Select final columns and drop helper columns
             final_cols = ['pair_id', 'premise_text', 'hypothesis_text', 'label']
             merged_df = merged_df[final_cols]

             # Handle potential missing text after merge (though unlikely if source data is clean)
             merged_df['premise_text'] = merged_df['premise_text'].fillna('')
             merged_df['hypothesis_text'] = merged_df['hypothesis_text'].fillna('')

             logger.info(f"Successfully loaded and merged raw text data. Shape: {merged_df.shape}")
             return merged_df

        except KeyError as e:
            logger.error(f"Missing expected column during raw text data merge: {e}")
            return None
        except Exception as e:
            logger.error(f"Error merging raw text data: {e}")
            return None


# Example of how a trainer class might look (similar to SVMTrainer but using LogisticTFIDFBaseline)
# This could be integrated into main.py or kept separate
class LogisticRegressionTrainer:
    def __init__(self, save_dir: str = os.path.join(MODELS_DIR, 'logistic_tfidf_baseline')):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.db_handler = DatabaseHandler() # Needs DB handler to load intermediate data

    def run_training(self, args):
        """Runs the training and evaluation for the Logistic Regression TF-IDF model."""
        logger.info(f"Starting Logistic Regression TF-IDF training for dataset: {args.dataset}")

        # --- Determine suffix based on args (e.g., sample_size) ---
        # This logic should align with how suffixes are determined in the preprocessing step
        # Example: If sample_size is provided, use it, otherwise assume 'full'
        suffix = f"sample{args.sample_size}" if args.sample_size else "full"
        logger.info(f"Using data suffix: {suffix}")

        # --- Load Raw Text Data ---
        train_data = LogisticTFIDFBaseline.load_raw_text_data(args.dataset, 'train', suffix, self.db_handler)
        val_data = LogisticTFIDFBaseline.load_raw_text_data(args.dataset, 'validation', suffix, self.db_handler)
        test_data = LogisticTFIDFBaseline.load_raw_text_data(args.dataset, 'test', suffix, self.db_handler) # Load test data if needed

        if train_data is None or train_data.empty:
             logger.error(f"Failed to load training data for suffix '{suffix}'. Aborting.")
             return None
        if val_data is None or val_data.empty:
             logger.warning(f"Failed to load validation data for suffix '{suffix}'. Validation will be skipped.")
             # Optionally split from train_data if validation is essential
             # train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42, stratify=train_data['label'])

        # --- Clean Data & Prepare Labels ---
        train_data, y_train = clean_dataset(train_data) # Re-use cleaning logic
        if val_data is not None and not val_data.empty:
            val_data, y_val = clean_dataset(val_data)
        else:
            y_val = None # No validation labels

        # --- Initialize and Train Model ---
        logger.info("Initializing LogisticTFIDFBaseline model...")
        model = LogisticTFIDFBaseline(
            C=args.C, # Get C from args
            max_features=args.max_features, # Get max_features from args
            # max_iter could also be an arg
        )

        # Fit TF-IDF on training data ONLY, then transform train data
        logger.info("Fitting TF-IDF on training data and transforming...")
        model.tfidf_extractor.fit(train_data)
        X_train = model.tfidf_extractor.transform(train_data)

        # Train the Logistic Regression model
        start_time = time.time()
        model.train(X_train, y_train)
        train_time = time.time() - start_time
        logger.info(f"Training completed in {train_time:.2f} seconds.")

        # --- Evaluate Model ---
        eval_results = {}
        if val_data is not None and y_val is not None and not val_data.empty:
            logger.info("Evaluating on validation data...")
            # Transform validation data using the *fitted* vectorizer
            X_val = model.tfidf_extractor.transform(val_data)
            eval_time, metrics = _evaluate_model(model, X_val, y_val) # Re-use evaluation logic
            eval_results = {**metrics, 'eval_time': eval_time}
            logger.info(f"Validation Results - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, Eval Time: {eval_time:.2f}s")
        else:
             logger.info("Skipping validation.")

        # --- Save Model ---
        model.save(self.save_dir)

        # --- Optionally Evaluate on Test Data ---
        if test_data is not None and not test_data.empty:
             logger.info("Evaluating on test data...")
             test_data, y_test = clean_dataset(test_data)
             X_test = model.tfidf_extractor.transform(test_data)
             test_eval_time, test_metrics = _evaluate_model(model, X_test, y_test)
             logger.info(f"Test Results - Accuracy: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1']:.4f}, Eval Time: {test_eval_time:.2f}s")
             # You might want to store or return test results separately

        return {**eval_results, 'train_time': train_time}