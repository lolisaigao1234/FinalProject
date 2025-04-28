# IS567FP/models/multinomial_naive_bayes_baseline.py
import logging
import os
import joblib
import pandas as pd
import numpy as np
from typing import Tuple, Optional
import time

from sklearn.feature_extraction.text import CountVectorizer # Using CountVectorizer for BoW
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline # To chain vectorizer and classifier

from utils.common import NLIModel
from models.SVMTrainer import clean_dataset, _evaluate_model # Re-use helper functions
from config import MODELS_DIR # Import necessary configs
from utils.database import DatabaseHandler # To load intermediate pairs/sentences data

logger = logging.getLogger(__name__)

class BoWExtractor:
    """
    Handles Bag-of-Words (BoW) vectorization using CountVectorizer
    for premise and hypothesis pairs.
    Fits the vectorizer on training data and transforms train/val/test data.
    """
    def __init__(self, max_features: Optional[int] = 10000, ngram_range: Tuple[int, int] = (1, 1)):
        # Use CountVectorizer for pure Bag-of-Words
        self.vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            binary=False # Use term frequency counts
        )
        self.is_fitted = False

    def fit(self, data: pd.DataFrame):
        """Fits the CountVectorizer on the combined premise and hypothesis text."""
        if 'premise_text' not in data.columns or 'hypothesis_text' not in data.columns:
            raise ValueError("DataFrame must contain 'premise_text' and 'hypothesis_text' columns for fitting.")

        # Combine premise and hypothesis into a single string per pair for fitting the vocabulary
        combined_texts = data['premise_text'].fillna('') + " " + data['hypothesis_text'].fillna('')
        logger.info(f"Fitting CountVectorizer (BoW) on {len(combined_texts)} combined texts...")
        self.vectorizer.fit(combined_texts)
        self.is_fitted = True
        logger.info(f"BoW vectorizer fitted with vocabulary size: {len(self.vectorizer.vocabulary_)}")

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """Transforms premise and hypothesis text into BoW features."""
        if not self.is_fitted:
            raise RuntimeError("BoW vectorizer must be fitted before transforming.")
        if 'premise_text' not in data.columns or 'hypothesis_text' not in data.columns:
            raise ValueError("DataFrame must contain 'premise_text' and 'hypothesis_text' columns for transforming.")

        logger.info(f"Transforming {len(data)} samples with BoW...")
        premise_bow = self.vectorizer.transform(data['premise_text'].fillna(''))
        hypothesis_bow = self.vectorizer.transform(data['hypothesis_text'].fillna(''))

        # Concatenate the BoW vectors for premise and hypothesis
        # Note: This doubles the feature dimension compared to TF-IDF example.
        # Alternative: Could concatenate texts first, then transform.
        features = np.concatenate([premise_bow.toarray(), hypothesis_bow.toarray()], axis=1)
        logger.info(f"Transformation complete. BoW Feature shape: {features.shape}")
        return features

    def save(self, filepath: str):
        """Saves the fitted vectorizer."""
        joblib.dump(self.vectorizer, filepath)
        logger.info(f"BoW vectorizer saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'BoWExtractor':
        """Loads a fitted vectorizer."""
        vectorizer = joblib.load(filepath)
        instance = cls()
        instance.vectorizer = vectorizer
        instance.is_fitted = True
        logger.info(f"BoW vectorizer loaded from {filepath}")
        return instance


class MultinomialNaiveBayesBaseline(NLIModel):
    """Baseline NLI model using Multinomial Naive Bayes with Bag-of-Words features."""

    def __init__(self, alpha: float = 1.0, max_features: Optional[int] = 10000, ngram_range: Tuple[int, int] = (1, 1)):
        """
        Args:
            alpha (float): Additive (Laplace/Lidstone) smoothing parameter
                           (0 for no smoothing).
            max_features (Optional[int]): Max features for CountVectorizer.
            ngram_range (Tuple[int, int]): N-gram range for CountVectorizer.
        """
        self.alpha = alpha
        self.bow_extractor = BoWExtractor(max_features=max_features, ngram_range=ngram_range)
        # Combine vectorizer and classifier into a pipeline for easier fit/predict
        # Note: If transforming separately first, just initialize MNB here.
        # self.model = MultinomialNB(alpha=self.alpha)
        # Pipeline approach (fits vectorizer and model together):
        self.pipeline = Pipeline([
            # The pipeline expects raw text, so we need a custom transformer step
            # if we stick to the separate extractor logic.
            # Let's stick to the separate extractor for consistency with other models.
            ('vectorizer', self.bow_extractor.vectorizer), # Pass the internal sklearn vectorizer
            ('classifier', MultinomialNB(alpha=self.alpha))
        ])
        self.is_trained = False

    def extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extracts BoW features using the fitted extractor."""
        if not self.bow_extractor.is_fitted:
            raise RuntimeError("BoW extractor must be fitted or loaded before extracting features.")
        # For MNB, we often concatenate premise and hypothesis *before* vectorizing
        # Let's adjust this logic compared to TF-IDF
        logger.info("Combining premise and hypothesis for BoW feature extraction...")
        combined_texts = data['premise_text'].fillna('') + " " + data['hypothesis_text'].fillna('')
        features = self.bow_extractor.vectorizer.transform(combined_texts) # Use the fitted internal vectorizer
        logger.info(f"BoW feature extraction complete. Shape: {features.shape}")
        # Return sparse matrix directly, as MNB handles it efficiently
        return features

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Trains the Multinomial Naive Bayes model."""
        # If using pipeline, fit takes raw text. If separate, fit takes features.
        # Sticking to separate extractor:
        logger.info(f"Training MultinomialNB with {X.shape[0]} samples and {X.shape[1]} features...")
        self.model = MultinomialNB(alpha=self.alpha) # Initialize classifier here
        self.model.fit(X, y)
        self.is_trained = True
        logger.info("MultinomialNB training complete.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Makes predictions using the trained model."""
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet.")
        logger.info(f"Predicting on {X.shape[0]} samples...")
        predictions = self.model.predict(X)
        logger.info("Prediction finished.")
        return predictions

    def save(self, directory: str, model_name: str = "mnb_bow_baseline") -> None:
        """Saves the trained model and the BoW vectorizer."""
        os.makedirs(directory, exist_ok=True)
        model_path = os.path.join(directory, f"{model_name}_model.joblib")
        vectorizer_path = os.path.join(directory, f"{model_name}_vectorizer.joblib")

        # Save the scikit-learn model and the custom BoWExtractor instance
        joblib.dump(self.model, model_path)
        self.bow_extractor.save(vectorizer_path) # Save the extractor which contains the vectorizer
        logger.info(f"MultinomialNB model saved to {model_path}")
        logger.info(f"BoW vectorizer saved to {vectorizer_path}")

    @classmethod
    def load(cls, directory: str, model_name: str = "mnb_bow_baseline") -> 'MultinomialNaiveBayesBaseline':
        """Loads the model and vectorizer."""
        model_path = os.path.join(directory, f"{model_name}_model.joblib")
        vectorizer_path = os.path.join(directory, f"{model_name}_vectorizer.joblib")

        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"Model or vectorizer file not found in directory: {directory}")

        loaded_model = joblib.load(model_path)
        loaded_bow_extractor = BoWExtractor.load(vectorizer_path) # Load the extractor

        # Reconstruct the instance
        # Get parameters from loaded objects if needed (e.g., max_features from vectorizer)
        instance = cls(
            alpha=loaded_model.alpha,
            max_features=loaded_bow_extractor.vectorizer.max_features,
            ngram_range=loaded_bow_extractor.vectorizer.ngram_range
            )
        instance.model = loaded_model
        instance.bow_extractor = loaded_bow_extractor # Assign the loaded extractor
        instance.is_trained = True
        logger.info(f"MNB BoW baseline loaded from {directory}")
        return instance


class MultinomialNaiveBayesTrainer:
    """Handles training and evaluation for the MNB BoW baseline."""
    def __init__(self, save_dir: str = os.path.join(MODELS_DIR, 'mnb_bow_baseline')):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.db_handler = DatabaseHandler() # Assuming raw text loading logic is needed

    def run_training(self, args):
        """Runs the training and evaluation for the MNB BoW model."""
        logger.info(f"Starting MNB BoW training for dataset: {args.dataset}")

        # Determine suffix based on sample_size arg
        suffix = f"sample{args.sample_size}" if args.sample_size else "full"
        logger.info(f"Using data suffix: {suffix}")

        # Load RAW text data - MNB+BoW works directly on text
        # Re-use the static method from Logistic baseline (or adapt if needed)
        from .logistic_tf_idf_baseline import LogisticTFIDFBaseline # Local import okay here
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

        if train_data is None: # Check if cleaning invalidated data
             logger.error("Training data became invalid after cleaning. Aborting.")
             return None

        # Initialize Model
        logger.info("Initializing MultinomialNaiveBayesBaseline model...")
        model = MultinomialNaiveBayesBaseline(
            alpha=1.0, # Default smoothing, consider making this an arg
            max_features=args.max_features,
            # ngram_range=(1,1) # Use default (1,1) for BoW unless specified
        )

        # Fit BoW Extractor on training data text
        logger.info("Fitting BoW extractor on training data...")
        # Combine premise and hypothesis for fitting the vocabulary
        train_combined_texts = train_data['premise_text'].fillna('') + " " + train_data['hypothesis_text'].fillna('')
        model.bow_extractor.vectorizer.fit(train_combined_texts) # Fit the internal vectorizer
        model.bow_extractor.is_fitted = True
        logger.info("BoW extractor fitting complete.")

        # Transform Training Data using the fitted extractor
        logger.info("Transforming training data...")
        x_train = model.extract_features(train_data) # Use the extract_features method

        # Train Model
        start_time = time.time()
        model.train(x_train, y_train)
        train_time = time.time() - start_time
        logger.info(f"Training completed in {train_time:.2f} seconds.")

        # Evaluate on Validation Set
        eval_results = {}
        if val_data is not None and y_val is not None:
            logger.info("Evaluating on validation data...")
            x_val = model.extract_features(val_data) # Transform validation data
            # Use the evaluation helper function
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
             if test_data is not None and y_test is not None:
                 x_test = model.extract_features(test_data) # Transform test data
                 test_eval_time, test_metrics = _evaluate_model(model, x_test, y_test)
                 logger.info(f"Test Results - Accuracy: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1']:.4f}, Eval Time: {test_eval_time:.2f}s")
             else:
                  logger.warning("Test data became invalid after cleaning, skipping test evaluation.")
        else:
             logger.info("Skipping test evaluation as test data is not available.")

        final_results = {**eval_results, 'train_time': train_time}
        if test_metrics:
             final_results['test_metrics'] = test_metrics

        return final_results