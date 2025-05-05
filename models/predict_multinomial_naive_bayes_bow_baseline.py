# File: IS567FP/models/predict_multinomial_naive_bayes_bow_baseline.py
import os
import joblib
import pandas as pd
import numpy as np
import argparse
import logging
from typing import Dict, Any

# Import the corresponding model class
try:
    # Note: Ensure the class name matches the one in the file
    from multinomial_naive_bayes_bow_baseline import MultinomialNaiveBayesBaseline
    from config import MODELS_DIR
except ImportError:
    print("Error: Could not import MultinomialNaiveBayesBaseline or config.")


    class MultinomialNaiveBayesBaseline:  # Dummy
        @classmethod
        def load(cls, *args): raise NotImplementedError("Dummy load")

        def predict(self, *args): raise NotImplementedError("Dummy predict")


    MODELS_DIR = "./saved_models"  # Dummy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LABEL_MAP_REVERSE: Dict[int, str] = {0: 'entailment', 1: 'contradiction', 2: 'neutral'}


def predict_nli(premise: str, hypothesis: str, model: Any, label_map_reverse: Dict[int, str]) -> str:
    """Makes a prediction using the loaded MNB BoW model."""
    if not premise or not hypothesis:
        logger.warning("Premise or hypothesis is empty.")
        return "unknown"

    input_df = pd.DataFrame({'premise_text': [premise], 'hypothesis_text': [hypothesis]})

    try:
        logger.debug("Extracting features for prediction using model's extract_features...")
        # MultinomialNaiveBayesBaseline (via TextBaselineModel) has extract_features
        features = model.extract_features(input_df)  # Uses BoWExtractor

        if features is None or features.shape[0] == 0 or features.shape[1] == 0:
            logger.warning("Feature extraction resulted in empty features. Cannot predict.")
            return "unknown"

        logger.debug(f"Predicting with model on features of shape: {features.shape}")
        # Predict uses the loaded+fitted MultinomialNB model
        prediction_int = model.predict(features)

        if prediction_int is None or len(prediction_int) == 0:
            logger.error("Model prediction returned None or empty array.")
            return "unknown"

        predicted_label = label_map_reverse.get(prediction_int[0], "unknown")
        logger.info(f"Raw prediction: {prediction_int[0]}, Mapped label: {predicted_label}")
        return predicted_label

    except RuntimeError as e:
        logger.error(f"Runtime error during prediction: {e}", exc_info=True)
        return "unknown"
    except ValueError as e:
        logger.error(f"Value error during prediction: {e}", exc_info=True)
        return "unknown"
    except Exception as e:
        logger.error(f"An unexpected error occurred during prediction: {e}", exc_info=True)
        return "unknown"
