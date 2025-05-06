# File: IS567FP/models/predict_decision_tree_bow_baseline.py
import os
import joblib
import pandas as pd
import numpy as np
import argparse
import logging
from typing import Dict, Any

# Import the corresponding model class
try:
    from decision_tree_bow_baseline import DecisionTreeBowBaseline
    from config import MODELS_DIR  # Assuming MODELS_DIR is in config
except ImportError:
    print("Error: Could not import DecisionTreeBowBaseline or config. Ensure this script is run correctly.")


    # Add dummy classes/variables if needed for basic execution without imports
    class DecisionTreeBowBaseline:  # Dummy
        @classmethod
        def load(cls, *args): raise NotImplementedError("Dummy load")

        def predict(self, *args): raise NotImplementedError("Dummy predict")


    MODELS_DIR = "./saved_models"  # Dummy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the reverse mapping from integer labels to string labels
LABEL_MAP_REVERSE: Dict[int, str] = {0: 'entailment', 1: 'contradiction', 2: 'neutral'}


def predict_nli(premise: str, hypothesis: str, model: Any, label_map_reverse: Dict[int, str]) -> str:
    """
    Makes a prediction for a single premise-hypothesis pair using the loaded model.

    Args:
        premise: The premise text.
        hypothesis: The hypothesis text.
        model: The loaded NLI model instance.
        label_map_reverse: Dictionary mapping integer predictions to string labels.

    Returns:
        The predicted string label ('entailment', 'contradiction', 'neutral', or 'unknown').
    """
    if not premise or not hypothesis:
        logger.warning("Premise or hypothesis is empty.")
        return "unknown"

    # Prepare input data in a DataFrame format expected by the BoW extractor/model
    input_df = pd.DataFrame({'premise_text': [premise], 'hypothesis_text': [hypothesis]})

    try:
        # DecisionTreeBowBaseline has extract_features and predict methods
        # 1. Extract features using the loaded model's extractor
        logger.debug("Extracting features for prediction...")
        # Assuming extract_features uses the fitted extractor within the loaded model
        features = model.extract_features(input_df)

        # Handle case where no features are extracted (e.g., empty vocabulary match)
        if features is None or features.shape[0] == 0 or features.shape[1] == 0:
            logger.warning("Feature extraction resulted in empty features. Cannot predict.")
            return "unknown"

        # 2. Predict using the loaded model's classifier
        logger.debug(f"Predicting with model on features of shape: {features.shape}")
        # The predict method uses the fitted classifier within the loaded model
        prediction_int = model.predict(features)  # Should return np.array([label_int])

        if prediction_int is None or len(prediction_int) == 0:
            logger.error("Model prediction returned None or empty array.")
            return "unknown"

        # Map integer prediction to string label
        predicted_label = label_map_reverse.get(prediction_int[0], "unknown")
        logger.info(f"Raw prediction: {prediction_int[0]}, Mapped label: {predicted_label}")
        return predicted_label

    except RuntimeError as e:
        logger.error(f"Runtime error during prediction (e.g., model not trained/loaded properly): {e}", exc_info=True)
        return "unknown"
    except ValueError as e:
        logger.error(f"Value error during prediction (e.g., feature mismatch): {e}", exc_info=True)
        return "unknown"
    except Exception as e:
        logger.error(f"An unexpected error occurred during prediction: {e}", exc_info=True)
        return "unknown"
