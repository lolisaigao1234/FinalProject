# File: models/predict_knn_bow_hand_crafted_syntactic_features_experiment_2.py
import os
import joblib
import pandas as pd
import numpy as np
import argparse
import logging
from typing import Dict, Any

# Import the corresponding model class
try:
    # Ensure the class name is correct
    from knn_bow_hand_crafted_syntactic_features_experiment_2 import KnnBowSyntacticExperiment2
    from config import MODELS_DIR
    # Again, syntactic prediction requires feature extraction logic or assumptions
    from features.feature_extractor import _extract_syntactic_features_row  # Example if needed

except ImportError:
    print("Error: Could not import KnnBowSyntacticExperiment2 or config.")


    class KnnBowSyntacticExperiment2:  # Dummy
        @classmethod
        def load(cls, *args): raise NotImplementedError("Dummy load")

        def predict(self, *args): raise NotImplementedError("Dummy predict")

        def extract_features(self, *args): raise NotImplementedError("Dummy extract")


    MODELS_DIR = "./saved_models"  # Dummy


    def _extract_syntactic_features_row(row):
        return pd.Series({})  # Dummy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LABEL_MAP_REVERSE: Dict[int, str] = {0: 'entailment', 1: 'contradiction', 2: 'neutral'}


def predict_nli(premise_data: Dict[str, Any], hypothesis_data: Dict[str, Any], model: Any,
                label_map_reverse: Dict[int, str]) -> str:
    """
    Makes a prediction using the loaded KNN Syntactic model.
    Similar to Exp1, assumes model.extract_features handles feature extraction
    based on stored model.feature_cols. Requires appropriate input data setup.
    """
    # Create DataFrame assuming extract_features will select based on stored feature_cols
    input_data = {
        'premise_text': [premise_data.get('text', '')],  # Needed?
        'hypothesis_text': [hypothesis_data.get('text', '')],  # Needed?
        'premise_constituency': [premise_data.get('constituency', '')],  # Needed?
        'premise_dependency': [premise_data.get('dependency', '')],  # Needed?
        'hypothesis_constituency': [hypothesis_data.get('constituency', '')],  # Needed?
        'hypothesis_dependency': [hypothesis_data.get('dependency', '')],  # Needed?
    }
    input_df = pd.DataFrame(input_data)

    try:
        logger.debug("Extracting features using model.extract_features...")
        features = model.extract_features(input_df)  # Uses SyntacticFeatureExtractor + stored cols

        if features is None or features.shape[0] == 0:
            logger.warning("Feature extraction returned empty. Cannot predict.")
            return "unknown"
        if model.feature_cols and features.shape[1] != len(model.feature_cols):
            logger.error(f"Extracted feature dimension ({features.shape[1]}) != expected ({len(model.feature_cols)}).")
            return "unknown"

        logger.debug(f"Predicting with model on features of shape: {features.shape}")
        prediction_int = model.predict(features)  # Uses fitted pipeline (scaler + KNN)

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
