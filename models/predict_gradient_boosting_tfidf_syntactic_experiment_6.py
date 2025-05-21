# File: models/predict_gradient_boosting_tfidf_syntactic_experiment_6.py
import os
import joblib
import pandas as pd
import numpy as np
import argparse
import logging
from typing import Dict, Any

# Import the corresponding model class
try:
    from gradient_boosting_tfidf_syntactic_experiment_6 import GradientBoostingTFIDFSyntacticExperiment6
    from config import MODELS_DIR
    # Syntactic part needs feature extraction logic or assumptions
    from features.feature_extractor import _extract_syntactic_features_row  # Example if needed

except ImportError:
    print("Error: Could not import GradientBoostingTFIDFSyntacticExperiment6 or config.")


    class GradientBoostingTFIDFSyntacticExperiment6:  # Dummy
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
    Makes a prediction using the loaded Gradient Boosting TFIDF + Syntactic model.
    Relies on the model's extract_features and predict methods.
    """
    # Create DataFrame with columns needed by the pipeline
    input_data = {
        'premise_text': [premise_data.get('text', '')],
        'hypothesis_text': [hypothesis_data.get('text', '')],
        # Add potential syntactic source columns IF needed
        # 'premise_constituency': [premise_data.get('constituency', '')],
        # ... etc ...
    }
    input_df = pd.DataFrame(input_data)

    try:
        logger.debug("Extracting features using model.extract_features...")
        # extract_features uses the fitted feature_pipeline
        features = model.extract_features(input_df)

        if features is None or features.shape[0] == 0:
            logger.warning("Feature extraction returned empty. Cannot predict.")
            return "unknown"

        logger.debug(f"Predicting with model on features of shape: {features.shape}")
        # predict uses the fitted classifier
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
