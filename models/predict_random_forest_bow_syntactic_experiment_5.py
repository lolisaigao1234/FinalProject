# File: models/predict_random_forest_bow_syntactic_experiment_5.py
import os
import joblib
import pandas as pd
import numpy as np
import argparse
import logging
from typing import Dict, Any

# Import the corresponding model class
try:
    from random_forest_bow_syntactic_experiment_5 import RandomForestBowSyntacticExperiment5
    from config import MODELS_DIR
    # Syntactic part needs feature extraction logic or assumptions
    from features.feature_extractor import _extract_syntactic_features_row  # Example if needed

except ImportError:
    print("Error: Could not import RandomForestBowSyntacticExperiment5 or config.")


    class RandomForestBowSyntacticExperiment5:  # Dummy
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
    Makes a prediction using the loaded Random Forest BoW + Syntactic model.
    Relies on the model's predict method which handles DataFrame input.
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
        logger.debug("Predicting using model.predict...")
        # This model's predict method handles DataFrame input and internal feature extraction
        prediction_int = model.predict(input_df)

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
