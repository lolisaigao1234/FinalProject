# File: IS567FP/models/predict_logistic_tfidf_syntactic_experiment_3.py
import os
import joblib
import pandas as pd
import numpy as np
import argparse
import logging
from typing import Dict, Any

# Import the corresponding model class
try:
    from logistic_tfidf_syntactic_experiment_3 import LogisticTFIDFSyntacticExperiment3
    from config import MODELS_DIR
    # Syntactic part needs feature extraction logic or assumptions
    from features.feature_extractor import _extract_syntactic_features_row  # Example if needed

except ImportError:
    print("Error: Could not import LogisticTFIDFSyntacticExperiment3 or config.")


    class LogisticTFIDFSyntacticExperiment3:  # Dummy
        @classmethod
        def load(cls, *args): raise NotImplementedError("Dummy load")

        def predict(self, *args): raise NotImplementedError("Dummy predict")

        def predict_on_dataframe(self, *args): raise NotImplementedError("Dummy predict_on_dataframe")

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
    Makes a prediction using the loaded Logistic TFIDF + Syntactic model.
    Relies on the model's 'extract_features' or 'predict_on_dataframe' method,
    which uses the loaded+fitted feature pipeline.
    """
    # Create DataFrame with columns needed by the pipeline (text + potential syntactic source cols)
    input_data = {
        'premise_text': [premise_data.get('text', '')],
        'hypothesis_text': [hypothesis_data.get('text', '')],
        # Add potential syntactic source columns IF predict_on_dataframe doesn't handle it
        # 'premise_constituency': [premise_data.get('constituency', '')],
        # 'premise_dependency': [premise_data.get('dependency', '')],
        # 'hypothesis_constituency': [hypothesis_data.get('constituency', '')],
        # 'hypothesis_dependency': [hypothesis_data.get('dependency', '')],
    }
    input_df = pd.DataFrame(input_data)

    try:
        logger.debug("Predicting using model.predict_on_dataframe...")
        # This model has predict_on_dataframe which should handle extraction internally
        if hasattr(model, 'predict_on_dataframe'):
            prediction_int = model.predict_on_dataframe(input_df)
        else:
            # Fallback if predict_on_dataframe is missing: extract then predict
            logger.warning("predict_on_dataframe not found, using extract_features + predict.")
            features = model.extract_features(input_df)  # Uses fitted pipeline
            if features is None or features.shape[0] == 0:
                logger.warning("Feature extraction returned empty. Cannot predict.")
                return "unknown"
            prediction_int = model.predict(features)  # Uses fitted classifier

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
