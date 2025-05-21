# File: models/predict_decision_tree_hand_crafted_syntactic_features_experiment_1.py
import os
import joblib
import pandas as pd
import numpy as np
import argparse
import logging
from typing import Dict, Any

# Import the corresponding model class
try:
    from decision_tree_hand_crafted_syntactic_features_experiment_1 import DecisionTreeSyntacticExperiment1
    from config import MODELS_DIR
    # We might need feature extraction helpers if prediction requires raw text -> features
    # However, Exp1 assumes features are precomputed and loaded during training/eval.
    # For prediction on new text, we'd NEED the feature extraction logic (e.g., Stanza) here.
    # Let's assume for now prediction takes PRE-COMPUTED features as input.
    # If prediction needs to run on raw text, this script needs major changes.
    # --- Alternative assumption: The model's predict method can handle a DataFrame ---
    # --- containing the necessary raw text/parse columns, and it calls its internal ---
    # --- feature extractor. This seems more likely based on the structure.        ---
    from features.feature_extractor import _extract_syntactic_features_row  # Example if needed

except ImportError:
    print("Error: Could not import DecisionTreeSyntacticExperiment1 or config.")


    class DecisionTreeSyntacticExperiment1:  # Dummy
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
    Makes a prediction using the loaded Decision Tree Syntactic model.
    Assumes input includes necessary data for syntactic feature extraction if needed by the model,
    or that the model's extract_features method handles it based on stored feature_cols.
    For simplicity now, it assumes the model's predict can work on features extracted
    from a DataFrame containing only the necessary columns (e.g., syntactic features).
    If raw text prediction is needed, this needs the Stanza pipeline integrated.

    Args:
        premise_data: Dict containing premise info (e.g., {'text': '...', 'constituency': '...', 'dependency': '...' })
                      *Modify based on what extract_features actually needs.*
        hypothesis_data: Dict containing hypothesis info.
        model: The loaded NLI model instance.
        label_map_reverse: Dictionary mapping integer predictions to string labels.

    Returns:
        The predicted string label.
    """
    # --- THIS IS THE COMPLEX PART FOR FEATURE-BASED MODELS ---
    # How do we get the features for the new premise/hypothesis?
    # Option 1: Assume user provides pre-computed features matching `model.feature_cols`. (Unlikely)
    # Option 2: Assume the model needs raw text/parses and run the *exact* feature extraction logic here. (Requires Stanza etc.)
    # Option 3: Assume the model's `extract_features` method can take a basic DataFrame
    #           and extract features based on its *stored* `feature_cols` list. This relies
    #           on the `extract_features` implementation correctly selecting/computing based on name.

    # Let's proceed with Option 3, assuming `model.extract_features` can work with a DataFrame
    # containing the necessary source columns (even if just dummy values for text if only syntactic are used).
    # We need to know which columns the SyntacticFeatureExtractor actually uses.
    # Based on baseline_base.py, SyntacticFeatureExtractor just selects columns ending in _const_, _dep_, etc.
    # So, we need to *simulate* having those columns in the input DataFrame for prediction.

    # This is problematic if the features aren't pre-computed.
    # A practical prediction script for Exp1 would need to:
    # 1. Run Stanza on the input premise/hypothesis.
    # 2. Call _extract_syntactic_features to get the dictionary of features.
    # 3. Create a DataFrame from this dictionary.
    # 4. Ensure the DataFrame columns match model.feature_cols (add missing as 0).
    # 5. Call model.predict() on the feature array.

    # --- Simplified approach (assuming predict works on features extracted based on stored names): ---
    # Create a DataFrame with the necessary *potential* columns (text/parses).
    # The model's extract_features should then use model.feature_cols to select the actual needed ones.
    input_data = {
        'premise_text': [premise_data.get('text', '')],  # Needed? Depends on extractor impl.
        'hypothesis_text': [hypothesis_data.get('text', '')],  # Needed?
        # Add potentially needed parse columns if extract_features requires them
        'premise_constituency': [premise_data.get('constituency', '')],
        'premise_dependency': [premise_data.get('dependency', '')],
        'hypothesis_constituency': [hypothesis_data.get('constituency', '')],
        'hypothesis_dependency': [hypothesis_data.get('dependency', '')],
    }
    input_df = pd.DataFrame(input_data)

    try:
        logger.debug("Extracting features using model.extract_features...")
        # This should use the SyntacticFeatureExtractor inside the model,
        # which uses the `model.feature_cols` stored during training.
        features = model.extract_features(input_df)

        if features is None or features.shape[0] == 0:
            logger.warning("Feature extraction returned empty. Cannot predict.")
            return "unknown"
        # Ensure features match expected dimensions based on model.feature_cols
        if model.feature_cols and features.shape[1] != len(model.feature_cols):
            logger.error(f"Extracted feature dimension ({features.shape[1]}) != expected ({len(model.feature_cols)}).")
            return "unknown"

        logger.debug(f"Predicting with model on features of shape: {features.shape}")
        # The predict method uses the fitted pipeline (scaler + classifier)
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
