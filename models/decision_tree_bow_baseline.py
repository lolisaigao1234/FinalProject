# models/decision_tree_bow_baseline.py
import logging
import time
import pandas as pd
import numpy as np
import joblib
import os
from typing import Optional, Dict, Any

# Scikit-learn components
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer

# Project-specific imports
from utils.common import NLIModel
from utils.database import DatabaseHandler

# Import base utilities
# Ensure TextBaselineModel is imported correctly, even if we don't call its __init__
from .baseline_base import TextBaselineModel, SimpleParquetLoader

logger = logging.getLogger(__name__)


# MODIFICATION: Change inheritance to TextBaselineModel
class DecisionTreeBowBaseline(TextBaselineModel):
    """Decision Tree baseline using Bag-of-Words features."""
    MODEL_NAME = "DecisionTree_BoW_Baseline"

    # MODIFICATION: Update __init__ signature and call super().__init__
    def __init__(self, args: Optional[object] = None,
                 max_features: int = 10000, max_depth: Optional[int] = None, random_state: int = 42, **kwargs):
        # Handle args object if passed
        if args:
            max_features = getattr(args, 'bow_max_features', getattr(args, 'max_features', max_features))
            max_depth = getattr(args, 'max_depth', max_depth)
            random_state = getattr(args, 'random_state', random_state)

        # Create the Scikit-learn vectorizer and model instances internally
        # These will be passed to the parent class constructor
        _vectorizer = CountVectorizer(max_features=max_features, lowercase=True, ngram_range=(1, 1))
        _model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)

        # MODIFICATION: Call the parent class's __init__ method
        # Note: TextBaselineModel expects an 'extractor' which should behave like TextFeatureExtractorBase
        #       CountVectorizer doesn't directly inherit from it, but we can wrap it or adapt TextBaselineModel.
        #       For now, let's pass the vectorizer directly, assuming TextBaselineModel's methods
        #       (like save/load) can handle it or are overridden here.
        #       We also need to assign the vectorizer to self.extractor for TextBaselineModel's methods to find it.
        super().__init__(extractor=_vectorizer, model_instance=_model)  # Pass the created instances

        # Assign the vectorizer to the 'extractor' attribute expected by TextBaselineModel methods like save/load
        # (super().__init__ might not do this depending on its implementation)
        # TextBaselineModel's __init__ already assigns these, so this line might be redundant
        # self.extractor = _vectorizer
        # self.model = _model # super().__init__ should handle this

        # Keep these if needed, __init__ in TextBaselineModel doesn't set them
        self.loader = SimpleParquetLoader()
        self.db_handler = DatabaseHandler()
        # self.is_trained is handled by TextBaselineModel.__init__

        # Remove internal _vectorizer and _model if super().__init__ handles assignment
        # del self._vectorizer
        # del self._model

        logger.debug(f"{self.MODEL_NAME} initialized with max_features={max_features}, max_depth={max_depth}")

    # --- IMPORTANT: Ensure methods like extract_features, train, predict, save, load ---
    # --- now correctly use self.extractor (the CountVectorizer) and self.model (the DecisionTree) ---
    # --- which were passed to and set by the super().__init__ call.                ---
    # --- Review the existing methods in DecisionTreeBowBaseline and ensure they      ---
    # --- are compatible with how TextBaselineModel expects them to work, or override ---
    # --- them completely if necessary.                                            ---

    # Example: Ensure extract_features uses self.extractor
    def extract_features(self, data: pd.DataFrame) -> Optional[Any]:
        logger.info(f"Extracting BoW features using self.extractor...")
        # Ensure extractor is fitted (CountVectorizer needs vocabulary_)
        if not hasattr(self.extractor, 'vocabulary_') or not self.extractor.vocabulary_:
            logger.error("Vectorizer (self.extractor) has not been fitted.")
            raise RuntimeError("Vectorizer (self.extractor) must be fitted before calling extract_features.")

        premise_col = 'premise' if 'premise' in data.columns else 'premise_text'
        hypothesis_col = 'hypothesis' if 'hypothesis' in data.columns else 'hypothesis_text'
        data[premise_col] = data[premise_col].fillna('')
        data[hypothesis_col] = data[hypothesis_col].fillna('')
        combined_text = data[premise_col] + " " + data[hypothesis_col]
        features = self.extractor.transform(combined_text)  # Use self.extractor
        return features

    # Example: Ensure train uses self.model
    def train(self, X: Any, y: np.ndarray) -> None:
        if X is None or y is None:
            raise ValueError("Features (X) or labels (y) are None.")
        if hasattr(X, "toarray"):
            X_dense = X.toarray()
        else:
            X_dense = X

        logger.info(f"Training Decision Tree model (self.model)...")
        start_time = time.time()
        self.model.fit(X_dense, y)  # Use self.model
        self.is_trained = True  # Set is_trained flag from parent class
        train_time = time.time() - start_time
        logger.info(f"Training finished in {train_time:.2f}s.")

    # ... (Review and potentially adjust predict, save, load, evaluate similarly) ...
    # Make sure save/load handle both self.model and self.extractor correctly,
    # potentially leveraging TextBaselineModel's save/load if compatible.

    # Keep the evaluate method, ensuring it uses self.loader, self.extract_features, self.predict correctly
    def evaluate(self, dataset_name: str, split: str, suffix: str) -> Optional[Dict[str, Any]]:
        # ... (Implementation as provided in the file, double-check it uses self.extractor/self.model) ...
        # Ensure clean_dataset is imported or accessible
        from .baseline_base import clean_dataset  # Make sure import is correct

        if not self.is_trained:  # Check flag set by parent
            logger.error(f"Cannot evaluate {self.MODEL_NAME}. Model is not trained.")
            return None
        if not hasattr(self.extractor, 'vocabulary_'):
            logger.error(f"Cannot evaluate {self.MODEL_NAME}. Extractor is not fitted.")
            return None

        logger.info(f"Evaluating {self.MODEL_NAME} on {dataset_name}/{split} ({suffix})")
        # ... rest of the evaluation logic ...

        try:
            df_eval_raw = self.loader.load_data(dataset_name, split, suffix)
            # ... data loading error handling ...
        except Exception as e:
            logger.error(f"Error loading eval data: {e}", exc_info=True)
            return None

        cleaned_data = clean_dataset(df_eval_raw)
        # ... data cleaning error handling ...
        df_eval_cleaned, y_true = cleaned_data
        # ... more checks ...

        try:
            X_eval = self.extract_features(df_eval_cleaned)  # Uses self.extractor
            # ... feature extraction error handling ...
        except Exception as e:
            logger.error(f"Error extracting eval features: {e}", exc_info=True)
            return None

        try:
            y_pred = self.predict(X_eval)  # Uses self.model
        except Exception as e:
            logger.error(f"Error predicting during eval: {e}", exc_info=True)
            return None

        # ... metrics calculation ...
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
        metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
        # ... add timing etc. ...
        logger.info(f"Evaluation complete. Metrics: {metrics}")
        return metrics


    # --- If TextBaselineModel's save/load are suitable, you might not need custom ones ---
    # --- Otherwise, keep the custom save/load methods from decision_tree_bow_baseline.py ---
    # --- ensuring they save/load self.model and self.extractor. ---

    # Keep the custom save method (adapts TextBaselineModel's structure)
    def save(self, directory: str, model_name: str) -> None:
        # Uses self.is_trained (from parent), self.model, self.extractor
        if not self.is_trained and (not hasattr(self.extractor, 'vocabulary_') or not self.extractor.vocabulary_):
            logger.warning(f"Attempting to save {self.MODEL_NAME} where neither model nor vectorizer is fitted.")
        # ... (rest of save logic from decision_tree_bow_baseline.py)
        os.makedirs(directory, exist_ok=True)
        model_path = os.path.join(directory, f"{model_name}_model.joblib")
        extractor_path = os.path.join(directory, f"{model_name}_extractor.joblib")
        if self.is_trained:
            joblib.dump(self.model, model_path)
            logger.info(f"Model saved to {model_path}")
        else:
            logger.warning("Model not trained, skipping model save.")
        if hasattr(self.extractor, 'vocabulary_') and self.extractor.vocabulary_:
            joblib.dump(self.extractor, extractor_path)
            logger.info(f"Extractor saved to {extractor_path}")
        else:
            logger.warning("Extractor not fitted, skipping extractor save.")


    # Keep the custom load classmethod
    @classmethod
    def load(cls, directory: str, model_name: str) -> 'DecisionTreeBowBaseline':
        # ... (load logic from decision_tree_bow_baseline.py) ...
        model_path = os.path.join(directory, f"{model_name}_model.joblib")
        extractor_path = os.path.join(directory, f"{model_name}_extractor.joblib")

        loaded_model = None
        loaded_extractor = None

        if not os.path.exists(extractor_path):
            raise FileNotFoundError(f"Extractor file not found at {extractor_path}. Cannot load model.")
        loaded_extractor = joblib.load(extractor_path)

        if os.path.exists(model_path):
            loaded_model = joblib.load(model_path)
        else:
            logger.warning(f"Model file not found at {model_path}. Model will not be loaded.")

        # Instantiate using default __init__ which now calls super()
        # We need a way to pass the loaded components WITHOUT calling the logic inside __init__ again
        # Option 1: Add flags/args to __init__ to skip creation (complex)
        # Option 2: Create instance then overwrite attributes (simpler)
        instance = cls()  # Create a basic instance

        # Overwrite attributes with loaded ones
        instance.extractor = loaded_extractor
        instance.model = loaded_model
        instance.is_trained = loaded_model is not None  # Set is_trained based on model load

        # Re-assign internal references if they exist in __init__
        # if hasattr(instance, '_vectorizer'): instance._vectorizer = loaded_extractor
        # if hasattr(instance, '_model'): instance._model = loaded_model

        logger.info(f"{cls.MODEL_NAME} loaded. Trained: {instance.is_trained}")
        return instance

    # --- ADD the load_raw_text_data static method ---
    #     It's inherited from TextBaselineModel, so no need to redefine unless overriding
    # @staticmethod
    # def load_raw_text_data(dataset_name: str, split: str, suffix: str, db_handler: DatabaseHandler) -> Optional[pd.DataFrame]:
    #     # This method is now inherited from TextBaselineModel
    #     # If you need specific logic for DecisionTreeBow, override it here.
    #     # Otherwise, remove this comment block.
    #     # Example of calling the parent's implementation if needed:
    #     # return super(DecisionTreeBowBaseline, cls).load_raw_text_data(dataset_name, split, suffix, db_handler)
    #     pass # Remove this if not overriding
