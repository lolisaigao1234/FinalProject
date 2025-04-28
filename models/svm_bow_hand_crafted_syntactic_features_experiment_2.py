# Create file: IS567FP/models/svm_bow_hand_crafted_syntactic_features_experiment_2.py
import logging
from typing import Optional

# Assuming SVMModel and CombinedFeatureExtractor are correctly defined in svm_bow_baseline
# Adjust the import path if they are in baseline_base or elsewhere.
from .svm_bow_baseline import SVMModel, CombinedFeatureExtractor, FeatureExtractor

logger = logging.getLogger(__name__)

class SVMBowHandCraftedSyntacticExperiment2(SVMModel):
    """
    Experiment 2: SVM combining BoW (lexical/statistical) and hand-crafted syntactic features.
    Inherits from SVMModel and uses the CombinedFeatureExtractor.
    Assumes features are pre-computed and loaded by the trainer.
    This is functionally similar to SVMWithBothFeatures but named explicitly for the experiment.
    """
    def __init__(self, kernel: str = 'linear', C: float = 1.0):
        """
        Initialize the Experiment 2 SVM model.

        Args:
            kernel (str): SVM kernel type ('linear', 'rbf', 'poly').
            C (float): SVM regularization parameter.
        """
        logger.info(f"Initializing Experiment 2 SVM with Combined Features (Kernel: {kernel}, C: {C})")
        # Explicitly pass the CombinedFeatureExtractor instance to the parent SVMModel
        super().__init__(feature_extractor=CombinedFeatureExtractor(), kernel=kernel, C=C)
        logger.debug(f"Assigned feature extractor: {self.feature_extractor.__class__.__name__}")

    @classmethod
    def load(cls, filepath: str, feature_extractor: Optional[FeatureExtractor] = None) -> 'SVMBowHandCraftedSyntacticExperiment2':
        """Loads the Experiment 2 SVM model."""
        # Ensure the correct extractor is used when loading
        if feature_extractor is None:
            # Default to the extractor this model uses
            feature_extractor = CombinedFeatureExtractor()
        elif not isinstance(feature_extractor, CombinedFeatureExtractor):
             logger.warning(f"Loading {cls.__name__} but received incorrect feature extractor type: {type(feature_extractor)}. Using CombinedFeatureExtractor instead.")
             feature_extractor = CombinedFeatureExtractor()

        logger.info(f"Loading Experiment 2 SVM model from {filepath}")
        # Use the parent class's load method, passing the specific extractor
        instance = super(SVMBowHandCraftedSyntacticExperiment2, cls).load(filepath, feature_extractor)

        # Ensure the loaded instance is of the correct type if super().load doesn't handle it
        if not isinstance(instance, cls):
             logger.warning(f"Loaded model type mismatch. Expected {cls.__name__}, got {type(instance)}. Re-instantiating.")
             # Manually create a new instance and copy attributes if needed
             re_instance = cls(kernel=instance.kernel, C=instance.C)
             re_instance.svm = instance.svm
             re_instance.is_trained = instance.is_trained
             re_instance.feature_cols = instance.feature_cols
             re_instance.feature_extractor = feature_extractor # Ensure correct extractor
             return re_instance

        instance.feature_extractor = feature_extractor # Ensure correct extractor upon load
        logger.info(f"Successfully loaded Experiment 2 SVM model.")
        return instance

# No changes needed for extract_features, train, predict, save as they are inherited
# and the CombinedFeatureExtractor handles selecting the appropriate columns.