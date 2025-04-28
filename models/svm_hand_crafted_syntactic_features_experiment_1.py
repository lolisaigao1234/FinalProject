# Create file: IS567FP/models/svm_hand_crafted_syntactic_features_experiment_1.py
# Dependencies: SVMModel and SyntacticFeatureExtractor are likely in svm_bow_baseline.py or baseline_base.py
# Adjust import path if necessary.

import logging
from typing import Optional

# Assuming SVMModel and SyntacticFeatureExtractor are correctly defined in svm_bow_baseline
# If they are in baseline_base or elsewhere, adjust the import path.
from .svm_bow_baseline import SVMModel, SyntacticFeatureExtractor, FeatureExtractor

logger = logging.getLogger(__name__)

class SVMHandcraftedSyntacticExperiment1(SVMModel):
    """
    Experiment 1: SVM using only hand-crafted syntactic features.
    Inherits from SVMModel and uses the SyntacticFeatureExtractor.
    Assumes features are pre-computed and loaded by the trainer.
    """
    def __init__(self, kernel: str = 'linear', C: float = 1.0):
        """
        Initialize the Experiment 1 SVM model.

        Args:
            kernel (str): SVM kernel type ('linear', 'rbf', 'poly').
            C (float): SVM regularization parameter.
        """
        logger.info(f"Initializing Experiment 1 SVM with Syntactic Features (Kernel: {kernel}, C: {C})")
        # Explicitly pass the SyntacticFeatureExtractor instance to the parent SVMModel
        super().__init__(feature_extractor=SyntacticFeatureExtractor(), kernel=kernel, C=C)
        logger.debug(f"Assigned feature extractor: {self.feature_extractor.__class__.__name__}")

    @classmethod
    def load(cls, filepath: str, feature_extractor: Optional[FeatureExtractor] = None) -> 'SVMHandcraftedSyntacticExperiment1':
        """Loads the Experiment 1 SVM model."""
        # Ensure the correct extractor is used when loading
        if feature_extractor is None:
            feature_extractor = SyntacticFeatureExtractor()
        # Use the parent class's load method, passing the specific extractor
        # The returned object needs to be cast or reconstructed if SVMModel.load doesn't return cls()
        logger.info(f"Loading Experiment 1 SVM model from {filepath}")
        # Re-implementing load slightly to ensure correct class instantiation
        instance = super(SVMHandcraftedSyntacticExperiment1, cls).load(filepath, feature_extractor)
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
        logger.info(f"Successfully loaded Experiment 1 SVM model.")
        return instance

    # Inherits extract_features, train, predict, save from SVMModel.
    # The key is that self.feature_extractor is set to SyntacticFeatureExtractor.