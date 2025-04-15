# data/preprocessor_nn.py
import logging
from typing import Optional

from data.preprocessor import TextPreprocessor
from utils.database import DatabaseHandler

logger = logging.getLogger(__name__)


class NeuralPreprocessor(TextPreprocessor):
    """Text preprocessor specialized for neural network models (BERT with syntactic features)"""

    def __init__(self, db_handler: DatabaseHandler, sample_size: Optional[int] = None):
        super().__init__(db_handler, sample_size)

    def _initialize_feature_extractor(self) -> None:
        """Override to use FeatureExtractorNN instead of the base feature extractor."""
        if not hasattr(self, '_feature_extractor') or self._feature_extractor is None:
            from features.feature_extractor_nn import FeatureExtractorNN
            self._feature_extractor = FeatureExtractorNN(self.db_handler, self)
            logger.info("Initialized FeatureExtractorNN for neural preprocessing")

    def preprocess_neural_dataset(self, dataset_name: str, sample_size: int,
                                  force_reprocess: bool) -> None:
        """Process dataset specifically for neural network training."""
        logger.info(f"Starting neural preprocessing pipeline for {dataset_name}")

        # Use the existing pipeline but with the neural-specific feature extractor
        self.preprocess_dataset_pipeline(
            dataset_name=dataset_name,
            total_sample_size=sample_size,
            force_reprocess=force_reprocess
        )

        logger.info(f"Completed neural preprocessing for {dataset_name}")
