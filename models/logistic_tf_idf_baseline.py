# IS567FP/models/logistic_tf_idf_baseline.py
import logging
import numpy as np
from typing import Optional, Tuple
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Inherit from the new base class
from .baseline_base import TextBaselineModel, TextFeatureExtractorBase

logger = logging.getLogger(__name__)

class TFIDFExtractor(TextFeatureExtractorBase):
    """Specialized TF-IDF extractor."""
    def __init__(self, max_features: Optional[int] = 10000, ngram_range: Tuple[int, int] = (1, 2), **kwargs):
        # Pass TfidfVectorizer and relevant args to base
        super().__init__(
            TfidfVectorizer,
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            **kwargs # Pass any other TfidfVectorizer args
        )

    def transform(self, data: pd.DataFrame, text_col1: str = 'premise_text', text_col2: str = 'hypothesis_text') -> np.ndarray:
        """Transforms premise and hypothesis text into TF-IDF features (concatenated)."""
        if not self.is_fitted:
            raise RuntimeError("TF-IDF vectorizer must be fitted before transforming.")
        if text_col1 not in data.columns or text_col2 not in data.columns:
            raise ValueError(f"DataFrame must contain '{text_col1}' and '{text_col2}' columns.")

        logger.debug(f"Transforming {len(data)} samples with TF-IDF...")
        premise_tfidf = self.vectorizer.transform(data[text_col1].fillna(''))
        hypothesis_tfidf = self.vectorizer.transform(data[text_col2].fillna(''))

        # Concatenate sparse matrices efficiently if possible, otherwise convert to dense
        # For compatibility with Logistic Regression, dense might be simpler here.
        features = np.concatenate([premise_tfidf.toarray(), hypothesis_tfidf.toarray()], axis=1)
        logger.debug(f"TF-IDF transformation complete. Feature shape: {features.shape}")
        return features


class LogisticTFIDFBaseline(TextBaselineModel):
    """Baseline NLI model using Logistic Regression with TF-IDF features."""

    def __init__(self, C: float = 1.0, max_iter: int = 1000, max_features: Optional[int] = 10000, ngram_range: Tuple[int, int] = (1, 2)):
        extractor = TFIDFExtractor(max_features=max_features, ngram_range=ngram_range)
        model_instance = LogisticRegression(C=C, max_iter=max_iter, solver='liblinear', random_state=42)
        super().__init__(extractor, model_instance)
        # Store specific params if needed later
        self.C = C
        self.max_iter = max_iter

    def extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extracts TF-IDF features using the assigned extractor."""
        if not self.extractor.is_fitted:
            raise RuntimeError("TF-IDF extractor must be fitted or loaded first.")
        return self.extractor.transform(data)

    # train, predict, save, load are inherited from TextBaselineModel
    # load_raw_text_data static method is now in TextBaselineModel

# Removed LogisticRegressionTrainer class and _evaluate_model helper