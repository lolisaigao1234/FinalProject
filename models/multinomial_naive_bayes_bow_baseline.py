# IS567FP/models/multinomial_naive_bayes_baseline.py
import logging
import numpy as np
from typing import Optional, Tuple
import pandas as pd
from scipy.sparse import hstack, csr_matrix # Import sparse matrix tools

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Inherit from the new base class
from .baseline_base import TextBaselineModel, TextFeatureExtractorBase

logger = logging.getLogger(__name__)

class BoWExtractor(TextFeatureExtractorBase):
    """Specialized Bag-of-Words extractor using CountVectorizer."""
    def __init__(self, max_features: Optional[int] = 10000, ngram_range: Tuple[int, int] = (1, 1), binary=False, **kwargs):
         super().__init__(
              CountVectorizer,
              max_features=max_features,
              ngram_range=ngram_range,
              stop_words='english',
              binary=binary, # Often False for MNB, True for some other models
              **kwargs
         )

    def transform(self, data: pd.DataFrame, text_col1: str = 'premise_text', text_col2: str = 'hypothesis_text') -> csr_matrix:
         """Transforms text into sparse BoW features (concatenated)."""
         if not self.is_fitted:
              raise RuntimeError("BoW vectorizer must be fitted before transforming.")
         if text_col1 not in data.columns or text_col2 not in data.columns:
              raise ValueError(f"DataFrame must contain '{text_col1}' and '{text_col2}' columns.")

         logger.debug(f"Transforming {len(data)} samples with BoW...")
         premise_bow = self.vectorizer.transform(data[text_col1].fillna(''))
         hypothesis_bow = self.vectorizer.transform(data[text_col2].fillna(''))

         # Concatenate sparse matrices horizontally
         features = hstack([premise_bow, hypothesis_bow], format='csr')
         logger.debug(f"BoW transformation complete. Feature shape: {features.shape}")
         return features


class MultinomialNaiveBayesBaseline(TextBaselineModel):
    """Baseline NLI model using Multinomial Naive Bayes with Bag-of-Words features."""

    def __init__(self, alpha: float = 1.0, max_features: Optional[int] = 10000, ngram_range: Tuple[int, int] = (1, 1)):
        extractor = BoWExtractor(max_features=max_features, ngram_range=ngram_range)
        model_instance = MultinomialNB(alpha=alpha)
        super().__init__(extractor, model_instance)
        self.alpha = alpha # Store specific param

    def extract_features(self, data: pd.DataFrame) -> csr_matrix: # Return sparse matrix
        """Extracts BoW features using the assigned extractor."""
        if not self.extractor.is_fitted:
            raise RuntimeError("BoW extractor must be fitted or loaded first.")
        # MNB typically works well with concatenated premise+hypothesis features
        # or separate features. The current extractor concatenates them.
        return self.extractor.transform(data)

    # train, predict, save, load are inherited from TextBaselineModel
    # load_raw_text_data static method is now in TextBaselineModel

# Removed MultinomialNaiveBayesTrainer class