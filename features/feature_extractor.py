import logging
from typing import List, Dict

import pandas as pd
import torch
from torch.utils.hipify.hipify_python import preprocessor
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from config import MODEL_NAME, MAX_SEQ_LENGTH
from utils.database import DatabaseHandler
from data.preprocessor import TextPreprocessor

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract both lexical and syntactic features for NLI tasks using downsampled datasets."""

    def __init__(self, db_handler: DatabaseHandler, preprocessor: TextPreprocessor):
        """Initialize feature extractor with BERT model."""
        self.db_handler = db_handler
        self.preprocessor = preprocessor
        self.suffix = preprocessor.suffix

        # Initialize BERT tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.bert_model = AutoModel.from_pretrained(MODEL_NAME)

        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert_model.to(self.device)

    def extract_features(
            self,
            dataset_name: str,
            split: str,
            sample_size: int,
            feature_types=None,
            force_recompute: bool = False
    ) -> pd.DataFrame:
        """Extract features from a downsampled dataset."""
        # Check if features already exist
        if feature_types is None:
            feature_types = ["lexical", "syntactic"]
        feature_name = "_".join(feature_types)
        feature_cache_name = f"features_{feature_name}_sample{sample_size}"

        if not force_recompute and self.db_handler.check_exists(
                dataset_name, split, feature_cache_name
        ):
            logger.info(f"Loading features for downsampled {dataset_name} {split} from database")
            return self.db_handler.load_dataframe(dataset_name, split, feature_cache_name)

        # Load pairs and parse trees from downsampled data
        logger.info(f"Loading downsampled {dataset_name} {split} data (sample size: {sample_size})")

        # Load the downsampled data with specific suffixes
        pairs = self.db_handler.load_dataframe(dataset_name, split, f"pairs_{self.suffix}")
        sentences = self.db_handler.load_dataframe(dataset_name, split, f"sentences_{self.suffix}")
        parse_trees = self.db_handler.load_dataframe(dataset_name, split, f"parse_trees_{self.suffix}")

        assert not pairs.empty, f"No downsampled pairs found for {dataset_name} {split}"
        assert not sentences.empty, f"No downsampled sentences found for {dataset_name} {split}"
        assert not parse_trees.empty, f"No downsampled parse trees found for {dataset_name} {split}"

        if pairs.empty or sentences.empty:
            logger.warning(f"No downsampled data found for {dataset_name} {split}. "
                           f"Make sure to run preprocessing on the downsampled data first.")
            return pd.DataFrame()

        # Join to get text and parse trees
        pairs_with_text = self._join_data(pairs, sentences, parse_trees)

        if pairs_with_text.empty:
            logger.warning(f"Failed to join data for {dataset_name} {split}")
            return pd.DataFrame()

        # Extract features
        features = []
        logger.info(f"Extracting features for {len(pairs_with_text)} downsampled pairs")

        for idx, row in tqdm(pairs_with_text.iterrows(), total=len(pairs_with_text)):
            pair_features = {
                "pair_id": row.get("id"),
                "label": row.get("label")
            }

            # Extract lexical features if requested
            if "lexical" in feature_types:
                lexical_features = self._extract_lexical_features(
                    row["premise_text"],
                    row["hypothesis_text"]
                )
                pair_features.update(lexical_features)

            # Extract syntactic features if requested
            if "syntactic" in feature_types and "premise_constituency" in row and "hypothesis_constituency" in row:
                syntactic_features = self._extract_syntactic_features(
                    row.get("premise_constituency", ""),
                    row.get("premise_dependency", ""),
                    row.get("hypothesis_constituency", ""),
                    row.get("hypothesis_dependency", "")
                )
                pair_features.update(syntactic_features)

            features.append(pair_features)

        # Convert to dataframe and store
        features_df = pd.DataFrame(features)
        self.db_handler.store_dataframe(
            features_df, dataset_name, split, feature_cache_name
        )
        logger.info(f"Stored {len(features_df)} features for downsampled {dataset_name} {split}")

        return features_df

    def _join_data(self, pairs, sentences, parse_trees):
        """Join pairs, sentences, and parse trees, with better error handling."""
        try:
            # Join pairs with sentences for premise
            pairs_with_text = pairs.merge(
                sentences, left_on="premise_id", right_on="id", how="left"
            ).rename(columns={"text": "premise_text"})

            # Join with sentences for hypothesis
            pairs_with_text = pairs_with_text.merge(
                sentences, left_on="hypothesis_id", right_on="id", how="left", suffixes=("", "_hyp")
            ).rename(columns={"text": "hypothesis_text"})

            # Optionally join with parse trees if available
            if not parse_trees.empty:
                # Join with parse trees for premise
                pairs_with_text = pairs_with_text.merge(
                    parse_trees, left_on="premise_id", right_on="sentence_id", how="left"
                ).rename(columns={
                    "constituency_tree": "premise_constituency",
                    "dependency_tree": "premise_dependency"
                })

                # Join with parse trees for hypothesis
                pairs_with_text = pairs_with_text.merge(
                    parse_trees, left_on="hypothesis_id", right_on="sentence_id", how="left", suffixes=("", "_hyp")
                ).rename(columns={
                    "constituency_tree": "hypothesis_constituency",
                    "dependency_tree": "hypothesis_dependency"
                })
            else:
                logger.warning("No parse trees available. Syntactic features will be limited.")
                pairs_with_text["premise_constituency"] = ""
                pairs_with_text["premise_dependency"] = ""
                pairs_with_text["hypothesis_constituency"] = ""
                pairs_with_text["hypothesis_dependency"] = ""

            return pairs_with_text

        except Exception as e:
            logger.error(f"Error joining data: {str(e)}")
            return pd.DataFrame()

    def _extract_lexical_features(self, premise_text: str, hypothesis_text: str) -> Dict:
        """Extract lexical features using BERT embeddings."""
        features = {}

        # BERT embeddings
        with torch.no_grad():
            # Tokenize
            inputs = self.tokenizer(
                [premise_text, hypothesis_text],
                max_length=MAX_SEQ_LENGTH,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get BERT embeddings
            outputs = self.bert_model(**inputs)

            # Use CLS token embeddings
            premise_embedding = outputs.last_hidden_state[0, 0, :].cpu().numpy()
            hypothesis_embedding = outputs.last_hidden_state[1, 0, :].cpu().numpy()

            # Store first 10 dimensions as features
            for i in range(10):
                features[f"premise_bert_{i}"] = float(premise_embedding[i])
                features[f"hypothesis_bert_{i}"] = float(hypothesis_embedding[i])

            # Compute difference and element-wise product
            diff_embedding = premise_embedding - hypothesis_embedding
            prod_embedding = premise_embedding * hypothesis_embedding

            for i in range(10):
                features[f"diff_bert_{i}"] = float(diff_embedding[i])
                features[f"prod_bert_{i}"] = float(prod_embedding[i])

        # Add text statistics
        features["premise_length"] = len(premise_text.split())
        features["hypothesis_length"] = len(hypothesis_text.split())
        features["length_diff"] = abs(features["premise_length"] - features["hypothesis_length"])
        features["length_ratio"] = (features["premise_length"] / features["hypothesis_length"]
                                    if features["hypothesis_length"] > 0 else 0)

        # Add word overlap metrics
        premise_words = set(premise_text.lower().split())
        hypothesis_words = set(hypothesis_text.lower().split())

        intersection = premise_words.intersection(hypothesis_words)
        union = premise_words.union(hypothesis_words)

        features["word_overlap_count"] = len(intersection)
        features["word_overlap_ratio"] = len(intersection) / len(union) if union else 0

        return features

    def _extract_syntactic_features(
            self,
            premise_constituency: str,
            premise_dependency: str,
            hypothesis_constituency: str,
            hypothesis_dependency: str
    ) -> Dict:
        """Extract syntactic features from parse trees."""
        features = {}

        # Extract features from constituency trees
        premise_const_features = self.preprocessor.extract_parse_features(
            premise_constituency, "constituency"
        )
        hypothesis_const_features = self.preprocessor.extract_parse_features(
            hypothesis_constituency, "constituency"
        )

        # Extract features from dependency trees
        premise_dep_features = self.preprocessor.extract_parse_features(
            premise_dependency, "dependency"
        )
        hypothesis_dep_features = self.preprocessor.extract_parse_features(
            hypothesis_dependency, "dependency"
        )

        # Add prefix to distinguish features
        for key, value in premise_const_features.items():
            features[f"premise_const_{key}"] = value

        for key, value in hypothesis_const_features.items():
            features[f"hypothesis_const_{key}"] = value

        for key, value in premise_dep_features.items():
            features[f"premise_dep_{key}"] = value

        for key, value in hypothesis_dep_features.items():
            features[f"hypothesis_dep_{key}"] = value

        # Compute feature differences
        for key in premise_const_features:
            p_val = premise_const_features.get(key, 0)
            h_val = hypothesis_const_features.get(key, 0)
            features[f"diff_const_{key}"] = p_val - h_val

        return features
