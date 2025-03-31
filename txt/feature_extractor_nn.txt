# features/feature_extractor_nn.py
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

from utils.common import NLPBaseComponent
from config import MODEL_NAME, SYNTACTIC_FEATURE_DIM


class FeatureExtractorNN(NLPBaseComponent):
    def __init__(self, db_handler, preprocessor=None):
        super().__init__(db_handler)
        self.preprocessor = preprocessor
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def extract_features(self, dataset_name, split, force_recompute=False, sample_size=None):
        """Extract both lexical and syntactic features for NLI data"""
        self.logger.info(f"Extracting features for {dataset_name} {split}")

        suffix = f"sample{sample_size}" if sample_size else "data"

        # Check if features already exist and force_recompute is False
        if not force_recompute and self.db_handler.check_exists(dataset_name, split,
                                                                f"features_lexical_syntactic_{suffix}"):
            self.logger.info(f"Features already exist for {dataset_name} {split}")
            return

        # Load data
        self.logger.info(f"Loading parse trees and sentence pairs for {dataset_name} {split}")
        parse_trees_df = self.db_handler.load_dataframe(dataset_name, split, f"parse_trees_{suffix}")
        pairs_df = self.db_handler.load_dataframe(dataset_name, split, f"pairs_{suffix}")
        sentences_df = self.db_handler.load_dataframe(dataset_name, split, f"sentences_{suffix}")

        if parse_trees_df.empty or pairs_df.empty or sentences_df.empty:
            self.logger.error(f"Missing required data for feature extraction: {dataset_name} {split}")
            return

        # Merge sentence data with parse trees
        sentences_with_trees = pd.merge(
            sentences_df,
            parse_trees_df,
            left_on="id",
            right_on="sentence_id",
            how="inner"
        )

        # Create a mapping from sentence ID to text
        id_to_text = dict(zip(sentences_df["id"], sentences_df["text"]))

        # Extract features for each pair
        features_list = []

        self.logger.info(f"Extracting features for {len(pairs_df)} sentence pairs")
        for idx, pair in tqdm(pairs_df.iterrows(), total=len(pairs_df)):
            premise_id = pair["premise_id"]
            hypothesis_id = pair["hypothesis_id"]

            # Get corresponding sentences and parse trees
            premise_row = sentences_with_trees[sentences_with_trees["id"] == premise_id]
            hypothesis_row = sentences_with_trees[sentences_with_trees["id"] == hypothesis_id]

            if premise_row.empty or hypothesis_row.empty:
                self.logger.warning(f"Missing parse tree for pair {pair['id']}")
                continue

            # Get the text
            premise_text = id_to_text[premise_id]
            hypothesis_text = id_to_text[hypothesis_id]

            # Extract BERT features (tokenization)
            encoded = self.tokenizer(
                premise_text,
                hypothesis_text,
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )

            # Extract syntactic features
            premise_features = self._extract_syntactic_features(
                premise_row["constituency_tree"].iloc[0],
                premise_row["dependency_tree"].iloc[0]
            )

            hypothesis_features = self._extract_syntactic_features(
                hypothesis_row["constituency_tree"].iloc[0],
                hypothesis_row["dependency_tree"].iloc[0]
            )

            # Create feature entry
            feature_entry = {
                "pair_id": pair["id"],
                "input_ids": encoded["input_ids"][0].tolist(),
                "attention_mask": encoded["attention_mask"][0].tolist(),
                "token_type_ids": encoded["token_type_ids"][0].tolist(),
                "premise_features": premise_features.tolist(),
                "hypothesis_features": hypothesis_features.tolist(),
                "label": pair["label"]
            }

            features_list.append(feature_entry)

        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)

        # Store in database
        self.logger.info(f"Saving {len(features_df)} feature entries to database")
        self.db_handler.store_dataframe(
            features_df,
            dataset_name,
            split,
            f"features_lexical_syntactic_{suffix}"
        )

        self.logger.info(f"Feature extraction completed for {dataset_name} {split}")

    def _extract_syntactic_features(self, constituency_tree, dependency_tree):
        """Extract fixed-dimension features from syntactic parse trees"""
        # Convert string representation back to data structure if needed
        if isinstance(dependency_tree, str) and dependency_tree.startswith("[{"):
            import ast
            try:
                dependency_list = ast.literal_eval(dependency_tree)
            except SyntaxError:
                self.logger.warning("Could not parse dependency tree string")
                dependency_list = []
        else:
            dependency_list = []

        # Extract feature vector
        # This is a simplified version - in a real implementation, you'd extract
        # meaningful syntactic features from the parse trees

        # Initialize a feature vector of the specified dimension
        feature_vector = np.zeros(SYNTACTIC_FEATURE_DIM)

        # Fill with some basic features based on parse tree properties
        # (This is placeholder logic - actual implementation would be more sophisticated)

        # 1. Basic tree statistics
        tree_depth = constituency_tree.count("(") // 2
        feature_vector[0] = min(tree_depth, 10) / 10  # Normalize

        # 2. POS tag distribution (example)
        pos_tags = [dep.get("pos", "") for dep in dependency_list if isinstance(dep, dict)]
        pos_set = {"NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP", "CONJ", "PART", "NUM"}

        for i, pos in enumerate(pos_set):
            if i < SYNTACTIC_FEATURE_DIM - 10:
                feature_vector[i + 10] = pos_tags.count(pos) / max(len(pos_tags), 1)

        # 3. Dependency relation features
        dep_relations = [dep.get("deprel", "") for dep in dependency_list if isinstance(dep, dict)]
        common_deps = {"nsubj", "obj", "iobj", "nmod", "amod", "aux", "cop", "conj"}

        for i, dep in enumerate(common_deps):
            if i + 20 < SYNTACTIC_FEATURE_DIM:
                feature_vector[i + 20] = dep_relations.count(dep) / max(len(dep_relations), 1)

        # Convert to tensor
        return torch.tensor(feature_vector, dtype=torch.float)
