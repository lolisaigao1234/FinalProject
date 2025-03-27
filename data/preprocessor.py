# data/preprocessor.py
# import logging
# from typing import List, Dict, Tuple, Optional
#
# import pandas as pd
# import numpy as np
# import stanza
# from tqdm import tqdm
#
# from config import STANZA_PROCESSORS, STANZA_LANG
# from utils.database import DatabaseHandler
#
# logger = logging.getLogger(__name__)
#
#
# class TextPreprocessor:
#     """Preprocess text for NLI tasks using Stanza for syntactic parsing."""
#
#     def __init__(self, db_handler: DatabaseHandler):
#         """Initialize preprocessor with Stanza pipeline."""
#         self.db_handler = db_handler
#
#         # Initialize Stanza pipeline
#         try:
#             self.nlp = stanza.Pipeline(
#                 lang=STANZA_LANG,
#                 processors=STANZA_PROCESSORS,
#                 use_gpu=True,
#                 verbose=False
#             )
#             logger.info(f"Stanza pipeline initialized with processors: {STANZA_PROCESSORS}")
#         except Exception as e:
#             logger.error(f"Failed to initialize Stanza pipeline: {str(e)}")
#             logger.info("Downloading Stanza models...")
#             stanza.download(STANZA_LANG)
#             self.nlp = stanza.Pipeline(
#                 lang=STANZA_LANG,
#                 processors=STANZA_PROCESSORS,
#                 use_gpu=True,
#                 verbose=False
#             )
#
#     def preprocess_dataset(
#             self,
#             dataset_name: str,
#             split: str,
#             sample_size: int = None,
#             force_reprocess: bool = False
#     ) -> pd.DataFrame:
#         """Preprocess a dataset with syntactic parsing."""
#         # Add sample suffix to cache keys if needed
#         suffix = f"_sample{sample_size}" if sample_size else ""
#
#         # Check if already processed with appropriate suffix
#         if not force_reprocess and self.db_handler.check_exists(dataset_name, split, f"parse_trees{suffix}"):
#             logger.info(f"Loading preprocessed {dataset_name} {split} from database")
#             return self.db_handler.load_dataframe(dataset_name, split, f"parse_trees{suffix}")
#
#         # Load sentences with sample suffix
#         sentences = self.db_handler.load_dataframe(dataset_name, split, f"sentences{suffix}")
#
#         if sentences.empty:
#             logger.warning(f"No sentences found for {dataset_name} {split}")
#             return pd.DataFrame()
#
#         # Process sentences with Stanza
#         parse_trees = []
#
#         logger.info(f"Processing {len(sentences)} sentences with Stanza")
#         for idx, row in tqdm(sentences.iterrows(), total=len(sentences)):
#             sentence_id = row["id"]
#             text = row["text"]
#
#             try:
#                 doc = self.nlp(text)
#
#                 # Extract constituency tree
#                 constituency_tree = str(doc.sentences[0].constituency)
#
#                 # Extract dependency tree as a serialized format
#                 dep_edges = []
#                 for word in doc.sentences[0].words:
#                     dep_edges.append({
#                         "id": word.id,
#                         "text": word.text,
#                         "lemma": word.lemma,
#                         "pos": word.pos,
#                         "head": word.head,
#                         "deprel": word.deprel
#                     })
#
#                 dependency_tree = str(dep_edges)
#
#                 parse_trees.append({
#                     "sentence_id": sentence_id,
#                     "constituency_tree": constituency_tree,
#                     "dependency_tree": dependency_tree
#                 })
#
#             except Exception as e:
#                 logger.error(f"Error processing sentence {sentence_id}: {str(e)}")
#                 parse_trees.append({
#                     "sentence_id": sentence_id,
#                     "constituency_tree": "",
#                     "dependency_tree": ""
#                 })
#
#         # Convert to dataframe and store with appropriate suffix
#         parse_trees_df = pd.DataFrame(parse_trees)
#         self.db_handler.store_dataframe(parse_trees_df, dataset_name, split, f"parse_trees{suffix}")
#
#         return parse_trees_df
#
#     # def preprocess_dataset(
#     #         self,
#     #         dataset_name: str,
#     #         split: str,
#     #         force_reprocess: bool = False
#     # ) -> pd.DataFrame:
#     #     """Preprocess a dataset with syntactic parsing."""
#     #     # Check if already processed
#     #     if not force_reprocess and self.db_handler.check_exists(dataset_name, split, "parse_trees"):
#     #         logger.info(f"Loading preprocessed {dataset_name} {split} from database")
#     #         return self.db_handler.load_dataframe(dataset_name, split, "parse_trees")
#     #
#     #     # Load sentences
#     #     sentences = self.db_handler.load_dataframe(dataset_name, split, "sentences")
#     #
#     #     if sentences.empty:
#     #         logger.warning(f"No sentences found for {dataset_name} {split}")
#     #         return pd.DataFrame()
#     #
#     #     # Process sentences with Stanza
#     #     parse_trees = []
#     #
#     #     logger.info(f"Processing {len(sentences)} sentences with Stanza")
#     #     for idx, row in tqdm(sentences.iterrows(), total=len(sentences)):
#     #         sentence_id = row["id"]
#     #         text = row["text"]
#     #
#     #         try:
#     #             doc = self.nlp(text)
#     #
#     #             # Extract constituency tree
#     #             constituency_tree = str(doc.sentences[0].constituency)
#     #
#     #             # Extract dependency tree as a serialized format
#     #             dep_edges = []
#     #             for word in doc.sentences[0].words:
#     #                 dep_edges.append({
#     #                     "id": word.id,
#     #                     "text": word.text,
#     #                     "lemma": word.lemma,
#     #                     "pos": word.pos,
#     #                     "head": word.head,
#     #                     "deprel": word.deprel
#     #                 })
#     #
#     #             dependency_tree = str(dep_edges)
#     #
#     #             parse_trees.append({
#     #                 "sentence_id": sentence_id,
#     #                 "constituency_tree": constituency_tree,
#     #                 "dependency_tree": dependency_tree
#     #             })
#     #
#     #         except Exception as e:
#     #             logger.error(f"Error processing sentence {sentence_id}: {str(e)}")
#     #             parse_trees.append({
#     #                 "sentence_id": sentence_id,
#     #                 "constituency_tree": "",
#     #                 "dependency_tree": ""
#     #             })
#     #
#     #     # Convert to dataframe and store
#     #     parse_trees_df = pd.DataFrame(parse_trees)
#     #     self.db_handler.store_dataframe(parse_trees_df, dataset_name, split, "parse_trees")
#     #
#     #     return parse_trees_df
#
#     def extract_parse_features(self, parse_tree_str: str, tree_type: str = "constituency") -> Dict:
#         """Extract features from a parse tree."""
#         features = {}
#
#         if not parse_tree_str:
#             return features
#
#         if tree_type == "constituency":
#             # Extract constituency features
#             features["tree_depth"] = parse_tree_str.count("(")
#
#             # Count phrase types
#             phrase_types = ["NP", "VP", "PP", "ADJP", "ADVP", "S", "SBAR"]
#             for phrase in phrase_types:
#                 features[f"count_{phrase}"] = parse_tree_str.count(f"({phrase} ")
#
#         elif tree_type == "dependency":
#             # Parse the string representation back to a list of dicts
#             try:
#                 dep_edges = eval(parse_tree_str)
#
#                 # Extract dependency features
#                 pos_counts = {}
#                 deprel_counts = {}
#
#                 for edge in dep_edges:
#                     pos = edge.get("pos", "UNKNOWN")
#                     deprel = edge.get("deprel", "UNKNOWN")
#
#                     pos_counts[pos] = pos_counts.get(pos, 0) + 1
#                     deprel_counts[deprel] = deprel_counts.get(deprel, 0) + 1
#
#                 # Add counts to features
#                 for pos, count in pos_counts.items():
#                     features[f"pos_{pos}"] = count
#
#                 for deprel, count in deprel_counts.items():
#                     features[f"deprel_{deprel}"] = count
#
#                 # Tree depth approximation
#                 features["tree_depth"] = max(edge.get("head", 0) for edge in dep_edges)
#
#             except Exception as e:
#                 logger.error(f"Error parsing dependency tree: {str(e)}")
#
#         return features


import logging
from typing import List, Dict, Tuple, Optional

import pandas as pd
import numpy as np
import stanza
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold

from config import STANZA_PROCESSORS, STANZA_LANG
from utils.database import DatabaseHandler

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Preprocess text for NLI tasks using Stanza for syntactic parsing."""

    def __init__(self, db_handler: DatabaseHandler):
        """Initialize preprocessor with Stanza pipeline."""
        self.db_handler = db_handler

        # Initialize Stanza pipeline
        try:
            self.nlp = stanza.Pipeline(
                lang=STANZA_LANG,
                processors=STANZA_PROCESSORS,
                use_gpu=True,
                verbose=False
            )
            logger.info(f"Stanza pipeline initialized with processors: {STANZA_PROCESSORS}")
        except Exception as e:
            logger.error(f"Failed to initialize Stanza pipeline: {str(e)}")
            logger.info("Downloading Stanza models...")
            stanza.download(STANZA_LANG)
            self.nlp = stanza.Pipeline(
                lang=STANZA_LANG,
                processors=STANZA_PROCESSORS,
                use_gpu=True,
                verbose=False
            )

    def preprocess_dataset(
            self,
            dataset_name: str,
            split: str,
            sample_size: int = None,
            force_reprocess: bool = False
    ) -> pd.DataFrame:
        """Preprocess a dataset with syntactic parsing."""
        # Add sample suffix to cache keys if needed
        suffix = f"_sample{sample_size}" if sample_size else ""

        # Check if already processed with appropriate suffix
        if not force_reprocess and self.db_handler.check_exists(dataset_name, split, f"parse_trees{suffix}"):
            logger.info(f"Loading preprocessed {dataset_name} {split} from database")
            return self.db_handler.load_dataframe(dataset_name, split, f"parse_trees{suffix}")

        # Load sentences with sample suffix
        sentences = self.db_handler.load_dataframe(dataset_name, split, f"sentences{suffix}")

        if sentences.empty:
            logger.warning(f"No sentences found for {dataset_name} {split}")
            return pd.DataFrame()

        # Process sentences with Stanza
        parse_trees = []

        logger.info(f"Processing {len(sentences)} sentences with Stanza")
        for idx, row in tqdm(sentences.iterrows(), total=len(sentences)):
            sentence_id = row["id"]
            text = row["text"]

            try:
                doc = self.nlp(text)

                # Extract constituency tree
                constituency_tree = str(doc.sentences[0].constituency)

                # Extract dependency tree as a serialized format
                dep_edges = []
                for word in doc.sentences[0].words:
                    dep_edges.append({
                        "id": word.id,
                        "text": word.text,
                        "lemma": word.lemma,
                        "pos": word.pos,
                        "head": word.head,
                        "deprel": word.deprel
                    })

                dependency_tree = str(dep_edges)

                parse_trees.append({
                    "sentence_id": sentence_id,
                    "constituency_tree": constituency_tree,
                    "dependency_tree": dependency_tree
                })

            except Exception as e:
                logger.error(f"Error processing sentence {sentence_id}: {str(e)}")
                parse_trees.append({
                    "sentence_id": sentence_id,
                    "constituency_tree": "",
                    "dependency_tree": ""
                })

        # Convert to dataframe and store with appropriate suffix
        parse_trees_df = pd.DataFrame(parse_trees)
        self.db_handler.store_dataframe(parse_trees_df, dataset_name, split, f"parse_trees{suffix}")

        return parse_trees_df

    def extract_parse_features(self, parse_tree_str: str, tree_type: str = "constituency") -> Dict:
        """Extract features from a parse tree."""
        features = {}

        if not parse_tree_str:
            return features

        if tree_type == "constituency":
            # Extract constituency features
            features["tree_depth"] = parse_tree_str.count("(")

            # Count phrase types
            phrase_types = ["NP", "VP", "PP", "ADJP", "ADVP", "S", "SBAR"]
            for phrase in phrase_types:
                features[f"count_{phrase}"] = parse_tree_str.count(f"({phrase} ")

        elif tree_type == "dependency":
            # Parse the string representation back to a list of dicts
            try:
                dep_edges = eval(parse_tree_str)

                # Extract dependency features
                pos_counts = {}
                deprel_counts = {}

                for edge in dep_edges:
                    pos = edge.get("pos", "UNKNOWN")
                    deprel = edge.get("deprel", "UNKNOWN")

                    pos_counts[pos] = pos_counts.get(pos, 0) + 1
                    deprel_counts[deprel] = deprel_counts.get(deprel, 0) + 1

                # Add counts to features
                for pos, count in pos_counts.items():
                    features[f"pos_{pos}"] = count

                for deprel, count in deprel_counts.items():
                    features[f"deprel_{deprel}"] = count

                # Tree depth approximation
                features["tree_depth"] = max(edge.get("head", 0) for edge in dep_edges)

            except Exception as e:
                logger.error(f"Error parsing dependency tree: {str(e)}")

        return features

    def prepare_sentence_pairs(self, split_data: pd.DataFrame, dataset_name: str, split: str) -> Tuple[
        pd.DataFrame, pd.DataFrame]:
        """Prepare sentence pairs and individual sentences for a dataset split."""
        # Extract unique sentences
        sentences = []
        sentence_ids = {}

        # Process premises
        for idx, premise in enumerate(split_data["premise"].unique()):
            sentence_id = f"{dataset_name}_{split}_p_{idx}"
            sentence_ids[premise] = sentence_id
            sentences.append({"id": sentence_id, "text": premise})

        # Process hypotheses
        for idx, hypothesis in enumerate(split_data["hypothesis"].unique()):
            if hypothesis not in sentence_ids:
                sentence_id = f"{dataset_name}_{split}_h_{idx}"
                sentence_ids[hypothesis] = sentence_id
                sentences.append({"id": sentence_id, "text": hypothesis})

        # Create sentences dataframe
        sentences_df = pd.DataFrame(sentences)

        # Create pairs dataframe
        pairs = []
        for idx, row in split_data.iterrows():
            pair_id = row.get("id", f"{dataset_name}_{split}_pair_{idx}")
            premise_id = sentence_ids[row["premise"]]
            hypothesis_id = sentence_ids[row["hypothesis"]]

            pair = {
                "id": pair_id,
                "premise_id": premise_id,
                "hypothesis_id": hypothesis_id,
                "label": row["label"]
            }

            pairs.append(pair)

        pairs_df = pd.DataFrame(pairs)

        # Store in database
        self.db_handler.store_dataframe(pairs_df, dataset_name, split, "pairs")
        self.db_handler.store_dataframe(sentences_df, dataset_name, split, "sentences")

        return pairs_df, sentences_df

    def create_stratified_sample_and_splits(
            self,
            dataset_name: str,
            label_column: str = "gold_label",
            total_samples: int = 300,
            samples_per_class: int = 100,
            test_size: float = 0.2,
            n_folds: int = 5,
            random_state: int = 42,
            force_reprocess: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Create a stratified sample from the dataset and split it into train/test sets.
        Additionally, generate k-fold cross-validation indices for the training set.

        Args:
            dataset_name: Name of the dataset (e.g., 'snli')
            label_column: Name of the column containing class labels
            total_samples: Total number of samples to extract (default: 300)
            samples_per_class: Number of samples per class (default: 100)
            test_size: Proportion of the data to use for testing (default: 0.2)
            n_folds: Number of folds for cross-validation (default: 5)
            random_state: Random seed for reproducibility
            force_reprocess: Whether to force reprocessing even if data exists

        Returns:
            Dictionary containing the generated dataframes and fold indices
        """
        # Check if already processed
        if not force_reprocess and all([
            self.db_handler.check_exists(dataset_name, "stratified_sample", "data"),
            self.db_handler.check_exists(dataset_name, "train_split", "data"),
            self.db_handler.check_exists(dataset_name, "test_split", "data")
        ]):
            logger.info(f"Loading preprocessed stratified samples and splits for {dataset_name}")
            result = {
                "stratified_sample": self.db_handler.load_dataframe(dataset_name, "stratified_sample", "data"),
                "train_split": self.db_handler.load_dataframe(dataset_name, "train_split", "data"),
                "test_split": self.db_handler.load_dataframe(dataset_name, "test_split", "data"),
            }

            # Load CV fold indices
            cv_folds = []
            for i in range(n_folds):
                if self.db_handler.check_exists(dataset_name, f"cv_fold_{i}", "indices"):
                    fold_indices = self.db_handler.load_dataframe(dataset_name, f"cv_fold_{i}", "indices")
                    cv_folds.append((fold_indices["train_indices"].tolist(), fold_indices["val_indices"].tolist()))

            result["cv_folds"] = cv_folds
            return result

        # Load full dataset
        logger.info(f"Loading full dataset: {dataset_name}")
        full_dataset = self.db_handler.load_dataframe(dataset_name, "all", f"sample{total_samples}")

        if full_dataset.empty:
            logger.warning(f"No data found for {dataset_name}")
            return {}

        # Check if the dataset already has split designations
        has_split_column = any(col in full_dataset.columns for col in ["split", "data_split", "dataset_split"])

        # If there's no split column, we'll create our own splits
        if not has_split_column:
            # Create stratified sample with equal class distribution
            # Check if label column exists
            if label_column not in full_dataset.columns:
                logger.error(f"Label column '{label_column}' not found in dataset")
                return {}

            # Get unique labels
            unique_labels = full_dataset[label_column].unique()

            # Sample equally from each class
            stratified_sample_indices = []

            for label in unique_labels:
                # Get indices where label matches
                label_indices = full_dataset[full_dataset[label_column] == label].index.tolist()

                # Check if we have enough samples for this class
                if len(label_indices) < samples_per_class:
                    logger.warning(f"Only {len(label_indices)} samples available for class {label}. "
                                   f"Using all available samples.")
                    stratified_sample_indices.extend(label_indices)
                else:
                    # Randomly sample from this class
                    np.random.seed(random_state)
                    sampled_indices = np.random.choice(label_indices, size=samples_per_class, replace=False)
                    stratified_sample_indices.extend(sampled_indices)

            # Extract the stratified sample
            stratified_sample = full_dataset.loc[stratified_sample_indices].copy()

            # Create train-test split
            X = stratified_sample.drop(columns=[label_column])
            y = stratified_sample[label_column]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=random_state
            )

            # Recombine features and labels
            train_split = X_train.copy()
            train_split[label_column] = y_train.values

            test_split = X_test.copy()
            test_split[label_column] = y_test.values

            # Set up cross-validation folds
            cv_folds = []
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

            for train_idx, val_idx in skf.split(X_train, y_train):
                cv_folds.append((train_idx, val_idx))

                # Store fold indices
                fold_index = len(cv_folds) - 1
                fold_indices_df = pd.DataFrame({
                    "train_indices": [train_idx],
                    "val_indices": [val_idx]
                })
                self.db_handler.store_dataframe(fold_indices_df, dataset_name, f"cv_fold_{fold_index}", "indices")

            # Store dataframes
            self.db_handler.store_dataframe(stratified_sample, dataset_name, "stratified_sample", "data")
            self.db_handler.store_dataframe(train_split, dataset_name, "train_split", "data")
            self.db_handler.store_dataframe(test_split, dataset_name, "test_split", "data")

            return {
                "stratified_sample": stratified_sample,
                "train_split": train_split,
                "test_split": test_split,
                "cv_folds": cv_folds
            }
        else:
            # Use existing splits from the dataset
            logger.info(f"Using existing splits from the dataset")

            # Identify the split column
            split_column = next(col for col in full_dataset.columns if col in ["split", "data_split", "dataset_split"])

            # Create stratified sample
            stratified_sample_indices = []
            unique_labels = full_dataset[label_column].unique()

            for label in unique_labels:
                label_indices = full_dataset[full_dataset[label_column] == label].index.tolist()
                np.random.seed(random_state)
                sampled_indices = np.random.choice(label_indices, size=samples_per_class, replace=False)
                stratified_sample_indices.extend(sampled_indices)

            stratified_sample = full_dataset.loc[stratified_sample_indices].copy()

            # Use predefined splits
            train_split = stratified_sample[stratified_sample[split_column] == "train"].copy()
            test_split = stratified_sample[stratified_sample[split_column] == "test"].copy()

            # Handle CV folds using only the training data
            X_train = train_split.drop(columns=[label_column, split_column])
            y_train = train_split[label_column]

            cv_folds = []
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

            for train_idx, val_idx in skf.split(X_train, y_train):
                cv_folds.append((train_idx, val_idx))

                # Store fold indices
                fold_index = len(cv_folds) - 1
                fold_indices_df = pd.DataFrame({
                    "train_indices": [train_idx],
                    "val_indices": [val_idx]
                })
                self.db_handler.store_dataframe(fold_indices_df, dataset_name, f"cv_fold_{fold_index}", "indices")

            # Store dataframes
            self.db_handler.store_dataframe(stratified_sample, dataset_name, "stratified_sample", "data")
            self.db_handler.store_dataframe(train_split, dataset_name, "train_split", "data")
            self.db_handler.store_dataframe(test_split, dataset_name, "test_split", "data")

            return {
                "stratified_sample": stratified_sample,
                "train_split": train_split,
                "test_split": test_split,
                "cv_folds": cv_folds
            }
