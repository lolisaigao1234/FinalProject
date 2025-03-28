# data/preprocessor.py
from typing import Dict, Tuple, List, Optional
import pandas as pd
import stanza
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from config import STANZA_PROCESSORS, STANZA_LANG
from utils.database import DatabaseHandler
from utils.common import NLPBaseComponent, PreprocessorInterface
from data import DatasetLoader


class TextPreprocessor(NLPBaseComponent, PreprocessorInterface):
    def __init__(self, db_handler: DatabaseHandler, sample_size: Optional[int] = None):
        super().__init__(db_handler)
        self._nlp = None  # Private backing field
        self.sample_size = sample_size
        self.data_loader = DatasetLoader(db_handler)
        self._initialize_stanza_pipeline()
        self.suffix = f"sample{sample_size}" if sample_size else "data"
        self._feature_extractor = None

    @property
    def nlp(self) -> stanza.Pipeline:
        """Implementation of interface property"""
        return self._nlp

    def _initialize_stanza_pipeline(self):
        try:
            self._nlp = stanza.Pipeline(  # Assign to private field
                lang=STANZA_LANG,
                processors=STANZA_PROCESSORS,
                use_gpu=True,
                verbose=False
            )
            self.logger.info(f"Stanza pipeline initialized with processors: {STANZA_PROCESSORS}")
        except Exception as e:
            # Rest of initialization code...
            self.logger.error(f"Failed to initialize Stanza pipeline: {str(e)}")
            self.logger.info("Downloading Stanza models...")
            stanza.download(STANZA_LANG)
            self.nlp = stanza.Pipeline(
                lang=STANZA_LANG,
                processors=STANZA_PROCESSORS,
                use_gpu=True,
                verbose=False
            )

    # ------------------------
    def _process_split(self, dataset_name: str, split_name: str, split_data: pd.DataFrame,
                       sample_size: int, force_reprocess: bool) -> None:
        """Process dataset split with sampling and feature extraction pipeline."""
        self.logger.info(f"Processing split {split_name}...")
        self._log_process_start(split_name)
        pairs_df, sentences_df = self.prepare_sentence_pairs(split_data, dataset_name, split_name)

        if not self._validate_dataframe(pairs_df, 'label', "pairs_df"):
            return

        self.logger.info(f"Sample size for {split_name}: {sample_size}")

        sample_size = self._adjust_sample_size(pairs_df, sample_size)
        sampled_pairs = pairs_df.sample(n=sample_size, random_state=42)

        self._store_split_data(dataset_name, {split_name: sampled_pairs})

        sampled_sentences = self._extract_split_sentences({split_name: sampled_pairs}, sentences_df)
        self._store_sentence_data(dataset_name, sampled_sentences)

        self._log_split_creation({split_name: sampled_pairs}, sampled_sentences)
        self._execute_processing_pipeline(dataset_name, sampled_sentences, force_reprocess, sample_size)

    # Helper functions ------------------------------------------------------------

    def _log_process_start(self, split_name: str) -> None:
        """Log initial processing messages."""
        self.logger.info(f"Processing sentences for {split_name} split")
        self.logger.info(f"Beginning sampling for {split_name} split")

    def _validate_dataframe(self, df: pd.DataFrame, required_col: str, df_name: str) -> bool:
        """Validate dataframe contains required column."""
        if required_col not in df.columns:
            self.logger.error(f"{required_col} column missing in {df_name}")
            return False
        return True

    def _adjust_sample_size(self, pairs_df: pd.DataFrame, requested_size: int) -> int:
        """Adjust sample size to available data."""
        if len(pairs_df) < requested_size:
            self.logger.warning(
                f"Requested sample {requested_size} > available {len(pairs_df)}. Using all data.")
            return len(pairs_df)
        return requested_size

    def _store_split_data(self, dataset_name: str, splits: dict) -> None:
        """Store split pairs in database."""
        for split_name, pairs in splits.items():
            self.db_handler.store_dataframe(
                pairs, dataset_name, split_name, f"pairs_{self.suffix}"
            )

    def _extract_split_sentences(self, splits: dict, sentences_df: pd.DataFrame) -> dict:
        """Extract sentences for each split from original dataframe."""
        return {
            split_name: sentences_df[sentences_df['id'].isin(
                set(pairs['premise_id'].tolist() + pairs['hypothesis_id'].tolist())
            )]
            for split_name, pairs in splits.items()
        }

    def _store_sentence_data(self, dataset_name: str, sentence_splits: dict) -> None:
        """Store sentence data in database."""
        for split_name, sentences in sentence_splits.items():
            self.db_handler.store_dataframe(
                sentences, dataset_name, split_name, f"sentences_{self.suffix}"
            )

    def _log_split_creation(self, splits: dict, sentence_splits: dict) -> None:
        """Log statistics about created splits."""
        self.logger.info("Finished sampling:")
        for split_name in splits:
            self.logger.info(
                f"{split_name}: {len(splits[split_name])} pairs, "
                f"{len(sentence_splits[split_name])} sentences"
            )

    def _execute_processing_pipeline(self, dataset_name: str, sentence_splits: dict,
                                     force_reprocess: bool, sample_size: int) -> None:
        """Execute Stanza processing and feature extraction for all splits."""
        for split_name, sentences in sentence_splits.items():
            self.logger.info(f"Processing {len(sentences)} sentences with Stanza for {split_name}")
            parse_trees = self.preprocess_dataset(
                dataset_name=dataset_name,
                split=split_name,
                sample_size=self.sample_size,  # Use class-level sample size
                force_reprocess=force_reprocess
            )

            if parse_trees is not None and not parse_trees.empty:
                self._initialize_feature_extractor()
                self._feature_extractor.extract_features(
                    dataset_name=dataset_name,
                    split=split_name,
                    force_recompute=force_reprocess,
                    sample_size=sample_size  # Use method parameter
                )

    def _initialize_feature_extractor(self) -> None:
        """Lazy initialization of feature extractor."""
        if not hasattr(self, '_feature_extractor') or self._feature_extractor is None:
            from features.feature_extractor import FeatureExtractor
            self._feature_extractor = FeatureExtractor(self.db_handler, self)

    # ------------------------

    # def _process_split(self, dataset_name: str, split_name: str, split_data: pd.DataFrame,
    #                    sample_size: int, force_reprocess: bool) -> None:
    #     self.logger.info(f"Processing sentences for {split_name} split")
    #     pairs_df, sentences_df = self.prepare_sentence_pairs(split_data, dataset_name, split_name)
    #     self.logger.info(f"Finished with the creation of sentences, pairs for {split_name} split")
    #
    #     self.logger.info(f"Beginning sub sampling for {split_name} split")
    #     self.logger.info(f"Sampling size {sample_size}")
    #
    #     # Check if we have enough data for the requested sample size
    #     if len(pairs_df) < sample_size:
    #         self.logger.warning(
    #             f"Requested sample size {sample_size} is larger than available data {len(pairs_df)}. Using all available data.")
    #         sample_size = len(pairs_df)
    #
    #     # Calculate sizes for each split (80/10/10 split)
    #     train_size = int(sample_size * 0.8)
    #     val_size = int(sample_size * 0.1)
    #     test_size = sample_size - train_size - val_size
    #
    #     # Ensure pairs_df has the label column
    #     if 'label' not in pairs_df.columns:
    #         self.logger.error("Label column missing in pairs_df")
    #         return
    #
    #     # Step 1: Use stratified sampling to create train and temporary sets (val+test combined)
    #     train_pairs, temp_pairs = train_test_split(
    #         pairs_df,
    #         train_size=train_size,
    #         test_size=val_size + test_size,
    #         stratify=pairs_df['label'],  # Ensures each label (0,1,2) is proportionally represented
    #         random_state=42
    #     )
    #
    #     # Step 2: Split temporary set into validation and test sets
    #     val_pairs, test_pairs = train_test_split(
    #         temp_pairs,
    #         train_size=val_size / (val_size + test_size),
    #         stratify=temp_pairs['label'],  # Maintains label distribution in val and test
    #         random_state=42
    #     )
    #
    #     # Store the sampled data with appropriate suffixes
    #     self.db_handler.store_dataframe(train_pairs, dataset_name, "train", f"pairs_{self.suffix}")
    #     self.db_handler.store_dataframe(val_pairs, dataset_name, "validation", f"pairs_{self.suffix}")
    #     self.db_handler.store_dataframe(test_pairs, dataset_name, "test", f"pairs_{self.suffix}")
    #
    #     # Extract sentence IDs from each split and get corresponding sentences
    #     train_sent_ids = set(train_pairs['premise_id'].tolist() + train_pairs['hypothesis_id'].tolist())
    #     val_sent_ids = set(val_pairs['premise_id'].tolist() + val_pairs['hypothesis_id'].tolist())
    #     test_sent_ids = set(test_pairs['premise_id'].tolist() + test_pairs['hypothesis_id'].tolist())
    #
    #     train_sentences = sentences_df[sentences_df['id'].isin(train_sent_ids)]
    #     val_sentences = sentences_df[sentences_df['id'].isin(val_sent_ids)]
    #     test_sentences = sentences_df[sentences_df['id'].isin(test_sent_ids)]
    #
    #     # Store sentences dataframes
    #     self.db_handler.store_dataframe(train_sentences, dataset_name, "train", f"sentences_{self.suffix}")
    #     self.db_handler.store_dataframe(val_sentences, dataset_name, "validation", f"sentences_{self.suffix}")
    #     self.db_handler.store_dataframe(test_sentences, dataset_name, "test", f"sentences_{self.suffix}")
    #
    #     self.logger.info(f"Finished sub sampling for {split_name} split")
    #     self.logger.info(f"Created train split with {len(train_pairs)} pairs and {len(train_sentences)} sentences")
    #     self.logger.info(f"Created validation split with {len(val_pairs)} pairs and {len(val_sentences)} sentences")
    #     self.logger.info(f"Created test split with {len(test_pairs)} pairs and {len(test_sentences)} sentences")
    #
    #     # Process each split separately
    #     for current_split, current_sentences in [
    #         ("train", train_sentences),
    #         ("validation", val_sentences),
    #         ("test", test_sentences)
    #     ]:
    #         self.logger.info(f"Processing {len(current_sentences)} sentences with Stanza for {current_split} split")
    #         parse_trees_df = self.preprocess_dataset(
    #             dataset_name=dataset_name,
    #             split=current_split,
    #             sample_size=self.sample_size,
    #             force_reprocess=force_reprocess
    #         )
    #
    #         if parse_trees_df is not None and not parse_trees_df.empty:
    #             self.logger.info(f"Extracting features for {current_split} split")
    #             # Lazy import to break circular dependency
    #             from features.feature_extractor import FeatureExtractor
    #             if not hasattr(self, '_feature_extractor') or self._feature_extractor is None:
    #                 self._feature_extractor = FeatureExtractor(self.db_handler, self)
    #
    #             self._feature_extractor.extract_features(
    #                 dataset_name=dataset_name,
    #                 split=current_split,
    #                 force_recompute=force_reprocess,
    #                 sample_size=sample_size
    #             )

    def preprocess_dataset(self, dataset_name: str, split: str, sample_size: int = None,
                           force_reprocess: bool = False) -> pd.DataFrame:
        self.logger.info(f"Preprocessing {dataset_name} {split} with syntactic parsing. Sample size: {sample_size}")

        if not force_reprocess and self._check_processed_data_exists(dataset_name, split, self.suffix):
            return self._load_processed_data(dataset_name, split, self.suffix)

        sentences = self._load_sentences(dataset_name, split, self.suffix)
        if sentences.empty:
            return pd.DataFrame()

        parse_trees = self._process_sentences_with_stanza(sentences)
        parse_trees_df = pd.DataFrame(parse_trees)
        self._store_processed_data(parse_trees_df, dataset_name, split, self.suffix)
        return parse_trees_df

    def _check_processed_data_exists(self, dataset_name: str, split: str, suffix: str) -> bool:
        return self.db_handler.check_exists(dataset_name, split, f"parse_trees_{suffix}")

    def _load_processed_data(self, dataset_name: str, split: str, suffix: str) -> pd.DataFrame:
        self.logger.info(f"Loading preprocessed {dataset_name} {split} from database")
        return self.db_handler.load_dataframe(dataset_name, split, f"parse_trees_{suffix}")

    def _load_sentences(self, dataset_name: str, split: str, suffix: str) -> pd.DataFrame:
        sentences = self.db_handler.load_dataframe(dataset_name, split, f"sentences_{suffix}")
        if sentences.empty:
            self.logger.warning(f"No sentences found for {dataset_name} {split}")
        return sentences

    def _process_sentences_with_stanza(self, sentences: pd.DataFrame) -> list:
        parse_trees = []
        self.logger.info(f"Processing {len(sentences)} sentences with Stanza")
        for idx, row in tqdm(sentences.iterrows(), total=len(sentences)):
            sentence_id, text = row["id"], row["text"]
            try:
                doc = self.nlp(text)
                constituency_tree = str(doc.sentences[0].constituency)
                dependency_tree = self._extract_dependency_tree(doc.sentences[0].words)
                parse_trees.append({"sentence_id": sentence_id, "constituency_tree": constituency_tree,
                                    "dependency_tree": dependency_tree})
            except Exception as e:
                self.logger.error(f"Error processing sentence {sentence_id}: {str(e)}")
                parse_trees.append({"sentence_id": sentence_id, "constituency_tree": "", "dependency_tree": ""})
        return parse_trees

    def _extract_dependency_tree(self, words):
        dep_edges = [{"id": word.id, "text": word.text, "lemma": word.lemma, "pos": word.pos, "head": word.head,
                      "deprel": word.deprel} for word in words]
        return str(dep_edges)

    def _store_processed_data(self, parse_trees_df: pd.DataFrame, dataset_name: str, split: str, suffix: str):
        self.db_handler.store_dataframe(parse_trees_df, dataset_name, split, f"parse_trees_{suffix}")

    def prepare_sentence_pairs(self, split_data: pd.DataFrame, dataset_name: str, split: str) \
            -> Tuple[pd.DataFrame, pd.DataFrame]:
        sentences, sentence_ids = self._extract_unique_sentences(split_data, dataset_name, split)
        sentences_df = pd.DataFrame(sentences)
        pairs_df = self._create_pairs_dataframe(split_data, sentence_ids, dataset_name, split)
        self._store_sentence_pairs(pairs_df, sentences_df, dataset_name, split) if not self.suffix else None
        return pairs_df, sentences_df

    def _extract_unique_sentences(self, split_data: pd.DataFrame, dataset_name: str, split: str) -> Tuple[list, dict]:
        sentences, sentence_ids = [], {}
        for idx, premise in enumerate(split_data["premise_text"].unique()):
            sentence_id = f"{dataset_name}_{split}_p_{idx}"
            sentence_ids[premise] = sentence_id
            sentences.append({"id": sentence_id, "text": premise})
        for idx, hypothesis in enumerate(split_data["hypothesis_text"].unique()):
            if hypothesis not in sentence_ids:
                sentence_id = f"{dataset_name}_{split}_h_{idx}"
                sentence_ids[hypothesis] = sentence_id
                sentences.append({"id": sentence_id, "text": hypothesis})
        return sentences, sentence_ids

    def _create_pairs_dataframe(self, split_data: pd.DataFrame, sentence_ids: dict, dataset_name: str,
                                split: str) -> pd.DataFrame:
        pairs = []
        for idx, row in split_data.iterrows():
            pair_id = row.get("id", f"{dataset_name}_{split}_pair_{idx}")
            premise_id = sentence_ids[row["premise_text"]]
            hypothesis_id = sentence_ids[row["hypothesis_text"]]
            pairs.append(
                {"id": pair_id, "premise_id": premise_id, "hypothesis_id": hypothesis_id, "label": row["label"]})
        return pd.DataFrame(pairs)

    def _store_sentence_pairs(self, pairs_df: pd.DataFrame, sentences_df: pd.DataFrame, dataset_name: str, split: str):
        self.db_handler.store_dataframe(pairs_df, dataset_name, split, f"pairs_{self.suffix}")
        self.db_handler.store_dataframe(sentences_df, dataset_name, split, f"sentences_{self.suffix}")
        self.logger.info(f"For dataset {dataset_name} and split {split}, created pairs_df and sentences_df")


    def _load_full_dataset(self, dataset_name: str, split_name: str) -> pd.DataFrame:
        self.logger.info(f"Loading full dataset: {dataset_name}")
        full_dataset = self.db_handler.load_dataframe(dataset_name, split_name, self.suffix)
        if full_dataset.empty:
            self.logger.warning(f"No data found for {dataset_name}")
        return full_dataset

    def _store_splits(self, dataset_name: str, train_split: pd.DataFrame, test_split: pd.DataFrame):
        self.db_handler.store_dataframe(train_split, dataset_name, "train_split", self.suffix)
        self.db_handler.store_dataframe(test_split, dataset_name, "test_split", self.suffix)

    def preprocess_dataset_pipeline(self, dataset_name: str,
                                    total_sample_size: int,
                                    train_ratio: float,
                                    force_reprocess: bool) -> None:
        self.logger.info(f"Starting preprocessing pipeline for {dataset_name}")

        # Calculate split sizes
        train_size = int(total_sample_size * train_ratio)
        val_size = int(total_sample_size * (1 - train_ratio) / 2)
        test_size = total_sample_size - train_size - val_size

        split_sizes = {
            "train": train_size,
            "validation": val_size,
            "test": test_size
        }

        for split_name, sample_size in split_sizes.items():
            self.logger.info(f"Processing {split_name} split")

            # Load the full split data
            full_split_data = self.data_loader.load_dataset(dataset_name, split=split_name)

            if full_split_data.empty:
                self.logger.warning(f"No data available for {split_name} split")
                continue

            # Sample the data
            sampled_data = full_split_data.sample(n=sample_size, random_state=42)

            # Create the new suffix
            new_suffix = f"sample{sample_size}"

            # Store the sampled data
            self.db_handler.store_dataframe(sampled_data, dataset_name, split_name, new_suffix)

            # Process the sampled split
            self._process_split(dataset_name, split_name, sampled_data, sample_size, force_reprocess)

        self.logger.info(f"Completed preprocessing pipeline for {dataset_name}")

    @nlp.setter
    def nlp(self, value):
        self._nlp = value
