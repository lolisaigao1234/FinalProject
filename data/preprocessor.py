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
        self.suffix = f"sample{sample_size}" if sample_size else ""
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

    def _process_split(self, dataset_name: str, split_name: str, split_data: pd.DataFrame,
                       sample_size: int, force_reprocess: bool) -> None:
        self.logger.info(f"Processing sentences for {split_name} split")
        pairs_df, sentences_df = self.prepare_sentence_pairs(split_data, dataset_name, split_name)

        self.logger.info(f"Processing {sample_size} sentences with Stanza for {split_name} split")
        parse_trees_df = self.preprocess_dataset(
            dataset_name=dataset_name,
            split=split_name,
            sample_size=sample_size,
            force_reprocess=force_reprocess
        )

        if parse_trees_df is not None and not parse_trees_df.empty:
            self.logger.info(f"Extracting features for {split_name} split")
            # Lazy import to break circular dependency
            from features.feature_extractor import FeatureExtractor
            if self._feature_extractor is None:
                self._feature_extractor = FeatureExtractor(self.db_handler, self)

            self._feature_extractor.extract_features(
                dataset_name=dataset_name,
                split=split_name,
                force_recompute=force_reprocess,
                sample_size=sample_size
            )


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
        self.db_handler.store_dataframe(parse_trees_df, dataset_name, split, f"parse_trees{suffix}")

    def prepare_sentence_pairs(self, split_data: pd.DataFrame, dataset_name: str, split: str) -> Tuple[
        pd.DataFrame, pd.DataFrame]:
        sentences, sentence_ids = self._extract_unique_sentences(split_data, dataset_name, split)
        sentences_df = pd.DataFrame(sentences)
        pairs_df = self._create_pairs_dataframe(split_data, sentence_ids, dataset_name, split)
        self._store_sentence_pairs(pairs_df, sentences_df, dataset_name, split)
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

    def create_train_test_split(self, dataset_name: str, label_column: str = "label", test_size: float = 0.2,
                                random_state: int = 42) -> Dict[str, pd.DataFrame]:
        full_dataset = self._load_full_dataset(dataset_name)
        if full_dataset.empty:
            return {}

        train_split, test_split = train_test_split(full_dataset, test_size=test_size,
                                                   stratify=full_dataset[label_column], random_state=random_state)

        self._store_splits(dataset_name, train_split, test_split)

        return {"train_split": train_split, "test_split": test_split}

    def _load_full_dataset(self, dataset_name: str) -> pd.DataFrame:
        self.logger.info(f"Loading full dataset: {dataset_name}")
        full_dataset = self.db_handler.load_dataframe(dataset_name, "all", self.suffix)
        if full_dataset.empty:
            self.logger.warning(f"No data found for {dataset_name}")
        return full_dataset

    def _store_splits(self, dataset_name: str, train_split: pd.DataFrame, test_split: pd.DataFrame):
        self.db_handler.store_dataframe(train_split, dataset_name, "train_split", self.suffix)
        self.db_handler.store_dataframe(test_split, dataset_name, "test_split", self.suffix)

    def preprocess_dataset_pipeline(self, dataset_name: str, sample_size: int, force_reprocess: bool) -> None:
        full_dataset = self.data_loader.load_dataset(dataset_name, sample_size=sample_size)
        self.logger.info(f"Loaded {len(full_dataset)} samples for {dataset_name}")

        splits = self.create_train_test_split(dataset_name)
        if not splits:
            self.logger.error("Train-test split creation failed.")
            return

        self.logger.info(
            f"Created train-test split: {len(splits['train_split'])} train, {len(splits['test_split'])} test samples")

        for split_name, split_data in splits.items():
            if split_data.empty:
                self.logger.warning(f"No data available for {split_name} split")
                continue

            self._process_split(dataset_name, split_name, split_data, sample_size, force_reprocess)

    @nlp.setter
    def nlp(self, value):
        self._nlp = value
