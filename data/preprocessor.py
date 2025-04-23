# # data/preprocessor.py
# from typing import Tuple, Optional
# import pandas as pd
# import stanza
# from tqdm import tqdm
# import os
#
# from config import STANZA_PROCESSORS, STANZA_LANG
# from utils.database import DatabaseHandler
# from utils.common import NLPBaseComponent, PreprocessorInterface
# from data import DatasetLoader
#
#
# def _extract_dependency_tree(words):
#     dep_edges = [{"id": word.id, "text": word.text, "lemma": word.lemma, "pos": word.pos, "head": word.head,
#                   "deprel": word.deprel} for word in words]
#     return str(dep_edges)
#
#
# def _extract_unique_sentences(split_data: pd.DataFrame, dataset_name: str, split: str) -> Tuple[list, dict]:
#     sentences, sentence_ids = [], {}
#     for idx, premise in enumerate(split_data["premise_text"].unique()):
#         sentence_id = f"{dataset_name}_{split}_p_{idx}"
#         sentence_ids[premise] = sentence_id
#         sentences.append({"id": sentence_id, "text": premise})
#     for idx, hypothesis in enumerate(split_data["hypothesis_text"].unique()):
#         if hypothesis not in sentence_ids:
#             sentence_id = f"{dataset_name}_{split}_h_{idx}"
#             sentence_ids[hypothesis] = sentence_id
#             sentences.append({"id": sentence_id, "text": hypothesis})
#     return sentences, sentence_ids
#
#
# def _create_pairs_dataframe(split_data: pd.DataFrame, sentence_ids: dict, dataset_name: str,
#                             split: str) -> pd.DataFrame:
#     pairs = []
#     for idx, row in split_data.iterrows():
#         pair_id = row.get("id", f"{dataset_name}_{split}_pair_{idx}")
#         premise_id = sentence_ids[row["premise_text"]]
#         hypothesis_id = sentence_ids[row["hypothesis_text"]]
#         pairs.append(
#             {"id": pair_id, "premise_id": premise_id, "hypothesis_id": hypothesis_id, "label": row["label"]})
#     return pd.DataFrame(pairs)
#
#
# class TextPreprocessor(NLPBaseComponent, PreprocessorInterface):
#     def __init__(self, db_handler: DatabaseHandler, sample_size: Optional[int] = None):
#         super().__init__(db_handler)
#         self._nlp = None  # Private backing field
#         self.sample_size = sample_size
#         self.data_loader = DatasetLoader(db_handler)
#         self._initialize_stanza_pipeline()
#         self.suffix = f"sample{sample_size}" if sample_size else "data"
#         self._feature_extractor = None
#         self.split_to_size = {}
#
#     @property
#     def nlp(self) -> stanza.Pipeline:
#         """Implementation of interface property"""
#         return self._nlp
#
#     def _initialize_stanza_pipeline(self):
#         num_threads = int(os.environ.get("OMP_NUM_THREADS", 64))
#         try:
#             # In data/preprocessor.py, inside _initialize_stanza_pipeline
#             self._nlp = stanza.Pipeline(
#                 lang=STANZA_LANG,
#                 processors=STANZA_PROCESSORS,
#                 use_gpu=True,  # <-- CHANGE THIS TO TRUE
#                 verbose=False,
#                 num_threads=num_threads  # Keep or adjust num_threads as needed
#             )
#             self.logger.info(f"Stanza pipeline initialized with processors: {STANZA_PROCESSORS}")
#         except Exception as e:
#             # Rest of initialization code...
#             self.logger.error(f"Failed to initialize Stanza pipeline: {str(e)}")
#             self.logger.info("Downloading Stanza models...")
#             stanza.download(STANZA_LANG)
#             self.nlp = stanza.Pipeline(
#                 lang=STANZA_LANG,
#                 processors=STANZA_PROCESSORS,
#                 use_gpu=True,
#                 verbose=False,
#                 num_threads=num_threads
#             )
#
#     # ------------------------
#
#     def _process_split(self, dataset_name: str, split_name: str, split_data: pd.DataFrame,
#                        sample_size: int, force_reprocess: bool) -> None:
#         """Process dataset split with stratified sampling and feature extraction pipeline."""
#         self.logger.info(f"Processing split {split_name}...")
#         self._log_process_start(split_name)
#
#         # Prepare sentence pairs and extract unique sentences
#         pairs_df, sentences_df = self.prepare_sentence_pairs(split_data, dataset_name, split_name)
#
#         if not self._validate_dataframe(pairs_df, 'label', "pairs_df"):
#             return
#
#         # Adjust sample size if necessary
#         sample_size = self._adjust_sample_size(pairs_df, sample_size)
#
#         # Perform stratified sampling
#         self.logger.info(f"Performing stratified sampling for {split_name} split")
#         sampled_pairs = pairs_df.groupby('label', group_keys=False).apply(
#             lambda x: x.sample(n=min(len(x), sample_size // len(pairs_df['label'].unique())), random_state=42)
#         )
#
#         # Store sampled data
#         self.db_handler.store_dataframe(sampled_pairs, dataset_name, split_name, f"pairs_{self.suffix}")
#
#         # Extract corresponding sentences
#         sent_ids = set(sampled_pairs['premise_id'].tolist() + sampled_pairs['hypothesis_id'].tolist())
#         sampled_sentences = sentences_df[sentences_df['id'].isin(sent_ids)]
#
#         # Store sampled sentences
#         self.db_handler.store_dataframe(sampled_sentences, dataset_name, split_name, f"sentences_{self.suffix}")
#
#         self.logger.info(
#             f"Finished processing {split_name} split with {len(sampled_pairs)} pairs and {len(sampled_sentences)} sentences")
#
#         # Execute processing pipeline (Stanza parsing and feature extraction)
#         self._execute_processing_pipeline(dataset_name, {split_name: sampled_sentences}, force_reprocess, sample_size)
#
#     # Helper functions ------------------------------------------------------------
#
#     def _log_process_start(self, split_name: str) -> None:
#         """Log initial processing messages."""
#         self.logger.info(f"Processing sentences for {split_name} split")
#         self.logger.info(f"Beginning sampling for {split_name} split")
#
#     def _validate_dataframe(self, df: pd.DataFrame, required_col: str, df_name: str) -> bool:
#         """Validate dataframe contains required column."""
#         if required_col not in df.columns:
#             self.logger.error(f"{required_col} column missing in {df_name}")
#             return False
#         return True
#
#     def _adjust_sample_size(self, pairs_df: pd.DataFrame, requested_size: int) -> int:
#         """Adjust sample size to available data."""
#         if len(pairs_df) < requested_size:
#             self.logger.warning(
#                 f"Requested sample size {requested_size} > available data {len(pairs_df)}. Using all data.")
#             return len(pairs_df)
#         return requested_size
#
#     def _execute_processing_pipeline(self, dataset_name: str, sentence_splits: dict,
#                                      force_reprocess: bool, sample_size: int) -> None:
#         """Execute Stanza processing and feature extraction for all splits."""
#         for split_name, sentences in sentence_splits.items():
#             self.logger.info(f"Processing {len(sentences)} sentences with Stanza for {split_name}")
#             parse_trees = self.preprocess_dataset(
#                 dataset_name=dataset_name,
#                 split=split_name,
#                 sample_size=self.sample_size,
#                 force_reprocess=force_reprocess
#             )
#
#             if parse_trees is not None and not parse_trees.empty:
#                 self._initialize_feature_extractor()
#                 self._feature_extractor.extract_features(
#                     dataset_name=dataset_name,
#                     split=split_name,
#                     force_recompute=True,
#                     sample_size=sample_size
#                 )
#
#     def _initialize_feature_extractor(self) -> None:
#         """Lazy initialization of feature extractor."""
#         if not hasattr(self, '_feature_extractor') or self._feature_extractor is None:
#             from features.feature_extractor import FeatureExtractor
#             self._feature_extractor = FeatureExtractor(self.db_handler, self)
#
#     def preprocess_dataset(self, dataset_name: str, split: str, sample_size: int = None,
#                            force_reprocess: bool = False) -> pd.DataFrame:
#         self.logger.info(f"Preprocessing {dataset_name} {split} with syntactic parsing. Sample size: {sample_size}")
#
#         if not force_reprocess and self._check_processed_data_exists(dataset_name, split, self.suffix):
#             return self._load_processed_data(dataset_name, split, self.suffix)
#
#         sentences = self._load_sentences(dataset_name, split, self.suffix)
#         if sentences.empty:
#             return pd.DataFrame()
#
#         parse_trees = self._process_sentences_with_stanza(sentences)
#         parse_trees_df = pd.DataFrame(parse_trees)
#         self._store_processed_data(parse_trees_df, dataset_name, split, self.suffix)
#         return parse_trees_df
#
#     def _check_processed_data_exists(self, dataset_name: str, split: str, suffix: str) -> bool:
#         return self.db_handler.check_exists(dataset_name, split, f"parse_trees_{suffix}")
#
#     def _load_processed_data(self, dataset_name: str, split: str, suffix: str) -> pd.DataFrame:
#         self.logger.info(f"Loading preprocessed {dataset_name} {split} from database")
#         return self.db_handler.load_dataframe(dataset_name, split, f"parse_trees_{suffix}")
#
#     def _load_sentences(self, dataset_name: str, split: str, suffix: str) -> pd.DataFrame:
#         sentences = self.db_handler.load_dataframe(dataset_name, split, f"sentences_{suffix}")
#         if sentences.empty:
#             self.logger.warning(f"No sentences found for {dataset_name} {split}")
#         return sentences
#
#     def _process_sentences_with_stanza(self, sentences: pd.DataFrame) -> list:
#         parse_trees = []
#         self.logger.info(f"Processing {len(sentences)} sentences with Stanza")
#         for idx, row in tqdm(sentences.iterrows(), total=len(sentences)):
#             sentence_id, text = row["id"], row["text"]
#             try:
#                 doc = self.nlp(text)
#                 constituency_tree = str(doc.sentences[0].constituency)
#                 dependency_tree = _extract_dependency_tree(doc.sentences[0].words)
#                 parse_trees.append({"sentence_id": sentence_id, "constituency_tree": constituency_tree,
#                                     "dependency_tree": dependency_tree})
#             except Exception as e:
#                 self.logger.error(f"Error processing sentence {sentence_id}: {str(e)}")
#                 parse_trees.append({"sentence_id": sentence_id, "constituency_tree": "", "dependency_tree": ""})
#         return parse_trees
#
#     def _store_processed_data(self, parse_trees_df: pd.DataFrame, dataset_name: str, split: str, suffix: str):
#         self.db_handler.store_dataframe(parse_trees_df, dataset_name, split, f"parse_trees_{suffix}")
#
#     def prepare_sentence_pairs(self, split_data: pd.DataFrame, dataset_name: str, split: str) \
#             -> Tuple[pd.DataFrame, pd.DataFrame]:
#         sentences, sentence_ids = _extract_unique_sentences(split_data, dataset_name, split)
#         sentences_df = pd.DataFrame(sentences)
#         pairs_df = _create_pairs_dataframe(split_data, sentence_ids, dataset_name, split)
#         self._store_sentence_pairs(pairs_df, sentences_df, dataset_name, split) if not self.suffix else None
#         return pairs_df, sentences_df
#
#     def _store_sentence_pairs(self, pairs_df: pd.DataFrame, sentences_df: pd.DataFrame, dataset_name: str, split: str):
#         self.db_handler.store_dataframe(pairs_df, dataset_name, split, f"pairs_{self.suffix}")
#         self.db_handler.store_dataframe(sentences_df, dataset_name, split, f"sentences_{self.suffix}")
#         self.logger.info(f"For dataset {dataset_name} and split {split}, created pairs_df and sentences_df")
#
#     def _load_full_dataset(self, dataset_name: str, split_name: str) -> pd.DataFrame:
#         self.logger.info(f"Loading full dataset: {dataset_name}")
#         full_dataset = self.db_handler.load_dataframe(dataset_name, split_name, self.suffix)
#         if full_dataset.empty:
#             self.logger.warning(f"No data found for {dataset_name}")
#         return full_dataset
#
#     def _store_splits(self, dataset_name: str, train_split: pd.DataFrame, test_split: pd.DataFrame):
#         self.db_handler.store_dataframe(train_split, dataset_name, "train_split", self.suffix)
#         self.db_handler.store_dataframe(test_split, dataset_name, "test_split", self.suffix)
#
#     def preprocess_dataset_pipeline(self, dataset_name: str,
#                                     total_sample_size: int,
#                                     force_reprocess: bool) -> None:
#         # global split_to_size
#         self.logger.info(f"Starting preprocessing pipeline for {dataset_name}")
#
#         split_names = [
#             "train",
#             "validation",
#             "test"
#         ]
#
#         # Calculate split sizes if total_sample_size is provided
#         if total_sample_size is not None:
#             train_ratio = 0.8
#             train_size = int(total_sample_size * train_ratio)
#             val_size = int(total_sample_size * (1 - train_ratio) / 2)
#             test_size = total_sample_size - train_size - val_size
#
#             # Verify the total adds up correctly
#             assert train_size + val_size + test_size == total_sample_size
#
#             # Map splits to their sizes
#             self.split_to_size = {
#                 "train": train_size,
#                 "validation": val_size,
#                 "test": test_size
#             }
#
#         for split_name in split_names:
#             self.logger.info(f"Processing {split_name} split")
#
#             # Load the full split data
#             full_split_data = self.data_loader.load_dataset(dataset_name, split=split_name)
#
#             if full_split_data.empty:
#                 self.logger.warning(f"No data available for {split_name} split")
#                 continue
#
#             # Determine whether to use full data or a sample
#             if total_sample_size is None:
#                 # Use the full dataset
#                 sample_size = len(full_split_data)
#                 sampled_data = full_split_data
#                 new_suffix = "full"
#             else:
#                 # Get the sample size for this split
#                 sample_size = self.split_to_size[split_name]
#
#                 # Make sure we don't try to sample more than available
#                 if sample_size > len(full_split_data):
#                     self.logger.warning(
#                         f"Requested sample size {sample_size} for {split_name} split exceeds available data ({len(full_split_data)}). Using all available data.")
#                     sample_size = len(full_split_data)
#                     sampled_data = full_split_data
#                 else:
#                     # Sample the data
#                     sampled_data = full_split_data.sample(n=sample_size, random_state=42)
#
#                 new_suffix = f"sample{sample_size}"
#
#             self.logger.info(f"Print out new suffix: {new_suffix}")
#
#             # Store the sampled data
#             self.db_handler.store_dataframe(sampled_data, dataset_name, split_name, new_suffix)
#
#             # Process the sampled split
#             self._process_split(dataset_name, split_name, sampled_data, sample_size, force_reprocess)
#
#         self.logger.info(f"Completed preprocessing pipeline for {dataset_name}")
#
#     @nlp.setter
#     def nlp(self, value):
#         self._nlp = value
from abc import ABC
# data/preprocessor.py
from typing import Tuple, Optional
import pandas as pd
import stanza
import torch
from tqdm import tqdm
import os

# Make sure logging is configured if not done elsewhere
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# --- End logging setup ---

from config import STANZA_PROCESSORS, STANZA_LANG
from utils.database import DatabaseHandler
from utils.common import NLPBaseComponent, PreprocessorInterface # Assuming common has these
# Adjust import if DatasetLoader is in a different module
from data.data_loader import DatasetLoader


def _extract_dependency_tree(words):
    dep_edges = [{"id": word.id, "text": word.text, "lemma": word.lemma, "pos": word.pos, "head": word.head,
                  "deprel": word.deprel} for word in words]
    # Ensure it's stored as a string representation of the list
    return str(dep_edges)


def _extract_unique_sentences(split_data: pd.DataFrame, dataset_name: str, split: str) -> Tuple[list, dict]:
    sentences, sentence_ids = [], {}
    # Ensure text columns exist and handle potential NaNs
    premise_texts = split_data["premise_text"].dropna().unique()
    hypothesis_texts = split_data["hypothesis_text"].dropna().unique()

    for idx, premise in enumerate(premise_texts):
        sentence_id = f"{dataset_name}_{split}_p_{idx}"
        sentence_ids[premise] = sentence_id
        sentences.append({"id": sentence_id, "text": premise})
    for idx, hypothesis in enumerate(hypothesis_texts):
        if hypothesis not in sentence_ids:
            # Ensure hypothesis is not empty or just whitespace
            if isinstance(hypothesis, str) and hypothesis.strip():
                sentence_id = f"{dataset_name}_{split}_h_{idx}" # Changed index source
                sentence_ids[hypothesis] = sentence_id
                sentences.append({"id": sentence_id, "text": hypothesis})
    return sentences, sentence_ids


def _create_pairs_dataframe(split_data: pd.DataFrame, sentence_ids: dict, dataset_name: str,
                            split: str) -> pd.DataFrame:
    pairs = []
    for idx, row in split_data.iterrows():
        # Handle potential missing text or labels gracefully
        premise_text = row.get("premise_text")
        hypothesis_text = row.get("hypothesis_text")
        label = row.get("label", -1) # Default to -1 if missing

        if pd.isna(premise_text) or pd.isna(hypothesis_text) or label == -1:
             logger.warning(f"Skipping row {idx} due to missing data: P='{premise_text}', H='{hypothesis_text}', L='{label}'")
             continue

        pair_id = row.get("id", f"{dataset_name}_{split}_pair_{idx}") # Use existing ID if available
        premise_id = sentence_ids.get(premise_text) # Use .get for safety
        hypothesis_id = sentence_ids.get(hypothesis_text)

        # Ensure both sentences were found in the unique sentence extraction
        if premise_id is not None and hypothesis_id is not None:
            pairs.append(
                {"id": pair_id, "premise_id": premise_id, "hypothesis_id": hypothesis_id, "label": label})
        else:
             logger.warning(f"Could not find sentence IDs for pair {pair_id}. Premise found: {premise_id is not None}, Hypothesis found: {hypothesis_id is not None}")

    if not pairs:
        logger.warning(f"No valid pairs created for {dataset_name} {split}.")
        return pd.DataFrame(columns=["id", "premise_id", "hypothesis_id", "label"]) # Return empty df with columns
    return pd.DataFrame(pairs)


# Assuming NLPBaseComponent and PreprocessorInterface are defined elsewhere
class TextPreprocessor(NLPBaseComponent, PreprocessorInterface, ABC):
    def __init__(self, db_handler: DatabaseHandler, sample_size: Optional[int] = None):
        super().__init__(db_handler) # Pass db_handler to parent if needed
        self.db_handler = db_handler # Store db_handler explicitly
        self._nlp = None  # Private backing field for Stanza pipeline
        self.sample_size = sample_size
        # Correctly initialize DatasetLoader - pass db_handler if it needs it
        self.data_loader = DatasetLoader(db_handler)
        self._initialize_stanza_pipeline()
        # Suffix will be determined dynamically based on whether sampling occurred
        self.suffix = None # Initialize suffix
        self._feature_extractor = None
        # split_to_size might be determined later in preprocess_dataset_pipeline
        self.split_to_size = {}

    @property
    def nlp(self) -> stanza.Pipeline:
        """Implementation of interface property"""
        if self._nlp is None:
             self._initialize_stanza_pipeline() # Ensure initialized on first access
        return self._nlp

    # Make sure setter is defined if needed by parent class or elsewhere
    @nlp.setter
    def nlp(self, value):
         self._nlp = value

    def _initialize_stanza_pipeline(self):
        # Ensure Stanza is only initialized once
        if self._nlp is not None:
            return
        # Use a reasonable default if OMP_NUM_THREADS not set
        num_threads = int(os.environ.get("OMP_NUM_THREADS", 4))
        try:
            # Use GPU if available, fall back to CPU otherwise
            use_gpu = torch.cuda.is_available()
            self.logger.info(f"Initializing Stanza pipeline. GPU available: {use_gpu}, Num Threads: {num_threads}")
            self._nlp = stanza.Pipeline(
                lang=STANZA_LANG,
                processors=STANZA_PROCESSORS,
                use_gpu=use_gpu,
                verbose=False,
                num_threads=num_threads, # Use specified thread count
                # Add memory batch size if needed, e.g., tokenize_batch_size=1000
            )
            self.logger.info(f"Stanza pipeline initialized with processors: {STANZA_PROCESSORS}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Stanza pipeline: {str(e)}")
            self.logger.info("Attempting to download Stanza models...")
            try:
                stanza.download(STANZA_LANG)
                # Retry initialization after download
                use_gpu = torch.cuda.is_available()
                self._nlp = stanza.Pipeline(
                    lang=STANZA_LANG,
                    processors=STANZA_PROCESSORS,
                    use_gpu=use_gpu,
                    verbose=False,
                    num_threads=num_threads
                )
                self.logger.info("Stanza pipeline initialized after downloading models.")
            except Exception as download_e:
                self.logger.error(f"Failed to download or initialize Stanza after download: {download_e}")
                raise RuntimeError("Could not initialize Stanza pipeline.") from download_e

    # ------------------------

    def _process_split(self, dataset_name: str, split_name: str, split_data: pd.DataFrame,
                       suffix: str, force_reprocess: bool) -> None: # Pass suffix
        """Process dataset split with feature extraction pipeline."""
        self.logger.info(f"Processing {dataset_name} / {split_name} (Suffix: {suffix})")

        # 1. Prepare sentence pairs and extract unique sentences from the input data
        pairs_df, sentences_df = self.prepare_sentence_pairs(split_data, dataset_name, split_name)

        if pairs_df.empty or sentences_df.empty:
             self.logger.warning(f"Empty pairs or sentences generated for {dataset_name}/{split_name}. Skipping further processing for this split.")
             return

        # Define table names using the suffix
        pairs_table = f"pairs_{suffix}"
        sentences_table = f"sentences_{suffix}"
        # parse_trees_table = f"parse_trees_{suffix}"

        # 2. Store the generated pairs and unique sentences
        self.db_handler.store_dataframe(pairs_df, dataset_name, split_name, pairs_table)
        self.db_handler.store_dataframe(sentences_df, dataset_name, split_name, sentences_table)
        self.logger.info(f"Stored intermediate data: {pairs_table}, {sentences_table}")

        # 3. Execute Stanza parsing on the unique sentences
        self.logger.info(f"Running Stanza parsing for {len(sentences_df)} sentences ({suffix})")
        parse_trees_df = self.run_stanza_parsing(
            sentences_df=sentences_df, # Pass the dataframe directly
            dataset_name=dataset_name,
            split=split_name,
            suffix=suffix, # Pass suffix
            force_reprocess=force_reprocess
        ) # Renamed from preprocess_dataset for clarity

        if parse_trees_df is not None and not parse_trees_df.empty:
             # Stanza parsing was successful (or loaded from cache)
             # 4. Trigger Feature Extraction (using the stored intermediate data)
             self.logger.info(f"Triggering feature extraction for {dataset_name}/{split_name} ({suffix})")
             self._initialize_feature_extractor() # Ensure extractor is ready
             self._feature_extractor.extract_features(
                 dataset_name=dataset_name,
                 split=split_name,
                 suffix=suffix, # Pass suffix to feature extractor
                 force_recompute=force_reprocess # Use force_reprocess flag
                 # Removed sample_size, as suffix implies it
             )
        else:
             self.logger.warning(f"Parse trees DataFrame is empty for {dataset_name}/{split_name} ({suffix}). Skipping feature extraction.")

        self.logger.info(f"Finished processing pipeline for {dataset_name} / {split_name} (Suffix: {suffix})")


    # --- Stanza Processing Function ---
    def run_stanza_parsing(self, sentences_df: pd.DataFrame, dataset_name: str, split: str, suffix: str,
                            force_reprocess: bool = False) -> Optional[pd.DataFrame]:
        """Loads or computes Stanza parse trees for the given sentences."""
        parse_trees_table = f"parse_trees_{suffix}"
        self.logger.info(f"Processing Stanza for {dataset_name}/{split}/{parse_trees_table}")

        # Check cache first
        if not force_reprocess and self.db_handler.check_exists(dataset_name, split, parse_trees_table):
            self.logger.info(f"Loading cached parse trees: {parse_trees_table}")
            return self.db_handler.load_dataframe(dataset_name, split, parse_trees_table)

        # Ensure sentences_df is not empty
        if sentences_df.empty:
            self.logger.warning(f"Input sentences DataFrame is empty for {dataset_name}/{split}. Cannot run Stanza.")
            return None # Indicate failure

        # Process sentences with Stanza
        parse_trees_list = self._process_sentences_with_stanza(sentences_df)
        if not parse_trees_list:
             self.logger.warning(f"Stanza processing returned no results for {dataset_name}/{split}.")
             return None # Indicate failure

        parse_trees_df = pd.DataFrame(parse_trees_list)

        # Store the results
        self.db_handler.store_dataframe(parse_trees_df, dataset_name, split, parse_trees_table)
        self.logger.info(f"Stored new parse trees: {parse_trees_table}")
        return parse_trees_df

    # (Keep _process_sentences_with_stanza as is)
    def _process_sentences_with_stanza(self, sentences: pd.DataFrame) -> list:
        parse_trees = []
        # Ensure the NLP pipeline is initialized
        stanza_pipeline = self.nlp # Access via property to ensure initialization
        if not stanza_pipeline:
            self.logger.error("Stanza pipeline not initialized. Cannot process sentences.")
            return []

        self.logger.info(f"Processing {len(sentences)} sentences with Stanza...")
        # Use tqdm for progress bar
        for _, row in tqdm(sentences.iterrows(), total=len(sentences), desc=f"Stanza Parsing"):
            sentence_id, text = row["id"], row["text"]
            # Basic check for valid text input
            if not isinstance(text, str) or not text.strip():
                self.logger.warning(f"Skipping invalid text for sentence {sentence_id}: '{text}'")
                parse_trees.append({"sentence_id": sentence_id, "constituency_tree": "", "dependency_tree": ""})
                continue
            try:
                doc = stanza_pipeline(text)
                # Check if Stanza produced output
                if doc.sentences:
                    # Safely access constituency and dependency trees
                    constituency_tree = str(doc.sentences[0].constituency) if doc.sentences[0].constituency else ""
                    dependency_tree = _extract_dependency_tree(doc.sentences[0].words) if doc.sentences[0].words else "[]" # Default to empty list string
                    parse_trees.append({"sentence_id": sentence_id, "constituency_tree": constituency_tree,
                                        "dependency_tree": dependency_tree})
                else:
                    self.logger.warning(f"Stanza returned no sentences for ID {sentence_id}. Text: '{text[:50]}...'")
                    parse_trees.append({"sentence_id": sentence_id, "constituency_tree": "", "dependency_tree": ""})

            except Exception as e:
                # Log specific error and provide defaults
                self.logger.error(f"Error processing sentence {sentence_id} ('{text[:50]}...'): {str(e)}", exc_info=False) # Set exc_info=False to avoid full trace unless debugging
                parse_trees.append({"sentence_id": sentence_id, "constituency_tree": "", "dependency_tree": ""})
        self.logger.info(f"Finished Stanza processing for {len(sentences)} sentences.")
        return parse_trees


    # --- Helper Methods ---
    def _initialize_feature_extractor(self) -> None:
        """Lazy initialization of feature extractor."""
        if not hasattr(self, '_feature_extractor') or self._feature_extractor is None:
            # Correct the import path if necessary
            from features.feature_extractor import FeatureExtractor
            # Pass self (TextPreprocessor instance) if FeatureExtractor needs it
            self._feature_extractor = FeatureExtractor(self.db_handler, self)
            self.logger.info("FeatureExtractor initialized.")


    def prepare_sentence_pairs(self, split_data: pd.DataFrame, dataset_name: str, split: str) \
            -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Extracts unique sentences and creates pairs dataframe from raw split data."""
        self.logger.info(f"Preparing sentence pairs for {dataset_name}/{split}")
        # Ensure required columns exist
        if "premise_text" not in split_data.columns or "hypothesis_text" not in split_data.columns:
             logger.error(f"Missing 'premise_text' or 'hypothesis_text' in input data for {dataset_name}/{split}")
             # Return empty dataframes with expected columns to prevent downstream errors
             return pd.DataFrame(columns=["id", "premise_id", "hypothesis_id", "label"]), pd.DataFrame(columns=["id", "text"])

        sentences_list, sentence_ids_map = _extract_unique_sentences(split_data, dataset_name, split)
        sentences_df = pd.DataFrame(sentences_list)
        pairs_df = _create_pairs_dataframe(split_data, sentence_ids_map, dataset_name, split)

        self.logger.info(f"Created {len(pairs_df)} pairs and {len(sentences_df)} unique sentences.")
        # Storing is handled in _process_split now
        # self._store_sentence_pairs(pairs_df, sentences_df, dataset_name, split, suffix)
        return pairs_df, sentences_df

    # Removed _store_sentence_pairs as storage happens in _process_split


    def preprocess_dataset_pipeline(self, dataset_name: str,
                                    total_sample_size: Optional[int], # Made explicit Optional
                                    force_reprocess: bool) -> None:
        """Loads data, samples if needed, and runs the processing pipeline for each split."""
        self.logger.info(f"Starting preprocessing pipeline for {dataset_name}. Sample size: {total_sample_size}. Force reprocess: {force_reprocess}")

        # Define standard splits (adjust if dev is not always 'validation')
        split_names = ["train", "validation", "test"]

        # Calculate dynamic split sizes only if total_sample_size is provided
        self.split_to_size = {}
        if total_sample_size is not None and total_sample_size > 0:
            train_ratio = 0.8 # Example ratio
            val_ratio = 0.1   # Example ratio
            # Ensure ratios sum close to 1
            # test_ratio = max(0.0, 1.0 - train_ratio - val_ratio)

            train_size = int(total_sample_size * train_ratio)
            val_size = int(total_sample_size * val_ratio)
            # Assign remaining to test split to ensure total matches
            test_size = total_sample_size - train_size - val_size

            self.split_to_size = {
                "train": train_size,
                "validation": val_size,
                "test": test_size
            }
            self.logger.info(f"Calculated sample sizes: {self.split_to_size}")
        else:
            self.logger.info("No total sample size provided, processing full splits.")


        for split_name in split_names:
            self.logger.info(f"--- Processing split: {split_name} ---")

            # Load the full data for the split using DatasetLoader
            # DatasetLoader should ideally handle loading from HF/cache and return a DataFrame
            try:
                # Pass the original split name ('validation', not 'dev' maybe)
                # DatasetLoader might need adjustment based on config.py splits mapping
                full_split_data = self.data_loader.load_dataset(dataset_name, split=split_name)
            except ValueError as e:
                 # Handle cases where a split might not exist for a dataset (e.g., ANLI validation)
                 self.logger.warning(f"Could not load split '{split_name}' for dataset '{dataset_name}': {e}. Skipping this split.")
                 continue # Skip to the next split

            if full_split_data.empty:
                self.logger.warning(f"No data loaded for {dataset_name}/{split_name}. Skipping this split.")
                continue

            # Determine the data to process (sampled or full) and the suffix
            if total_sample_size is not None and split_name in self.split_to_size:
                target_size = self.split_to_size[split_name]
                available_size = len(full_split_data)

                if target_size <= 0:
                    self.logger.info(f"Sample size for {split_name} is {target_size}. Skipping sampling and processing for this split.")
                    continue # Skip if calculated size is zero or negative

                if target_size >= available_size:
                    self.logger.info(f"Requested sample size {target_size} >= available {available_size} for {split_name}. Using all available data.")
                    data_to_process = full_split_data
                    # Suffix reflects the *intended* sample size for consistency, even if using full data
                    # Or adjust suffix logic if needed: suffix = f"full_as_sample{target_size}"
                    current_suffix = f"sample{target_size}"
                else:
                    self.logger.info(f"Sampling {target_size} examples for {split_name} from {available_size}.")
                    # Ensure stratification if labels are available and sampling is significant
                    if 'label' in full_split_data.columns and len(full_split_data['label'].unique()) > 1:
                         try:
                              # Stratified sampling
                              data_to_process = full_split_data.groupby('label', group_keys=False).apply(
                                   lambda x: x.sample(n=min(len(x), max(1, int(target_size * len(x) / available_size))), random_state=42) # Proportional sampling within strata
                                   # Adjust n calculation if simple count per label is desired:
                                   # n=min(len(x), target_size // len(full_split_data['label'].unique()))
                              ).sample(frac=1, random_state=42) # Shuffle after stratified sample
                              # Ensure we don't exceed target size due to rounding in strata
                              if len(data_to_process) > target_size:
                                   data_to_process = data_to_process.sample(n=target_size, random_state=42)
                         except Exception as e:
                              logger.warning(f"Stratified sampling failed for {split_name}: {e}. Falling back to simple random sampling.")
                              data_to_process = full_split_data.sample(n=target_size, random_state=42)
                    else:
                         # Simple random sampling if no labels or only one label
                         data_to_process = full_split_data.sample(n=target_size, random_state=42)
                    current_suffix = f"sample{target_size}"
            else:
                # Process the full split
                self.logger.info(f"Processing full {split_name} split ({len(full_split_data)} examples).")
                data_to_process = full_split_data
                current_suffix = "full" # Use 'full' suffix for clarity

            # Set the determined suffix for this split's processing run
            self.suffix = current_suffix
            self.logger.info(f"Using suffix '{self.suffix}' for {dataset_name}/{split_name}")


            # Store the raw/sampled data that will be processed (optional but good for reproducibility)
            # Use a table name indicating it's the input data for this suffix
            raw_input_table = f"raw_input_{self.suffix}"
            self.db_handler.store_dataframe(data_to_process, dataset_name, split_name, raw_input_table)
            self.logger.info(f"Stored raw input data for processing: {raw_input_table}")


            # Process the selected data (sampled or full) for this split
            self._process_split(
                 dataset_name=dataset_name,
                 split_name=split_name,
                 split_data=data_to_process, # Pass the data to be processed
                 suffix=self.suffix, # Pass the determined suffix
                 force_reprocess=force_reprocess
            )

        self.logger.info(f"Completed preprocessing pipeline for {dataset_name}")


# --- Interface Properties/Methods ---
# Ensure these are implemented if defined in the base classes

# Example implementation for NLPBaseComponent logger if not inherited
# class NLPBaseComponent:
#     def __init__(self, db_handler=None): # Added db_handler potentially
#         self.logger = logging.getLogger(self.__class__.__name__)
#         self.db_handler = db_handler # Store if needed by base

# class PreprocessorInterface:
#     @property
#     def nlp(self):
#         raise NotImplementedError
#     def preprocess_dataset_pipeline(self, dataset_name, total_sample_size, force_reprocess):
#         raise NotImplementedError
# --- End Interface ---