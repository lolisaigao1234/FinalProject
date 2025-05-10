# data/preprocessor.py
from typing import Tuple, Optional, Any # Added Any
import pandas as pd
import stanza
import torch
from tqdm import tqdm
import os
from abc import ABC # Import ABC

# Make sure logging is configured if not done elsewhere
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# --- End logging setup ---

from config import STANZA_PROCESSORS, STANZA_LANG
from utils.database import DatabaseHandler
# Correctly import from common.py provided by user
from utils.common import NLPBaseComponent, PreprocessorInterface
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
    if "premise_text" not in split_data.columns or "hypothesis_text" not in split_data.columns:
        logger.error(f"Missing 'premise_text' or 'hypothesis_text' in input for _extract_unique_sentences ({dataset_name}/{split})")
        return [], {}
    premise_texts = split_data["premise_text"].dropna().unique()
    hypothesis_texts = split_data["hypothesis_text"].dropna().unique()

    for idx, premise in enumerate(premise_texts):
        sentence_id = f"{dataset_name}_{split}_p_{idx}"
        sentence_ids[premise] = sentence_id
        sentences.append({"id": sentence_id, "text": premise})
    for idx, hypothesis in enumerate(hypothesis_texts):
        # Check if hypothesis is already processed (as a premise) and is a valid string
        if hypothesis not in sentence_ids and isinstance(hypothesis, str) and hypothesis.strip():
            sentence_id = f"{dataset_name}_{split}_h_{idx}" # Changed index source
            sentence_ids[hypothesis] = sentence_id
            sentences.append({"id": sentence_id, "text": hypothesis})
    return sentences, sentence_ids


def _create_pairs_dataframe(split_data: pd.DataFrame, sentence_ids: dict, dataset_name: str,
                            split: str) -> pd.DataFrame:
    pairs = []
    required_cols = ["premise_text", "hypothesis_text", "label"]
    if not all(col in split_data.columns for col in required_cols):
        logger.error(f"Missing required columns {required_cols} in input for _create_pairs_dataframe ({dataset_name}/{split})")
        return pd.DataFrame(columns=["id", "premise_id", "hypothesis_id", "label"])

    for idx, row in split_data.iterrows():
        # Handle potential missing text or labels gracefully
        premise_text = row.get("premise_text")
        hypothesis_text = row.get("hypothesis_text")
        label = row.get("label", -1) # Default to -1 if missing

        # Check for NaN/None explicitly
        if pd.isna(premise_text) or pd.isna(hypothesis_text) or pd.isna(label) or label == -1:
             logger.warning(f"Skipping row {idx} due to missing data: P='{premise_text}', H='{hypothesis_text}', L='{label}'")
             continue

        pair_id = row.get("id", f"{dataset_name}_{split}_pair_{idx}") # Use existing ID if available
        premise_id = sentence_ids.get(premise_text) # Use .get for safety
        hypothesis_id = sentence_ids.get(hypothesis_text)

        # Ensure both sentences were found in the unique sentence extraction
        if premise_id is not None and hypothesis_id is not None:
            pairs.append(
                {"id": pair_id, "premise_id": premise_id, "hypothesis_id": hypothesis_id, "label": int(label)}) # Ensure label is int
        else:
             logger.warning(f"Could not find sentence IDs for pair {pair_id}. Premise found: {premise_id is not None}, Hypothesis found: {hypothesis_id is not None}")

    if not pairs:
        logger.warning(f"No valid pairs created for {dataset_name} {split}.")
        return pd.DataFrame(columns=["id", "premise_id", "hypothesis_id", "label"]) # Return empty df with columns
    return pd.DataFrame(pairs)


# Implement the required abstract methods from PreprocessorInterface
class TextPreprocessor(NLPBaseComponent, PreprocessorInterface, ABC): # Added ABC back if NLPBaseComponent doesn't inherit it
    def __init__(self, db_handler: DatabaseHandler, sample_size: Optional[int] = None):
        super().__init__(db_handler) # Pass db_handler to parent
        self.db_handler = db_handler # Store db_handler explicitly
        self._nlp = None  # Private backing field for Stanza pipeline
        # self.sample_size = sample_size # sample_size is per-split, total_sample_size is for pipeline
        # Correctly initialize DatasetLoader - pass db_handler if it needs it
        self.data_loader = DatasetLoader(db_handler)
        self._initialize_stanza_pipeline()
        # Suffix will be determined dynamically based on whether sampling occurred
        self.suffix = None # Initialize suffix
        self._feature_extractor = None
        # split_to_size might be determined later in preprocess_dataset_pipeline
        self.split_to_size = {}

    @property
    def nlp(self) -> Any: # Match type hint Any from interface
        """Implementation of interface property"""
        if self._nlp is None:
             self._initialize_stanza_pipeline() # Ensure initialized on first access
        return self._nlp

    # Make sure setter is defined if needed by parent class or elsewhere
    @nlp.setter
    def nlp(self, value: Any): # Match type hint
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
        """Processes a single split: prepares pairs/sentences, runs parsing, triggers feature extraction."""
        self.logger.info(f"Processing {dataset_name} / {split_name} (Suffix: {suffix})")

        # 1. Prepare sentence pairs and extract unique sentences from the input data
        # Use the implementation from PreprocessorInterface
        pairs_df, sentences_df = self.prepare_sentence_pairs(split_data, dataset_name, split_name)

        if pairs_df.empty or sentences_df.empty:
             self.logger.warning(f"Empty pairs or sentences generated for {dataset_name}/{split_name}. Skipping further processing for this split.")
             return

        # Define table names using the suffix
        pairs_table = f"pairs_{suffix}"
        sentences_table = f"sentences_{suffix}"

        # 2. Store the generated pairs and unique sentences
        self.db_handler.store_dataframe(pairs_df, dataset_name, split_name, pairs_table)
        self.db_handler.store_dataframe(sentences_df, dataset_name, split_name, sentences_table)
        self.logger.info(f"Stored intermediate data: {pairs_table}, {sentences_table}")

        # 3. Execute Stanza parsing on the unique sentences
        # This now calls the required preprocess_dataset method which handles parsing for the split
        self.logger.info(f"Running syntactic parsing (preprocess_dataset) for {len(sentences_df)} sentences ({suffix})")
        parse_trees_df = self.preprocess_dataset( # Call the method required by the interface
            dataset_name=dataset_name,
            split=split_name,
            # sample_size=None, # sample_size is not directly relevant here, suffix handles it
            force_reprocess=force_reprocess
        )

        if parse_trees_df is not None and not parse_trees_df.empty:
             # Stanza parsing was successful (or loaded from cache)
             # 4. Trigger Feature Extraction (using the stored intermediate data)
             self.logger.info(f"Triggering feature extraction for {dataset_name}/{split_name} ({suffix})")
             self._initialize_feature_extractor() # Ensure extractor is ready
             # Adapt call if FeatureExtractorInterface changes its signature
             self._feature_extractor.extract_features(
                 dataset_name=dataset_name,
                 split=split_name,
                 suffix=suffix, # Pass suffix to feature extractor
                 force_recompute=force_reprocess # Use force_reprocess flag
             )
        else:
             self.logger.warning(f"Parse trees DataFrame is empty for {dataset_name}/{split_name} ({suffix}). Skipping feature extraction.")

        self.logger.info(f"Finished processing pipeline for {dataset_name} / {split_name} (Suffix: {suffix})")


    # --- Stanza Processing Function (Helper for preprocess_dataset) ---
    def run_stanza_parsing(self, sentences_df: pd.DataFrame, dataset_name: str, split: str, suffix: str,
                            force_reprocess: bool = False) -> Optional[pd.DataFrame]:
        """Loads or computes Stanza parse trees for the given sentences. (Internal helper)"""
        parse_trees_table = f"parse_trees_{suffix}"
        self.logger.info(f"Processing Stanza for {dataset_name}/{split}/{parse_trees_table}")

        # Check cache first
        if not force_reprocess and self.db_handler.check_exists(dataset_name, split, parse_trees_table):
            self.logger.info(f"Loading cached parse trees: {parse_trees_table}")
            return self.db_handler.load_dataframe(dataset_name, split, parse_trees_table)

        # Ensure sentences_df is not empty
        if sentences_df.empty:
            self.logger.warning(f"Input sentences DataFrame is empty for {dataset_name}/{split}. Cannot run Stanza.")
            # Check if sentences can be loaded from DB as fallback
            sentences_table = f"sentences_{suffix}"
            sentences_df = self.db_handler.load_dataframe(dataset_name, split, sentences_table)
            if sentences_df.empty:
                 self.logger.error(f"Sentences still empty after trying DB load for {dataset_name}/{split}/{sentences_table}. Aborting Stanza.")
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

    # --- Implement Abstract Methods ---

    # 1. preprocess_dataset (Concrete Implementation)
    def preprocess_dataset(self, dataset_name: str, split: str,
                           sample_size: Optional[int] = None, # Keep optional as per interface
                           force_reprocess: bool = False) -> pd.DataFrame:
        """
        Processes sentences for a given split using Stanza (loads/computes parse trees).
        This method fulfills the preprocess_dataset abstract method requirement.
        """
        self.logger.info(f"Running preprocess_dataset (Stanza Parsing) for {dataset_name}/{split}. Force: {force_reprocess}")
        # Determine the correct suffix based on sample_size (if provided) or fallback
        # This assumes the suffix relevant for loading sentences is already set or derivable
        # If called independently, might need logic to determine suffix.
        # Using self.suffix which should be set by preprocess_dataset_pipeline
        current_suffix = self.suffix
        if not current_suffix:
            # Fallback if called outside the pipeline context (may not be ideal)
             current_suffix = f"sample{sample_size}" if sample_size else "full"
             self.logger.warning(f"preprocess_dataset called outside pipeline? Using fallback suffix: {current_suffix}")

        # Load sentences corresponding to this split and suffix
        sentences_table = f"sentences_{current_suffix}"
        sentences_df = self.db_handler.load_dataframe(dataset_name, split, sentences_table)

        if sentences_df.empty:
            logger.warning(f"No sentences found in {sentences_table} for {dataset_name}/{split}. Cannot generate parse trees.")
            return pd.DataFrame() # Return empty DataFrame as per signature

        # Call the helper that actually runs stanza
        parse_trees_df = self.run_stanza_parsing(
             sentences_df=sentences_df,
             dataset_name=dataset_name,
             split=split,
             suffix=current_suffix,
             force_reprocess=force_reprocess
        )
        # Return the DataFrame (or None if run_stanza_parsing failed)
        return parse_trees_df if parse_trees_df is not None else pd.DataFrame()


    # 2. prepare_sentence_pairs (Concrete Implementation)
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
        return pairs_df, sentences_df


    # 3. preprocess_dataset_pipeline (Concrete Implementation - signature fixed)
    def preprocess_dataset_pipeline(self, dataset_name: str,
                                    total_sample_size: int, # Changed Optional[int] to int
                                    force_reprocess: bool) -> None:
        """Loads data, samples if needed, and runs the processing pipeline for each split."""
        # Handle case where total_sample_size might be passed as None from args
        effective_sample_size: Optional[int] = total_sample_size if total_sample_size is not None and total_sample_size > 0 else None

        self.logger.info(f"Starting preprocessing pipeline for {dataset_name}. Sample size: {effective_sample_size}. Force reprocess: {force_reprocess}")

        # Define standard splits (adjust if dev is not always 'validation')
        split_names = ["train", "validation", "test"]

        # Calculate dynamic split sizes only if effective_sample_size is provided
        self.split_to_size = {}
        if effective_sample_size is not None:
            # Use ratios to divide the total sample size among splits
            train_ratio = 0.8 # Example ratio, adjust as needed
            val_ratio = 0.1   # Example ratio
            test_ratio = max(0.0, 1.0 - train_ratio - val_ratio) # Ensure non-negative

            train_size = int(effective_sample_size * train_ratio)
            val_size = int(effective_sample_size * val_ratio)
            # Assign remaining to test split to ensure total adds up correctly
            test_size = effective_sample_size - train_size - val_size

            self.split_to_size = {
                "train": train_size,
                "validation": val_size,
                "test": test_size
            }
            self.logger.info(f"Calculated target sample sizes: {self.split_to_size}")
        else:
            self.logger.info("No total sample size provided or zero, processing full splits.")


        for split_name in split_names:
            self.logger.info(f"--- Processing split: {split_name} ---")

            # Load the full data for the split using DatasetLoader
            try:
                full_split_data = self.data_loader.load_dataset(dataset_name, split=split_name)
            except ValueError as e:
                 # Handle cases where a split might not exist
                 self.logger.warning(f"Could not load split '{split_name}' for dataset '{dataset_name}': {e}. Skipping this split.")
                 continue # Skip to the next split

            if full_split_data.empty:
                self.logger.warning(f"No data loaded for {dataset_name}/{split_name}. Skipping this split.")
                continue

            # Determine the data to process (sampled or full) and the suffix
            data_to_process: pd.DataFrame
            current_suffix: str

            if effective_sample_size is not None and split_name in self.split_to_size:
                target_size = self.split_to_size[split_name]
                available_size = len(full_split_data)

                if target_size <= 0:
                    self.logger.info(f"Target sample size for {split_name} is {target_size}. Skipping sampling and processing for this split.")
                    continue # Skip if calculated size is zero or negative

                current_suffix = f"sample{target_size}" # Suffix reflects target size

                if target_size >= available_size:
                    self.logger.info(f"Target sample size {target_size} >= available {available_size} for {split_name}. Using all available data for suffix '{current_suffix}'.")
                    data_to_process = full_split_data
                else:
                    self.logger.info(f"Sampling {target_size} examples for {split_name} (from {available_size}) for suffix '{current_suffix}'.")
                    # Stratified sampling (ensure 'label' column exists)
                    if 'label' in full_split_data.columns and len(full_split_data['label'].unique()) > 1:
                         try:
                              # Calculate proportional sample size per stratum
                              n_per_stratum = (full_split_data['label'].value_counts(normalize=True) * target_size).round().astype(int)
                              # Adjust total size if rounding caused mismatch
                              size_diff = target_size - n_per_stratum.sum()
                              if size_diff != 0:
                                  # Add/remove difference from the largest stratum
                                  largest_stratum = n_per_stratum.idxmax()
                                  n_per_stratum[largest_stratum] += size_diff

                              data_to_process = full_split_data.groupby('label', group_keys=False).apply(
                                   lambda x: x.sample(n=min(len(x), n_per_stratum.get(x.name, 0)), random_state=42) # Use calculated n, protect against 0
                              ).sample(frac=1, random_state=42) # Shuffle results

                              # Final check on size
                              if len(data_to_process) != target_size:
                                   logger.warning(f"Stratified sampling resulted in {len(data_to_process)} rows, expected {target_size}. Adjusting...")
                                   if len(data_to_process) > target_size:
                                        data_to_process = data_to_process.sample(n=target_size, random_state=42)
                                   # If less, it might be due to small strata - usually acceptable

                         except Exception as e:
                              logger.warning(f"Stratified sampling failed for {split_name}: {e}. Falling back to simple random sampling.")
                              data_to_process = full_split_data.sample(n=target_size, random_state=42)
                    else:
                         # Simple random sampling if no labels or only one label
                         logger.info("Performing simple random sampling (no/single label).")
                         data_to_process = full_split_data.sample(n=target_size, random_state=42)
            else:
                # Process the full split
                self.logger.info(f"Processing full {split_name} split ({len(full_split_data)} examples).")
                data_to_process = full_split_data
                current_suffix = "full" # Use 'full' suffix

            # Set the determined suffix for this split's processing run
            self.suffix = current_suffix
            self.logger.info(f"Using suffix '{self.suffix}' for {dataset_name}/{split_name}")

            # Store the raw/sampled data that will be processed (optional but good for reproducibility)
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


    # --- Other Helper Methods ---
    def _initialize_feature_extractor(self) -> None:
        """Lazy initialization of feature extractor."""
        if not hasattr(self, '_feature_extractor') or self._feature_extractor is None:
            # Correct the import path if necessary
            try:
                from features.feature_extractor import FeatureExtractor
                # Pass self (TextPreprocessor instance) if FeatureExtractor needs it
                # FeatureExtractor __init__ was updated to take preprocessor as Optional
                self._feature_extractor = FeatureExtractor(self.db_handler, preprocessor=self)
                self.logger.info("FeatureExtractor initialized.")
            except ImportError:
                 logger.error("Could not import FeatureExtractor. Feature extraction step will be skipped.")
                 self._feature_extractor = None # Ensure it's None if import fails

    # --- Internal Stanza Processing Logic ---
    def _process_sentences_with_stanza(self, sentences: pd.DataFrame) -> list:
        """Helper to run Stanza on a DataFrame of sentences."""
        parse_trees = []
        # Ensure the NLP pipeline is initialized
        stanza_pipeline = self.nlp # Access via property to ensure initialization
        if not stanza_pipeline:
            self.logger.error("Stanza pipeline not initialized. Cannot process sentences.")
            return []

        self.logger.info(f"Processing {len(sentences)} sentences with Stanza...")
        # Use tqdm for progress bar
        for _, row in tqdm(sentences.iterrows(), total=len(sentences), desc=f"Stanza Parsing ({self.suffix})"):
            sentence_id, text = row["id"], row["text"]
            # Basic check for valid text input
            if not isinstance(text, str) or not text.strip():
                self.logger.warning(f"Skipping invalid text for sentence {sentence_id}: '{text}'")
                parse_trees.append({"sentence_id": sentence_id, "constituency_tree": "", "dependency_tree": "[]"}) # Use empty list string
                continue
            try:
                # Limit text length for Stanza if necessary (e.g., avoid extremely long sentences)
                max_stanza_len = 2000 # Example limit
                if len(text) > max_stanza_len:
                    logger.warning(f"Truncating long sentence {sentence_id} (len {len(text)}) to {max_stanza_len} chars for Stanza.")
                    text = text[:max_stanza_len]

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
                    parse_trees.append({"sentence_id": sentence_id, "constituency_tree": "", "dependency_tree": "[]"}) # Use empty list string

            except Exception as e:
                # Log specific error and provide defaults
                self.logger.error(f"Error processing sentence {sentence_id} ('{text[:50]}...'): {str(e)}", exc_info=False) # Set exc_info=False to avoid full trace unless debugging
                parse_trees.append({"sentence_id": sentence_id, "constituency_tree": "", "dependency_tree": "[]"}) # Use empty list string
        self.logger.info(f"Finished Stanza processing for {len(sentences)} sentences.")
        return parse_trees


# --- End Interface Implementation ---