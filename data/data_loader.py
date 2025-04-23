# data/data_loader.py
import pandas as pd
from datasets import load_dataset, get_dataset_split_names
from typing import Optional, List, Any # Added List, Dict, Any

# Use logging from config or standard logging
# from config import logging # If logging setup is centralized
import logging
logger = logging.getLogger(__name__)
# ---

# Import DATA_DIR and DATASETS config
from config import DATASETS, HF_CACHE_DIR

# Import DatabaseHandler if storing directly
from utils.database import DatabaseHandler


def _map_hf_split_name(dataset_name: str, requested_split: str) -> str:
    """Maps standard split names (train, validation, test) to HuggingFace split names using config."""
    if dataset_name not in DATASETS:
        logger.warning(f"Dataset '{dataset_name}' not in config. Using requested split name '{requested_split}' directly.")
        return requested_split
    # Use 'dev' from config if requested_split is 'validation'
    split_to_use = 'dev' if requested_split == 'validation' else requested_split
    hf_split = DATASETS[dataset_name]["splits"].get(split_to_use)
    if not hf_split:
        logger.warning(f"Split '{requested_split}' (mapped to '{split_to_use}') not found in config for {dataset_name}. Trying '{requested_split}' directly.")
        return requested_split # Fallback to original name
    logger.debug(f"Mapped '{requested_split}' to HuggingFace split '{hf_split}' for {dataset_name}")
    return hf_split


def _convert_hf_to_dataframe(hf_dataset: Any, dataset_name: str, split_name: str) -> pd.DataFrame:
    """Converts HuggingFace dataset object (or dict entry) to a standardized pandas DataFrame."""
    logger.debug(f"Converting HF data for {dataset_name}/{split_name}")
    # Handle DatasetDict vs Dataset (check if it has splits or is a single split)
    if hasattr(hf_dataset, 'keys') and isinstance(hf_dataset, dict):
        # It's likely a DatasetDict - this function expects a single split Dataset
        logger.warning("Received DatasetDict in _convert_hf_to_dataframe. Expected single Dataset. Check loading logic.")
        # Attempt to process the split if the key matches
        if split_name in hf_dataset:
             df = pd.DataFrame(hf_dataset[split_name])
        else:
             logger.error(f"Split '{split_name}' not found in provided DatasetDict.")
             return pd.DataFrame() # Return empty
    elif hasattr(hf_dataset, 'features'):
        # It's likely a Dataset object (single split)
        df = pd.DataFrame(hf_dataset)
    else:
        logger.error("Input to _convert_hf_to_dataframe is not a recognized HuggingFace Dataset or Dict.")
        return pd.DataFrame()

    # --- Data Cleaning and Standardization ---
    # Remove rows with invalid labels (e.g., -1 in SNLI/MNLI)
    initial_rows = len(df)
    if 'label' in df.columns:
        # Ensure label column is numeric if possible, handle errors
        df['label'] = pd.to_numeric(df['label'], errors='coerce')
        df = df.dropna(subset=['label']) # Drop rows where label couldn't be numeric
        df = df[df['label'] != -1] # Filter out -1 specifically
        if len(df) < initial_rows:
             logger.debug(f"Removed {initial_rows - len(df)} rows with invalid labels (-1 or non-numeric).")

    # Standardize column names (premise, hypothesis, label)
    rename_map = {}
    if "premise" in df.columns: rename_map["premise"] = "premise_text"
    if "hypothesis" in df.columns: rename_map["hypothesis"] = "hypothesis_text"
    if "sentence1" in df.columns: rename_map["sentence1"] = "premise_text" # Overwrite if both exist? Decide priority.
    if "sentence2" in df.columns: rename_map["sentence2"] = "hypothesis_text"
    # Add more mappings if needed for other datasets

    if rename_map:
         df = df.rename(columns=rename_map)
         logger.debug(f"Renamed columns: {rename_map}")

    # Check if essential columns exist after rename
    required_cols = ["premise_text", "hypothesis_text", "label"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
         logger.error(f"Dataset {dataset_name}/{split_name} is missing required columns after processing: {missing_cols}")
         # Decide how to handle: return empty, raise error, or continue with missing data?
         # return pd.DataFrame() # Example: return empty

    # Add unique IDs if not present (using dataset, split, and index)
    if "id" not in df.columns:
        df["id"] = [f"{dataset_name}_{split_name}_{i}" for i in range(len(df))]
        logger.debug("Added 'id' column.")

    # Add 'split' column for clarity
    if 'split' not in df.columns:
         df['split'] = split_name
         logger.debug(f"Added 'split' column with value '{split_name}'.")

    # Optional: Standardize label format (e.g., ensure numeric 0, 1, 2)
    # This was handled partially by filtering -1 and coercing to numeric.
    # If string labels exist, map them here. Example:
    if df["label"].dtype == object:
        label_map = {"entailment": 0, "contradiction": 1, "neutral": 2}
        df["label"] = df["label"].map(lambda x: label_map.get(x, -1)) # Map or set invalid to -1
        df = df[df['label'] != -1] # Filter again if mapping introduced -1

    logger.debug(f"Finished converting HF data. Shape: {df.shape}")
    return df


class DatasetLoader:
    """Handles loading NLI datasets from HuggingFace, caching, and optional storage via DatabaseHandler."""

    def __init__(self, db_handler: Optional[DatabaseHandler] = None): # Made db_handler optional
        """Initialize the dataset loader."""
        self.db_handler = db_handler
        # Use DATASETS config directly
        self.dataset_mapping = {name: details["hf_name"] for name, details in DATASETS.items()}
        # self.hf_cache_dir = HF_CACHE_DIR # Use cache dir from config
        # DATA_DIR is likely for local Parquet files, not the HF cache itself
        # self.data_dir = DATA_DIR # Local cache dir for processed files (if not using db_handler)
        # os.makedirs(self.data_dir, exist_ok=True)

    def load_dataset(self, dataset_name: str, split: Optional[str] = None) -> pd.DataFrame:
        """
        Loads a specific split (or all splits if None) for a given dataset.
        Uses HuggingFace 'load_dataset' and standardizes the output DataFrame.
        Optionally stores the raw loaded data using DatabaseHandler if provided.

        Args:
            dataset_name: The standard name of the dataset (e.g., "SNLI", "MNLI").
            split: The standard split name ("train", "validation", "test") or None to load all.

        Returns:
            A pandas DataFrame containing the loaded and standardized data for the requested split(s).
        """
        if dataset_name not in self.dataset_mapping:
            raise ValueError(f"Dataset '{dataset_name}' not configured in DATASETS.")

        hf_dataset_name = self.dataset_mapping[dataset_name]
        logger.info(f"Loading dataset: {dataset_name} (HF: {hf_dataset_name}), Split: {split or 'all'}")

        splits_to_load: List[str] = []
        if split:
            # Map the requested standard split name to the HF split name
            hf_split_name = _map_hf_split_name(dataset_name, split)
            splits_to_load.append(hf_split_name)
            standard_split_names = {hf_split_name: split} # Map back for standardization
        else:
            # If no split specified, try loading all configured splits
            standard_split_names = {}
            if dataset_name in DATASETS:
                for std_name, hf_name in DATASETS[dataset_name]["splits"].items():
                     splits_to_load.append(hf_name)
                     standard_split_names[hf_name] = std_name
            else:
                 # Fallback: try getting splits from HF (might be slow)
                 try:
                      logger.info(f"Attempting to get split names from HuggingFace for {hf_dataset_name}")
                      splits_to_load = get_dataset_split_names(hf_dataset_name)
                      # Assume HF names are standard if not in config
                      standard_split_names = {name: name for name in splits_to_load}
                 except Exception as e:
                      logger.error(f"Could not retrieve split names for {hf_dataset_name} from HF: {e}")
                      raise ValueError(f"No splits specified and could not retrieve them for {dataset_name}.") from e


        all_split_dfs = []
        for hf_split in splits_to_load:
            # Determine the standard split name for saving/logging
            standard_split = standard_split_names.get(hf_split, hf_split) # Fallback to hf_split name
            logger.debug(f"Processing HF split: '{hf_split}' (Standard name: '{standard_split}')")

            # --- Define table name for potential storage ---
            # Use 'raw_data_full' as the table name for the originally loaded data
            # (Suffix logic is handled in preprocessor, this is just the initial load)
            storage_table_name = "raw_data_full" # Consistent name for initial load

            # --- Check if db_handler exists and data is already stored ---
            if self.db_handler and self.db_handler.check_exists(dataset_name, standard_split, storage_table_name):
                logger.info(f"Loading raw {dataset_name}/{standard_split} from DB table: {storage_table_name}")
                try:
                    df = self.db_handler.load_dataframe(dataset_name, standard_split, storage_table_name)
                    if not df.empty:
                        # Ensure 'split' column matches the standard split name
                        df['split'] = standard_split
                        all_split_dfs.append(df)
                        continue # Skip HuggingFace load if loaded from DB
                    else:
                        logger.warning(f"Loaded empty DataFrame from DB for {dataset_name}/{standard_split}/{storage_table_name}. Will attempt loading from HuggingFace.")
                except Exception as e:
                    logger.warning(f"Error loading {dataset_name}/{standard_split} from DB table {storage_table_name}: {e}. Will attempt loading from HuggingFace.")

            # --- Load from HuggingFace ---
            try:
                logger.info(f"Loading from HuggingFace: {hf_dataset_name}, split: {hf_split}")
                # Special handling for ANLI which requires loading R1, R2, R3
                if dataset_name == "ANLI":
                     anli_rounds = ["r1", "r2", "r3"]
                     anli_hf_splits = [f"{hf_split}_{r}" for r in anli_rounds] # e.g., train_r1, train_r2,...
                     logger.info(f"ANLI dataset: Loading rounds {anli_rounds} for split {hf_split}")
                     round_dfs = []
                     for anli_split in anli_hf_splits:
                          try:
                               hf_data_round = load_dataset(hf_dataset_name, split=anli_split, cache_dir=HF_CACHE_DIR)
                               df_round = _convert_hf_to_dataframe(hf_data_round, dataset_name, standard_split) # Use standard split name
                               df_round['anli_round'] = anli_split.split('_')[-1] # Add round info
                               round_dfs.append(df_round)
                          except ValueError as e:
                               # Handle if a specific round doesn't exist for a split (e.g., test might only have one round)
                               logger.warning(f"Could not load ANLI split '{anli_split}': {e}")
                     if not round_dfs:
                          logger.error(f"Failed to load any rounds for ANLI split {hf_split}")
                          continue # Skip this split if no rounds loaded
                     hf_data = pd.concat(round_dfs, ignore_index=True)
                else:
                     # Standard load for other datasets
                     hf_data = load_dataset(hf_dataset_name, split=hf_split, cache_dir=HF_CACHE_DIR)

                # Convert to standardized DataFrame
                df = _convert_hf_to_dataframe(hf_data, dataset_name, standard_split) # Pass standard split name

                if df.empty:
                     logger.warning(f"Converted DataFrame is empty for {dataset_name}/{standard_split} (HF split: {hf_split})")
                     continue # Skip if conversion failed

                # --- Store raw data using DatabaseHandler if provided ---
                if self.db_handler:
                    try:
                        logger.info(f"Storing raw loaded data to DB: {dataset_name}/{standard_split}/{storage_table_name}")
                        self.db_handler.store_dataframe(df, dataset_name, standard_split, storage_table_name)
                    except Exception as e:
                        logger.error(f"Failed to store raw data for {dataset_name}/{standard_split} in DB: {e}")

                all_split_dfs.append(df)

            except ValueError as e:
                 # Specific error if split doesn't exist in HF
                 logger.warning(f"HuggingFace split '{hf_split}' not found for dataset {hf_dataset_name}: {e}. Skipping this split.")
            except Exception as e:
                 logger.error(f"Unexpected error loading {hf_dataset_name} split '{hf_split}' from HuggingFace: {e}", exc_info=False)
                 # Optionally raise the error or continue
                 # raise e # Or continue to next split

        # --- Combine DataFrames from all loaded splits ---
        if not all_split_dfs:
            logger.error(f"Failed to load any data for dataset '{dataset_name}' (Split: {split or 'all'}). Returning empty DataFrame.")
            return pd.DataFrame()
        elif len(all_split_dfs) == 1:
             logger.info(f"Loaded {len(all_split_dfs[0])} examples for {dataset_name}/{split}")
             return all_split_dfs[0]
        else:
             combined_df = pd.concat(all_split_dfs, ignore_index=True)
             logger.info(f"Combined {len(all_split_dfs)} splits for {dataset_name}. Total examples: {len(combined_df)}")
             return combined_df

    # Removed _load_single_dataset, logic merged into load_dataset
    # Removed _handle_dataset_specific_splits, logic integrated into load_dataset loop