# data/data_loader.py
import logging
from typing import Optional, List, Any  # Added Dict

import pandas as pd
from datasets import load_dataset

logger = logging.getLogger(__name__)

# Import DATA_DIR and DATASETS config
from config import DATASETS, HF_CACHE_DIR

# Import DatabaseHandler if storing directly
from utils.database import DatabaseHandler


def _map_hf_split_name(dataset_name: str, requested_split: str) -> str:
    """Maps standard split names (train, validation, test) to HuggingFace split names using config."""
    if dataset_name not in DATASETS:
        logger.warning(
            f"Dataset '{dataset_name}' not in config. Using requested split name '{requested_split}' directly.")
        return requested_split

    # Use 'dev' from config if requested_split is 'validation'
    split_to_use = 'dev' if requested_split == 'validation' else requested_split
    hf_split = DATASETS[dataset_name]["splits"].get(split_to_use)

    if not hf_split:
        logger.warning(
            f"Split '{requested_split}' (mapped to '{split_to_use}') not found in config for {dataset_name}. Trying '{requested_split}' directly.")
        return requested_split  # Fallback to original name
    logger.debug(f"Mapped '{requested_split}' to HuggingFace split '{hf_split}' for {dataset_name}")
    return hf_split


def _convert_hf_to_dataframe(hf_dataset: Any, dataset_name: str, standard_split_name: str) -> pd.DataFrame:
    """
    Converts HuggingFace dataset object (or dict entry) to a standardized pandas DataFrame.
    'standard_split_name' is the generic name like 'train', 'dev', 'test'.
    """
    logger.debug(f"Converting HF data for {dataset_name}/{standard_split_name}")

    current_df = pd.DataFrame()
    # Handle DatasetDict vs Dataset (check if it has splits or is a single split)
    if hasattr(hf_dataset, 'keys') and isinstance(hf_dataset, dict):
        # This case might occur if load_dataset returns a DatasetDict even when a split is specified.
        # We are usually interested in the first (and ideally only) key if a specific split was loaded.
        if standard_split_name in hf_dataset:  # Check if the standard name itself is a key
            current_df = pd.DataFrame(hf_dataset[standard_split_name])
        elif hf_dataset.keys():  # Try the first available key if standard_split_name is not a direct match
            first_key = list(hf_dataset.keys())[0]
            logger.warning(
                f"Received DatasetDict in _convert_hf_to_dataframe for {dataset_name}/{standard_split_name}. "
                f"Using data from the first key: '{first_key}'.")
            current_df = pd.DataFrame(hf_dataset[first_key])
        else:
            logger.error(f"Received an empty DatasetDict for {dataset_name}/{standard_split_name}.")
            return pd.DataFrame()
    elif hasattr(hf_dataset, 'features'):  # It's likely a Dataset object (single split)
        current_df = pd.DataFrame(hf_dataset)
    else:
        logger.error(f"Input to _convert_hf_to_dataframe for {dataset_name}/{standard_split_name} "
                     "is not a recognized HuggingFace Dataset or DatasetDict.")
        return pd.DataFrame()

    if current_df.empty:
        logger.warning(f"Initial DataFrame for {dataset_name}/{standard_split_name} is empty before standardization.")
        return pd.DataFrame()

    # --- Data Cleaning and Standardization ---
    initial_rows = len(current_df)
    if 'label' in current_df.columns:
        current_df['label'] = pd.to_numeric(current_df['label'], errors='coerce')
        current_df = current_df.dropna(subset=['label'])
        current_df = current_df[current_df['label'] != -1]
        if len(current_df) < initial_rows:
            logger.debug(
                f"Removed {initial_rows - len(current_df)} rows with invalid labels (-1 or non-numeric) for {dataset_name}/{standard_split_name}.")

    # Standardize column names using text_cols from config
    text_cols_map = DATASETS.get(dataset_name, {}).get("text_cols", {})
    rename_map = {}
    # Map from config keys ('premise', 'hypothesis') to actual column names in HF dataset
    # to our standard names ('premise_text', 'hypothesis_text')
    for standard_key, hf_col_name in text_cols_map.items():
        if hf_col_name in current_df.columns:
            rename_map[hf_col_name] = f"{standard_key}_text"  # e.g. premise -> premise_text

    # Fallback for common names if not in config or specific dataset handling
    if "premise" in current_df.columns and "premise" not in rename_map.values(): rename_map["premise"] = "premise_text"
    if "hypothesis" in current_df.columns and "hypothesis" not in rename_map.values(): rename_map[
        "hypothesis"] = "hypothesis_text"
    if "sentence1" in current_df.columns and "premise_text" not in rename_map.values(): rename_map[
        "sentence1"] = "premise_text"
    if "sentence2" in current_df.columns and "hypothesis_text" not in rename_map.values(): rename_map[
        "sentence2"] = "hypothesis_text"
    # Add more general mappings if necessary

    if rename_map:
        current_df = current_df.rename(columns=rename_map)
        logger.debug(f"Renamed columns for {dataset_name}/{standard_split_name}: {rename_map}")

    required_cols = ["premise_text", "hypothesis_text", "label"]
    missing_cols = [col for col in required_cols if col not in current_df.columns]
    if missing_cols:
        logger.error(
            f"Dataset {dataset_name}/{standard_split_name} is missing required columns after processing: {missing_cols}. Columns found: {current_df.columns.tolist()}")
        # Depending on strictness, you might return an empty DF or raise an error
        # For now, let's allow it to proceed but log an error. It might fail later.

    if "id" not in current_df.columns:
        current_df["id"] = [f"{dataset_name}_{standard_split_name}_{i}" for i in range(len(current_df))]
        logger.debug(f"Added 'id' column for {dataset_name}/{standard_split_name}.")

    if 'split' not in current_df.columns:
        current_df['split'] = standard_split_name
        logger.debug(
            f"Added 'split' column with value '{standard_split_name}' for {dataset_name}/{standard_split_name}.")

    # Standardize label format (e.g., ensure numeric 0, 1, 2)
    if 'label' in current_df.columns and current_df["label"].dtype == object:
        label_map = {"entailment": 0, "contradiction": 1, "neutral": 2}  # General NLI map
        # For ANLI, labels are already 0, 1, 2. This map is more for SNLI/MNLI if they come as strings.
        original_labels_sample = current_df["label"].unique()[:5]
        current_df["label"] = current_df["label"].map(
            lambda x: label_map.get(x, x if isinstance(x, (int, float)) else -1))
        current_df["label"] = pd.to_numeric(current_df['label'], errors='coerce')
        current_df = current_df.dropna(subset=['label'])
        current_df = current_df[current_df['label'] != -1]
        mapped_labels_sample = current_df["label"].unique()[:5]
        if list(original_labels_sample) != list(mapped_labels_sample):  # Log if mapping changed anything
            logger.debug(
                f"Mapped string labels for {dataset_name}/{standard_split_name}. Original sample: {original_labels_sample}, Mapped sample: {mapped_labels_sample}")

    logger.debug(f"Finished converting HF data for {dataset_name}/{standard_split_name}. Shape: {current_df.shape}")
    return current_df


class DatasetLoader:
    """Handles loading NLI datasets from HuggingFace, caching, and optional storage via DatabaseHandler."""

    def __init__(self, db_handler: Optional[DatabaseHandler] = None):
        self.db_handler = db_handler
        self.dataset_mapping = {name: details["hf_name"] for name, details in DATASETS.items()}

    def load_dataset(self, dataset_name: str, split: Optional[str] = None) -> pd.DataFrame:
        """
        Loads a specific split (or all splits if None) for a given dataset.
        Uses HuggingFace 'load_dataset' and standardizes the output DataFrame.

        Args:
            dataset_name: The standard name of the dataset (e.g., "SNLI", "ANLI").
            split: The standard split name ("train", "validation", "test") or None to load all.
                   For ANLI, this will be "train", "dev", or "test" and rounds r1,r2,r3 will be loaded.

        Returns:
            A pandas DataFrame containing the loaded and standardized data.
        """
        if dataset_name not in self.dataset_mapping:
            logger.error(f"Dataset '{dataset_name}' not configured in DATASETS.")
            raise ValueError(f"Dataset '{dataset_name}' not configured in DATASETS.")

        hf_dataset_name = self.dataset_mapping[dataset_name]
        logger.info(
            f"Loading dataset: {dataset_name} (HF: {hf_dataset_name}), Requested standard split: {split or 'all'}")

        # Determine which standard splits (train, dev, test) need to be processed.
        standard_splits_to_process: List[str] = []
        if split:
            standard_splits_to_process.append(split)
        else:  # If no specific split is requested, load all splits defined in config for the dataset
            if dataset_name in DATASETS and "splits" in DATASETS[dataset_name]:
                standard_splits_to_process.extend(DATASETS[dataset_name]["splits"].keys())
            else:
                logger.error(
                    f"Cannot load 'all' splits for {dataset_name} as it's not fully defined in DATASETS config with a 'splits' dictionary.")
                return pd.DataFrame()

        if not standard_splits_to_process:
            logger.error(
                f"No standard splits determined for processing for dataset '{dataset_name}' and split request '{split}'.")
            return pd.DataFrame()

        all_processed_dfs: List[pd.DataFrame] = []

        for standard_split_name in standard_splits_to_process:  # e.g. "train", "dev", "test"
            # Map standard split name (train) to HuggingFace base name (e.g., train, validation_matched)
            # For ANLI (with updated config), "train" -> "train", "dev" -> "dev", "test" -> "test"
            hf_base_split_name = _map_hf_split_name(dataset_name, standard_split_name)

            logger.debug(
                f"Processing standard split: '{standard_split_name}' (HF base: '{hf_base_split_name}') for dataset {dataset_name}")

            # --- DB Check (Optional) ---
            # This part is simplified; adapt if your DB logic is more complex or per-round for ANLI
            storage_table_name = f"raw_data_{standard_split_name}_full"  # e.g. raw_data_train_full
            if self.db_handler and self.db_handler.check_exists(dataset_name, standard_split_name, storage_table_name):
                logger.info(
                    f"Attempting to load processed {dataset_name}/{standard_split_name} from DB table: {storage_table_name}")
                try:
                    df_from_db = self.db_handler.load_dataframe(dataset_name, standard_split_name, storage_table_name)
                    if not df_from_db.empty:
                        logger.info(
                            f"Successfully loaded {dataset_name}/{standard_split_name} from DB. Shape: {df_from_db.shape}")
                        # Ensure 'split' column is correctly set if loaded from DB
                        df_from_db['split'] = standard_split_name
                        all_processed_dfs.append(df_from_db)
                        continue  # Skip HuggingFace loading for this standard_split_name
                    else:
                        logger.warning(
                            f"Loaded empty DataFrame from DB for {dataset_name}/{standard_split_name}. Will attempt loading from HuggingFace.")
                except Exception as e:
                    logger.warning(
                        f"Error loading {dataset_name}/{standard_split_name} from DB: {e}. Will attempt loading from HuggingFace.")

            # --- Load from HuggingFace ---
            current_split_dfs_to_concat: List[pd.DataFrame] = []

            if dataset_name == "ANLI":
                anli_rounds = ["r1", "r2", "r3"]
                # hf_base_split_name is "train", "dev", or "test"
                anli_actual_hf_splits_to_load = [f"{hf_base_split_name}_{r}" for r in anli_rounds]
                logger.info(
                    f"ANLI: Loading rounds {anli_rounds} for base split '{hf_base_split_name}'. Target HF splits: {anli_actual_hf_splits_to_load}")

                for anli_hf_name_with_round in anli_actual_hf_splits_to_load:
                    try:
                        logger.debug(f"ANLI: Attempting to load {hf_dataset_name}, split: {anli_hf_name_with_round}")
                        hf_data_round = load_dataset(hf_dataset_name, name=None, split=anli_hf_name_with_round,
                                                     cache_dir=HF_CACHE_DIR)
                        # Pass 'standard_split_name' (train, dev, test) for consistent processing in _convert_hf_to_dataframe
                        df_round = _convert_hf_to_dataframe(hf_data_round, dataset_name, standard_split_name)
                        if not df_round.empty:
                            df_round['anli_round'] = anli_hf_name_with_round.split('_')[-1]
                            current_split_dfs_to_concat.append(df_round)
                        else:
                            logger.warning(
                                f"ANLI: Conversion returned empty DataFrame for HF split '{anli_hf_name_with_round}' (standard split '{standard_split_name}').")
                    except ValueError as e:  # Often "Unknown split"
                        logger.warning(
                            f"ANLI: Could not load HF split '{anli_hf_name_with_round}' for dataset {hf_dataset_name}: {e}")
                    except Exception as e:
                        logger.error(
                            f"ANLI: Error processing HF split '{anli_hf_name_with_round}' for {hf_dataset_name}: {e}",
                            exc_info=True)

                if not current_split_dfs_to_concat:
                    logger.warning(f"ANLI: Failed to load any rounds for standard split '{standard_split_name}'.")
                    # Potentially add an empty DataFrame to all_processed_dfs or handle as error
                    # For now, it will just result in this split being empty if no rounds load.
            else:
                # Standard loading for non-ANLI datasets (SNLI, MNLI, etc.)
                try:
                    logger.debug(f"Attempting to load {hf_dataset_name}, split: {hf_base_split_name}")
                    hf_data = load_dataset(hf_dataset_name, name=None, split=hf_base_split_name, cache_dir=HF_CACHE_DIR)
                    df_single = _convert_hf_to_dataframe(hf_data, dataset_name, standard_split_name)
                    if not df_single.empty:
                        current_split_dfs_to_concat.append(df_single)
                    else:
                        logger.warning(
                            f"Conversion returned empty DataFrame for {dataset_name}/{standard_split_name} (HF split: {hf_base_split_name}).")
                except ValueError as e:  # Often "Unknown split"
                    logger.warning(f"Could not load HF split '{hf_base_split_name}' for dataset {hf_dataset_name}: {e}")
                except Exception as e:
                    logger.error(
                        f"Error processing {dataset_name}/{standard_split_name} (HF split: {hf_base_split_name}): {e}",
                        exc_info=True)

            # Concatenate DataFrames for the current standard_split_name (e.g., all ANLI rounds for 'train')
            if current_split_dfs_to_concat:
                final_df_for_standard_split = pd.concat(current_split_dfs_to_concat, ignore_index=True)
                logger.info(
                    f"Successfully loaded and processed {len(final_df_for_standard_split)} examples for {dataset_name}/{standard_split_name}.")
                all_processed_dfs.append(final_df_for_standard_split)

                # --- Store processed data using DatabaseHandler if provided ---
                if self.db_handler:  # and not db_loaded_this_split (a flag you might set earlier)
                    try:
                        logger.info(
                            f"Storing processed data to DB: {dataset_name}/{standard_split_name}/{storage_table_name}")
                        self.db_handler.store_dataframe(final_df_for_standard_split, dataset_name, standard_split_name,
                                                        storage_table_name)
                    except Exception as e:
                        logger.error(
                            f"Failed to store processed data for {dataset_name}/{standard_split_name} in DB: {e}")
            else:
                logger.warning(f"No data loaded for {dataset_name}/{standard_split_name}. This split will be empty.")

        # --- Combine DataFrames from all processed standard splits (if 'split' was None and multiple were loaded) ---
        if not all_processed_dfs:
            logger.error(
                f"Failed to load any data for dataset '{dataset_name}' (Requested standard split(s): {standard_splits_to_process}). Returning empty DataFrame.")
            return pd.DataFrame()
        elif len(all_processed_dfs) == 1:
            # This will be the case if a single 'split' was requested, or if 'all' resulted in one split.
            return all_processed_dfs[0]
        else:
            # This case handles if 'split=None' and multiple standard splits (e.g. train, dev, test) were processed.
            combined_df = pd.concat(all_processed_dfs, ignore_index=True)
            logger.info(
                f"Combined {len(all_processed_dfs)} standard splits for {dataset_name}. Total examples: {len(combined_df)}")
            return combined_df
