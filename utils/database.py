# utils/database.py
import logging
import os
import torch
import glob

import pandas as pd
from datasets import load_dataset

from config import PARQUET_DIR, DATASETS, HF_CACHE_DIR

logger = logging.getLogger(__name__)


class DatabaseHandler:
    """Efficient database handler using Parquet for preprocessed data storage with HuggingFace integration."""

    def __init__(self, db_type: str = "parquet"):
        """Initialize the database handler."""
        self.db_type = db_type
        os.makedirs(PARQUET_DIR, exist_ok=True)
        self.dataset_cache = {}  # Cache for loaded datasets

    @staticmethod
    def _get_intermediate_filepath(dataset: str, split: str, table: str) -> str:
        """Constructs the filepath for intermediate Parquet files."""
        directory = os.path.join(PARQUET_DIR, dataset, split)
        os.makedirs(directory, exist_ok=True)
        filename = f"{table}.parquet"
        return os.path.join(directory, filename)

    @staticmethod
    def _is_final_features_table(table_name: str) -> bool:
        """Checks if a table name corresponds to the final features format."""
        # Check if it contains the core components of the final feature filename
        # Example: SNLI_train_features_lexical_syntactic_sample100
        parts = table_name.split('_')
        return len(parts) >= 4 and "features" in parts and "lexical" in parts and "syntactic" in parts


    @staticmethod
    def _get_final_features_filepath(table: str) -> str:
        """Constructs the filepath for saving/loading final features Parquet files in the root dir."""
        # Expects 'table' to be the full filename base, e.g., "SNLI_train_features_lexical_syntactic_sample100"
        filename = f"{table}.parquet"
        return os.path.join(PARQUET_DIR, filename)

    # <<< CHANGE THIS METHOD >>>
    @staticmethod
    def _get_final_features_pattern(dataset: str, split: str) -> str:
        """Constructs the pattern for final features Parquet files using the new format."""
        # Keep final features directly in PARQUET_DIR with the new name format
        filename_pattern = f"{dataset}_{split}_features_lexical_syntactic_*.parquet" # UPDATED PATTERN
        return os.path.join(PARQUET_DIR, filename_pattern)


    def load_dataframe(self, dataset: str, split: str, table: str = None) -> pd.DataFrame:
        """Load a dataframe from Parquet storage or from HuggingFace if not in storage."""
        if table is None:
            # If no specific table is requested, attempt loading from HuggingFace (original data)
            logger.info(f"No specific table requested for {dataset}_{split}, attempting to load original from HuggingFace")
            return self.load_from_huggingface(dataset, split)

        # <<< CHANGE THIS SECTION >>>
        # Check if the requested table is potentially a final features file first
        if self._is_final_features_table(table):
            final_feature_path = self._get_final_features_filepath(table)
            if os.path.exists(final_feature_path):
                 logger.info(f"Loading final features data from {final_feature_path}")
                 return pd.read_parquet(final_feature_path)
            else:
                 # If the final feature file doesn't exist, log warning and try intermediate path next
                 logger.warning(f"Final feature file not found at {final_feature_path}. Checking intermediate paths...")

        # If not a final feature file OR if final feature file wasn't found, check intermediate path
        intermediate_filepath = self._get_intermediate_filepath(dataset, split, table)
        logger.debug(f"Attempting to load intermediate file {dataset}/{split}/{table} from {intermediate_filepath}")

        if os.path.exists(intermediate_filepath):
            logger.info(f"Loading intermediate data from {intermediate_filepath}")
            return pd.read_parquet(intermediate_filepath)
        # --- End Change ---

        logger.warning(f"No data found at intermediate path {intermediate_filepath} or final path for table '{table}'.")
        # Consider if loading original HF data is desired here as a last resort?
        return pd.DataFrame()


    def store_dataframe(self, df: pd.DataFrame, dataset: str, split: str, table: str):
        """Store a dataframe in Parquet format based on the table name (step)."""
        if not table:
             raise ValueError("A 'table' name (representing the processing step or final features) is required for storing data.")

        # <<< CHANGE THIS SECTION >>>
        # Check if the table name indicates it's the final lexical-syntactic features using the helper
        if self._is_final_features_table(table):
             # Use the method that saves to the root PARQUET_DIR
             filepath = self._get_final_features_filepath(table)
             logger.info(f"Saving final features dataframe to {filepath}")
        # --- End Change ---
        else:
             # Otherwise, store in the dataset/split subdirectory
             filepath = self._get_intermediate_filepath(dataset, split, table)
             logger.info(f"Saving intermediate dataframe ({table}) to {filepath}")

        df.to_parquet(filepath, index=False)
        logger.info(f"Saved dataframe to {filepath}")


    def check_exists(self, dataset: str, split: str, table: str) -> bool:
        """Check if data exists in storage."""
        if not table:
             raise ValueError("A 'table' name is required for checking existence.")

        # <<< CHANGE THIS SECTION >>>
        # Check if it's potentially a final feature file first
        if self._is_final_features_table(table):
            final_feature_path = self._get_final_features_filepath(table)
            if os.path.exists(final_feature_path):
                return True
            # If final doesn't exist, still check intermediate below

        # Check intermediate path
        intermediate_filepath = self._get_intermediate_filepath(dataset, split, table)
        if os.path.exists(intermediate_filepath):
            return True
        # --- End Change ---

        return False

    def load_from_huggingface(self, dataset_name: str, split: str = None) -> pd.DataFrame:
        """Load a dataset directly from HuggingFace.
        (No changes needed in this method related to final feature filename)
        """
        # ... (keep existing implementation) ...
        if dataset_name not in DATASETS:
            raise ValueError(f"Dataset {dataset_name} not configured in config.py")

        hf_name = DATASETS[dataset_name]["hf_name"]

        if split:
            hf_split = DATASETS[dataset_name]["splits"].get(split)
            if not hf_split:
                 logger.warning(f"Split '{split}' not explicitly configured for dataset {dataset_name}. Trying '{split}' directly.")
                 hf_split = split
        else:
            hf_split = None

        cache_key = f"{hf_name}_{hf_split if hf_split else 'all'}"

        if cache_key in self.dataset_cache:
            logger.debug(f"Returning cached HuggingFace data for {cache_key}")
            return self.dataset_cache[cache_key]

        try:
            logger.info(f"Loading {dataset_name} ({hf_name}) from HuggingFace, split: {hf_split or 'all'}")
            hf_dataset = load_dataset(hf_name, split=hf_split, cache_dir=HF_CACHE_DIR)

            if isinstance(hf_dataset, dict):
                dfs = []
                for split_name, split_data in hf_dataset.items():
                    df = pd.DataFrame(split_data)
                    standard_split = next((k for k, v in DATASETS[dataset_name]["splits"].items() if v == split_name), split_name)
                    df["split"] = standard_split
                    dfs.append(df)
                df = pd.concat(dfs, ignore_index=True)
            else:
                df = pd.DataFrame(hf_dataset)
                if split and 'split' not in df.columns:
                     df['split'] = split

            self.dataset_cache[cache_key] = df
            logger.info(f"Successfully loaded and cached data from HuggingFace for {cache_key}")
            return df

        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name} from HuggingFace: {str(e)}")
            raise


    def get_preprocessed_data(self, dataset: str, split: str) -> dict:
        """
        Load final preprocessed data (lexical/syntactic features) for neural network training.
        Uses the updated pattern to find the correct file.
        """
        logger.info(f"Loading final preprocessed data for {dataset}_{split}")

        # Handle 'val' vs 'validation' naming consistency if needed
        if split == "val":
            split = "validation" # Or adjust based on your convention

        # <<< CHANGE IS HERE (uses updated pattern implicitly via helper) >>>
        # Find feature files with lexical and syntactic features in the main PARQUET_DIR
        # Uses the _get_final_features_pattern which now has the correct pattern
        features_pattern = self._get_final_features_pattern(dataset, split)
        features_files = glob.glob(features_pattern)

        if not features_files:
            logger.error(f"No final feature files found matching pattern '{features_pattern}'")
            return {}

        # --- Rest of the method remains the same ---
        feature_file = features_files[0] # Use the first match
        try:
            df_features = pd.read_parquet(feature_file)
            logger.info(f"Loaded final features from {os.path.basename(feature_file)}")
        except Exception as e:
             logger.error(f"Error reading Parquet file {feature_file}: {e}")
             return {}

        if 'label' not in df_features.columns:
            logger.error(f"Missing 'label' column in final features file: {feature_file}")
            return {}

        try:
            # Extract feature columns (adjust prefixes if needed)
            premise_cols = [col for col in df_features.columns if
                            col.startswith('premise_') and not col.startswith('premise_bert_') and not col.startswith('premise_cls_bert_')]
            hypothesis_cols = [col for col in df_features.columns if
                               col.startswith('hypothesis_') and not col.startswith('hypothesis_bert_') and not col.startswith('hypothesis_cls_bert_')]

            premise_bert_cols = [col for col in df_features.columns if col.startswith('premise_cls_bert_')] # Or premise_bert_
            hypothesis_bert_cols = [col for col in df_features.columns if col.startswith('hypothesis_cls_bert_')] # Or hypothesis_bert_

            # Create Tensors
            batch_size = len(df_features)
            premise_features = torch.tensor(df_features[premise_cols].values, dtype=torch.float) if premise_cols else torch.empty(batch_size, 0)
            hypothesis_features = torch.tensor(df_features[hypothesis_cols].values, dtype=torch.float) if hypothesis_cols else torch.empty(batch_size, 0)
            premise_bert_features = torch.tensor(df_features[premise_bert_cols].values, dtype=torch.float) if premise_bert_cols else torch.empty(batch_size, 0)
            hypothesis_bert_features = torch.tensor(df_features[hypothesis_bert_cols].values, dtype=torch.float) if hypothesis_bert_cols else torch.empty(batch_size, 0)
            labels = torch.tensor(df_features['label'].values, dtype=torch.long)

            # Placeholder tensors for BERT inputs if needed by the model architecture downstream
            # Determine seq_length based on actual BERT features if available
            seq_length = premise_bert_features.shape[1] if premise_bert_features.numel() > 0 else 10 # Example: use embedding dim or default

            input_ids = torch.zeros((batch_size, seq_length), dtype=torch.long) # Dummy IDs
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
            token_type_ids = torch.zeros((batch_size, seq_length), dtype=torch.long)

            data_dict = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "labels": labels
            }
            if premise_features.numel() > 0: data_dict["syntax_features_premise"] = premise_features
            if hypothesis_features.numel() > 0: data_dict["syntax_features_hypothesis"] = hypothesis_features
            if premise_bert_features.numel() > 0: data_dict["premise_bert_features"] = premise_bert_features
            if hypothesis_bert_features.numel() > 0: data_dict["hypothesis_bert_features"] = hypothesis_bert_features

            logger.debug(f"Returning preprocessed data with keys: {list(data_dict.keys())}")
            return data_dict

        except KeyError as e:
             logger.error(f"Missing expected column in feature file {feature_file}: {e}")
             return{}
        except Exception as e:
            logger.error(f"Error processing features from {feature_file}: {str(e)}")
            return {}

    def clear_cache(self):
        """Clear the dataset cache."""
        self.dataset_cache = {}
        logger.info("Dataset cache cleared")

    def close(self):
        """Placeholder for API compatibility."""
        self.clear_cache()