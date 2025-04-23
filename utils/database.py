# # utils/database.py
# import logging
# import os
# import torch
# import glob
#
# import pandas as pd
# from datasets import load_dataset
#
# from config import PARQUET_DIR, DATASETS, HF_CACHE_DIR
#
# logger = logging.getLogger(__name__)
#
#
# class DatabaseHandler:
#     """Efficient database handler using Parquet for preprocessed data storage with HuggingFace integration."""
#
#     def __init__(self, db_type: str = "parquet"):
#         """Initialize the database handler."""
#         self.db_type = db_type
#         os.makedirs(PARQUET_DIR, exist_ok=True)
#         self.dataset_cache = {}  # Cache for loaded datasets
#
#     def load_dataframe(self, dataset: str, split: str, table: str = None) -> pd.DataFrame:
#         """Load a dataframe from Parquet storage or from HuggingFace if not in storage."""
#         filename = f"{dataset}_{split}_{table if table else ''}.parquet"
#         filepath = os.path.join(PARQUET_DIR, filename)
#
#         logger.info(f"Loading {dataset}/{split}/{table} from {filepath}")
#
#         # Remove the assert False line that was blocking execution
#
#         if os.path.exists(filepath):
#             return pd.read_parquet(filepath)
#         elif table is None:
#             # If no specific table is requested and file doesn't exist, try loading from HuggingFace
#             logger.info(f"No cached data found for {dataset}_{split}, attempting to load from HuggingFace")
#             return self.load_from_huggingface(dataset, split)
#         else:
#             logger.warning(f"No data found at {filepath}")
#             return pd.DataFrame()
#
#     @staticmethod
#     def store_dataframe(df: pd.DataFrame, dataset: str, split: str, table: str = None):
#         """Store a dataframe in Parquet format."""
#         filename = f"{dataset}_{split}_{table if table else 'data'}.parquet"
#         filepath = os.path.join(PARQUET_DIR, filename)
#         df.to_parquet(filepath, index=False)
#         logger.info(f"Saved dataframe to {filepath}")
#
#     @staticmethod
#     def check_exists(dataset: str, split: str, table: str = None) -> bool:
#         """Check if data exists in storage."""
#         filename = f"{dataset}_{split}_{table if table else 'data'}.parquet"
#         filepath = os.path.join(PARQUET_DIR, filename)
#         return os.path.exists(filepath)
#
#     def load_from_huggingface(self, dataset_name: str, split: str = None) -> pd.DataFrame:
#         """Load a dataset directly from HuggingFace.
#
#         Args:
#             dataset_name: Name of the dataset (SNLI, MNLI, SICK)
#             split: Dataset split to load (train, validation, test)
#
#         Returns:
#             Pandas DataFrame containing the dataset
#         """
#         if dataset_name not in DATASETS:
#             raise ValueError(f"Dataset {dataset_name} not configured in config.py")
#
#         # Get HuggingFace dataset name and split mapping
#         hf_name = DATASETS[dataset_name]["hf_name"]
#
#         # Determine the actual split name in HuggingFace
#         if split:
#             hf_split = DATASETS[dataset_name]["splits"].get(split)
#             if not hf_split:
#                 raise ValueError(f"Split {split} not configured for dataset {dataset_name}")
#         else:
#             hf_split = None  # Load all splits
#
#         # Create cache key
#         cache_key = f"{hf_name}_{hf_split if hf_split else 'all'}"
#
#         # Check if already in cache
#         if cache_key in self.dataset_cache:
#             return self.dataset_cache[cache_key]
#
#         try:
#             # Load from HuggingFace
#             logger.info(f"Loading {dataset_name} ({hf_name}) from HuggingFace, split: {hf_split or 'all'}")
#             hf_dataset = load_dataset(hf_name, split=hf_split, cache_dir=HF_CACHE_DIR)
#
#             # Convert to pandas DataFrame
#             if isinstance(hf_dataset, dict):
#                 # Multiple splits were loaded
#                 dfs = []
#                 for split_name, split_data in hf_dataset.items():
#                     df = pd.DataFrame(split_data)
#                     df["split"] = split_name
#                     dfs.append(df)
#                 df = pd.concat(dfs, ignore_index=True)
#             else:
#                 # Single split was loaded
#                 df = pd.DataFrame(hf_dataset)
#
#             # Cache the result
#             self.dataset_cache[cache_key] = df
#
#             return df
#
#         except Exception as e:
#             logger.error(f"Error loading dataset {dataset_name} from HuggingFace: {str(e)}")
#             raise
#
#     @staticmethod
#     def get_preprocessed_data(dataset: str, split: str) -> dict:
#         """
#         Load preprocessed data from database for neural network training.
#
#         Args:
#             dataset: Name of the dataset (e.g., 'SNLI', 'MNLI')
#             split: Data split ('train', 'val', 'test')
#
#         Returns:
#             Dictionary containing preprocessed tensors ready for neural network training
#         """
#         logger.info(f"Loading preprocessed data for {dataset}_{split}")
#
#         # Handle 'val' vs 'validation' naming
#         if split == "val":
#             split = "validation"
#
#         # Find feature files with lexical and syntactic features
#         features_pattern = f"{dataset}_{split}_features_lexical_syntactic_*.parquet"
#         features_files = glob.glob(os.path.join(PARQUET_DIR, features_pattern))
#
#         if not features_files:
#             logger.error(f"No feature files found for {dataset}_{split}")
#             return {}
#
#         # Load the features
#         feature_file = features_files[0]
#         df_features = pd.read_parquet(feature_file)
#         logger.info(f"Loaded features from {os.path.basename(feature_file)}")
#
#         # Check if label column exists directly in the features file
#         if 'label' not in df_features.columns:
#             logger.error("Missing label column in features file")
#             return {}
#
#         try:
#             # Extract premise and hypothesis features for syntactic processing
#             premise_cols = [col for col in df_features.columns if
#                             col.startswith('premise_') and not col.startswith('premise_bert_')]
#             hypothesis_cols = [col for col in df_features.columns if
#                                col.startswith('hypothesis_') and not col.startswith('hypothesis_bert_')]
#
#             # Create synthetic input_ids - we need to create integer indices for BERT
#             # Instead of using bert embeddings as input_ids, we'll create sequential token indices
#             batch_size = len(df_features)
#             seq_length = 10  # Assuming a sequence length of 10 based on your column structure
#
#             # Create placeholder tensors that BERT can process
#             # These are dummy values since we're not actually using BERT for encoding
#             input_ids = torch.arange(seq_length).unsqueeze(0).repeat(batch_size, 1)
#             attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
#             token_type_ids = torch.zeros((batch_size, seq_length), dtype=torch.long)
#
#             # Now process the actual embeddings as features for your model
#             # Extract BERT embedding columns
#             premise_bert_cols = [col for col in df_features.columns if col.startswith('premise_bert_')]
#             hypothesis_bert_cols = [col for col in df_features.columns if col.startswith('hypothesis_bert_')]
#
#             # Create feature tensors from actual data
#             premise_features = torch.tensor(df_features[premise_cols].values, dtype=torch.float)
#             hypothesis_features = torch.tensor(df_features[hypothesis_cols].values, dtype=torch.float)
#
#             # Create BERT embedding tensors
#             premise_bert_features = torch.tensor(df_features[premise_bert_cols].values, dtype=torch.float)
#             hypothesis_bert_features = torch.tensor(df_features[hypothesis_bert_cols].values, dtype=torch.float)
#
#             # Extract labels
#             labels = torch.tensor(df_features['label'].values, dtype=torch.long)
#
#             return {
#                 # Standard BERT inputs (with correct data types)
#                 "input_ids": input_ids.long(),  # Must be long for embedding lookup
#                 "attention_mask": attention_mask,
#                 "token_type_ids": token_type_ids,
#
#                 # Linguistic features
#                 "syntax_features_premise": premise_features,
#                 "syntax_features_hypothesis": hypothesis_features,
#
#                 # Pre-computed BERT embeddings
#                 "premise_bert_features": premise_bert_features,
#                 "hypothesis_bert_features": hypothesis_bert_features,
#
#                 "labels": labels
#             }
#
#         except Exception as e:
#             logger.error(f"Error processing features: {str(e)}")
#             return {}
#
#     def clear_cache(self):
#         """Clear the dataset cache."""
#         self.dataset_cache = {}
#         logger.info("Dataset cache cleared")
#
#     def close(self):
#         """Placeholder for API compatibility."""
#         self.clear_cache()


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
    def _get_final_features_pattern(dataset: str, split: str) -> str:
        """Constructs the pattern for final features Parquet files."""
        # Keep final features directly in PARQUET_DIR
        filename_pattern = f"features_{dataset}_{split}_lexical_syntactic_*.parquet"
        return os.path.join(PARQUET_DIR, filename_pattern)

    @staticmethod
    def _get_final_features_filepath(table: str) -> str:
        """Constructs the filepath for saving final features Parquet files."""
        # Keep final features directly in PARQUET_DIR
        filename = f"{table}.parquet" # The table name itself should distinguish the file
        return os.path.join(PARQUET_DIR, filename)


    def load_dataframe(self, dataset: str, split: str, table: str = None) -> pd.DataFrame:
        """Load a dataframe from Parquet storage or from HuggingFace if not in storage."""
        if table is None:
            # If no specific table is requested, attempt loading from HuggingFace (original data)
            logger.info(f"No specific table requested for {dataset}_{split}, attempting to load original from HuggingFace")
            return self.load_from_huggingface(dataset, split)

        # Construct path for intermediate/processed files
        filepath = self._get_intermediate_filepath(dataset, split, table)
        logger.info(f"Attempting to load {dataset}/{split}/{table} from {filepath}")

        if os.path.exists(filepath):
            logger.info(f"Loading data from {filepath}")
            return pd.read_parquet(filepath)
        else:
            # Check if it's a request for the final features file in the root PARQUET_DIR
            # Note: Loading final features usually happens via get_preprocessed_data,
            # but this provides a way to load them directly if needed, assuming 'table' matches the final feature filename structure.
            final_feature_path = os.path.join(PARQUET_DIR, f"{table}.parquet")
            if os.path.exists(final_feature_path) and table.startswith(f"features_{dataset}_{split}_lexical_syntactic"):
                 logger.info(f"Loading final features data from {final_feature_path}")
                 return pd.read_parquet(final_feature_path)

            logger.warning(f"No data found at {filepath} or corresponding final feature file in root.")
            # If the specific intermediate file doesn't exist, maybe return empty or raise error?
            # For now, returning empty DataFrame. Consider if loading original HF data is desired here.
            return pd.DataFrame()


    def store_dataframe(self, df: pd.DataFrame, dataset: str, split: str, table: str):
        """Store a dataframe in Parquet format based on the table name (step)."""
        if not table:
             raise ValueError("A 'table' name (representing the processing step) is required for storing data.")

        # Check if the table name indicates it's the final lexical-syntactic features
        if table.startswith("features_") and "lexical_syntactic" in table:
             filepath = self._get_final_features_filepath(table)
             logger.info(f"Saving final features dataframe to {filepath}")
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

        # Check intermediate path first
        filepath = self._get_intermediate_filepath(dataset, split, table)
        if os.path.exists(filepath):
            return True

        # If not found, check if it's a final feature file in the root dir
        final_feature_path = os.path.join(PARQUET_DIR, f"{table}.parquet")
        if table.startswith(f"features_{dataset}_{split}_lexical_syntactic") and os.path.exists(final_feature_path):
            return True

        return False

    def load_from_huggingface(self, dataset_name: str, split: str = None) -> pd.DataFrame:
        """Load a dataset directly from HuggingFace.

        Args:
            dataset_name: Name of the dataset (SNLI, MNLI, SICK)
            split: Dataset split to load (train, validation, test)

        Returns:
            Pandas DataFrame containing the dataset
        """
        if dataset_name not in DATASETS:
            raise ValueError(f"Dataset {dataset_name} not configured in config.py")

        # Get HuggingFace dataset name and split mapping
        hf_name = DATASETS[dataset_name]["hf_name"]

        # Determine the actual split name in HuggingFace
        if split:
            hf_split = DATASETS[dataset_name]["splits"].get(split)
            if not hf_split:
                 # Allow direct use of HF split names if not in config mapping
                 logger.warning(f"Split '{split}' not explicitly configured for dataset {dataset_name}. Trying '{split}' directly.")
                 hf_split = split
                # raise ValueError(f"Split {split} not configured for dataset {dataset_name}") # Optional: make stricter
        else:
            hf_split = None  # Load all splits

        # Create cache key
        cache_key = f"{hf_name}_{hf_split if hf_split else 'all'}"

        # Check if already in cache
        if cache_key in self.dataset_cache:
            logger.debug(f"Returning cached HuggingFace data for {cache_key}")
            return self.dataset_cache[cache_key]

        try:
            # Load from HuggingFace
            logger.info(f"Loading {dataset_name} ({hf_name}) from HuggingFace, split: {hf_split or 'all'}")
            hf_dataset = load_dataset(hf_name, split=hf_split, cache_dir=HF_CACHE_DIR)

            # Convert to pandas DataFrame
            if isinstance(hf_dataset, dict):
                # Multiple splits were loaded
                dfs = []
                for split_name, split_data in hf_dataset.items():
                    df = pd.DataFrame(split_data)
                    # Try to determine the 'standard' split name (train, validation, test)
                    standard_split = next((k for k, v in DATASETS[dataset_name]["splits"].items() if v == split_name), split_name)
                    df["split"] = standard_split
                    dfs.append(df)
                df = pd.concat(dfs, ignore_index=True)
            else:
                # Single split was loaded
                df = pd.DataFrame(hf_dataset)
                 # Add split column if loading a single split directly
                if split and 'split' not in df.columns:
                     df['split'] = split


            # Cache the result
            self.dataset_cache[cache_key] = df
            logger.info(f"Successfully loaded and cached data from HuggingFace for {cache_key}")
            return df

        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name} from HuggingFace: {str(e)}")
            raise

    def get_preprocessed_data(self, dataset: str, split: str) -> dict:
        """
        Load final preprocessed data (lexical/syntactic features) for neural network training.

        Args:
            dataset: Name of the dataset (e.g., 'SNLI', 'MNLI')
            split: Data split ('train', 'val', 'test')

        Returns:
            Dictionary containing preprocessed tensors ready for neural network training
        """
        logger.info(f"Loading final preprocessed data for {dataset}_{split}")

        # Handle 'val' vs 'validation' naming consistency if needed
        if split == "val":
            split = "validation" # Or adjust based on your convention

        # Find feature files with lexical and syntactic features in the main PARQUET_DIR
        features_pattern = self._get_final_features_pattern(dataset, split)
        features_files = glob.glob(features_pattern) # Uses the helper method

        if not features_files:
            logger.error(f"No final feature files found matching pattern {features_pattern}")
            return {}

        # Load the features (use the first match, assuming only one final file per split)
        # Consider adding logic if multiple matches are possible/problematic
        feature_file = features_files[0]
        try:
            df_features = pd.read_parquet(feature_file)
            logger.info(f"Loaded final features from {os.path.basename(feature_file)}")
        except Exception as e:
             logger.error(f"Error reading Parquet file {feature_file}: {e}")
             return {}


        # Check if label column exists directly in the features file
        if 'label' not in df_features.columns:
            logger.error(f"Missing 'label' column in final features file: {feature_file}")
            return {}

        try:
            # Extract premise and hypothesis features for syntactic processing
            # Filter based on prefix and avoid BERT embeddings if they are separate
            premise_cols = [col for col in df_features.columns if
                            col.startswith('premise_') and not col.startswith('premise_bert_')]
            hypothesis_cols = [col for col in df_features.columns if
                               col.startswith('hypothesis_') and not col.startswith('hypothesis_bert_')]

            # Create synthetic input_ids - if still needed for your model architecture
            batch_size = len(df_features)
            # Determine seq_length based on your actual feature structure if possible, otherwise use a default
            # Example: Check how many 'premise_bert_embedding_' columns exist
            bert_cols = [col for col in df_features.columns if col.startswith('premise_bert_')]
            seq_length = len(bert_cols) if bert_cols else 10 # Default if no BERT cols found

            # Placeholder tensors if your model expects standard BERT inputs
            input_ids = torch.zeros((batch_size, seq_length), dtype=torch.long) # Use zeros or appropriate placeholders
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
            token_type_ids = torch.zeros((batch_size, seq_length), dtype=torch.long)


            # Extract actual linguistic feature tensors
            premise_features = torch.tensor(df_features[premise_cols].values, dtype=torch.float) if premise_cols else torch.empty(batch_size, 0)
            hypothesis_features = torch.tensor(df_features[hypothesis_cols].values, dtype=torch.float) if hypothesis_cols else torch.empty(batch_size, 0)


            # Extract Pre-computed BERT embeddings if they exist
            premise_bert_cols = [col for col in df_features.columns if col.startswith('premise_bert_')]
            hypothesis_bert_cols = [col for col in df_features.columns if col.startswith('hypothesis_bert_')]

            premise_bert_features = torch.tensor(df_features[premise_bert_cols].values, dtype=torch.float) if premise_bert_cols else torch.empty(batch_size, 0)
            hypothesis_bert_features = torch.tensor(df_features[hypothesis_bert_cols].values, dtype=torch.float) if hypothesis_bert_cols else torch.empty(batch_size, 0)


            # Extract labels
            labels = torch.tensor(df_features['label'].values, dtype=torch.long)

            # --- Construct the final dictionary ---
            data_dict = {
                # Include standard BERT inputs only if your model architecture requires them
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,

                "labels": labels
            }

            # Add linguistic features if they exist
            if premise_features.numel() > 0:
                data_dict["syntax_features_premise"] = premise_features
            if hypothesis_features.numel() > 0:
                data_dict["syntax_features_hypothesis"] = hypothesis_features

            # Add pre-computed BERT embeddings if they exist
            if premise_bert_features.numel() > 0:
                 data_dict["premise_bert_features"] = premise_bert_features
            if hypothesis_bert_features.numel() > 0:
                 data_dict["hypothesis_bert_features"] = hypothesis_bert_features

            # Log the keys being returned for verification
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