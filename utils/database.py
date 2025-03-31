# utils/database.py
import os
import logging

import pandas as pd
from datasets import load_dataset

from config import PARQUET_DIR, DATASETS, HF_CACHE_DIR

logger = logging.getLogger(__name__)


def store_dataframe(df: pd.DataFrame, dataset: str, split: str, table: str = None):
    """Store a dataframe in Parquet format."""
    filename = f"{dataset}_{split}_{table if table else 'data'}.parquet"
    filepath = os.path.join(PARQUET_DIR, filename)
    df.to_parquet(filepath, index=False)
    logger.info(f"Saved dataframe to {filepath}")


def check_exists(dataset: str, split: str, table: str = None) -> bool:
    """Check if data exists in storage."""
    filename = f"{dataset}_{split}_{table if table else 'data'}.parquet"
    filepath = os.path.join(PARQUET_DIR, filename)
    return os.path.exists(filepath)


class DatabaseHandler:
    """Efficient database handler using Parquet for preprocessed data storage with HuggingFace integration."""

    def __init__(self, db_type: str = "parquet"):
        """Initialize the database handler."""
        self.db_type = db_type
        os.makedirs(PARQUET_DIR, exist_ok=True)
        self.dataset_cache = {}  # Cache for loaded datasets

    def load_dataframe(self, dataset: str, split: str, table: str = None) -> pd.DataFrame:
        """Load a dataframe from Parquet storage or from HuggingFace if not in storage."""
        filename = f"{dataset}_{split}_{table if table else ''}.parquet"
        filepath = os.path.join(PARQUET_DIR, filename)

        if os.path.exists(filepath):
            return pd.read_parquet(filepath)
        elif table is None:
            # If no specific table is requested and file doesn't exist, try loading from HuggingFace
            logger.info(f"No cached data found for {dataset}_{split}, attempting to load from HuggingFace")
            return self.load_from_huggingface(dataset, split)
        else:
            logger.warning(f"No data found at {filepath}")
            return pd.DataFrame()

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
                raise ValueError(f"Split {split} not configured for dataset {dataset_name}")
        else:
            hf_split = None  # Load all splits

        # Create cache key
        cache_key = f"{hf_name}_{hf_split if hf_split else 'all'}"

        # Check if already in cache
        if cache_key in self.dataset_cache:
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
                    df["split"] = split_name
                    dfs.append(df)
                df = pd.concat(dfs, ignore_index=True)
            else:
                # Single split was loaded
                df = pd.DataFrame(hf_dataset)

            # Cache the result
            self.dataset_cache[cache_key] = df

            return df

        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name} from HuggingFace: {str(e)}")
            raise

    def get_preprocessed_data(self, dataset: str, split: str) -> dict:
        """
        Load preprocessed data from database for neural network training.

        Args:
            dataset: Name of the dataset (e.g., 'snli', 'mnli')
            split: Data split ('train', 'val', 'test')

        Returns:
            Dictionary containing preprocessed tensors ready for neural network training
        """
        import torch  # Add this import at the top of the file if not already present

        logger.info(f"Loading preprocessed data for {dataset}_{split}")

        # Load the main dataframe
        df_main = self.load_dataframe(dataset, split)

        # Load BERT features
        df_bert = self.load_dataframe(dataset, split, table="bert_features")

        # Load syntactic features
        df_syntax = self.load_dataframe(dataset, split, table="syntax_features")

        # Convert pandas dataframes to PyTorch tensors
        input_ids = torch.tensor(df_bert["input_ids"].tolist(), dtype=torch.long)
        attention_mask = torch.tensor(df_bert["attention_mask"].tolist(), dtype=torch.long)
        token_type_ids = torch.tensor(df_bert["token_type_ids"].tolist(), dtype=torch.long)

        syntax_features_premise = torch.tensor(df_syntax["premise_features"].tolist(), dtype=torch.float)
        syntax_features_hypothesis = torch.tensor(df_syntax["hypothesis_features"].tolist(), dtype=torch.float)

        # Convert labels if they exist (not for test set)
        labels = None
        if "label" in df_main.columns:
            labels = torch.tensor(df_main["label"].tolist(), dtype=torch.long)

        # Return the preprocessed data dictionary
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "syntax_features_premise": syntax_features_premise,
            "syntax_features_hypothesis": syntax_features_hypothesis,
            "labels": labels
        }

    def clear_cache(self):
        """Clear the dataset cache."""
        self.dataset_cache = {}
        logger.info("Dataset cache cleared")

    def close(self):
        """Placeholder for API compatibility."""
        self.clear_cache()
