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

    def load_dataframe(self, dataset: str, split: str, table: str = None) -> pd.DataFrame:
        """Load a dataframe from Parquet storage or from HuggingFace if not in storage."""
        filename = f"{dataset}_{split}_{table if table else ''}.parquet"
        filepath = os.path.join(PARQUET_DIR, filename)

        logger.info(f"Loading {dataset}/{split}/{table} from {filepath}")

        # Remove the assert False line that was blocking execution

        if os.path.exists(filepath):
            return pd.read_parquet(filepath)
        elif table is None:
            # If no specific table is requested and file doesn't exist, try loading from HuggingFace
            logger.info(f"No cached data found for {dataset}_{split}, attempting to load from HuggingFace")
            return self.load_from_huggingface(dataset, split)
        else:
            logger.warning(f"No data found at {filepath}")
            return pd.DataFrame()

    @staticmethod
    def store_dataframe(df: pd.DataFrame, dataset: str, split: str, table: str = None):
        """Store a dataframe in Parquet format."""
        filename = f"{dataset}_{split}_{table if table else 'data'}.parquet"
        filepath = os.path.join(PARQUET_DIR, filename)
        df.to_parquet(filepath, index=False)
        logger.info(f"Saved dataframe to {filepath}")

    @staticmethod
    def check_exists(dataset: str, split: str, table: str = None) -> bool:
        """Check if data exists in storage."""
        filename = f"{dataset}_{split}_{table if table else 'data'}.parquet"
        filepath = os.path.join(PARQUET_DIR, filename)
        return os.path.exists(filepath)

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

    @staticmethod
    def get_preprocessed_data(dataset: str, split: str) -> dict:
        """
        Load preprocessed data from database for neural network training.

        Args:
            dataset: Name of the dataset (e.g., 'SNLI', 'MNLI')
            split: Data split ('train', 'val', 'test')

        Returns:
            Dictionary containing preprocessed tensors ready for neural network training
        """
        logger.info(f"Loading preprocessed data for {dataset}_{split}")

        # Handle 'val' vs 'validation' naming
        if split == "val":
            split = "validation"

        # Find feature files with lexical and syntactic features
        features_pattern = f"{dataset}_{split}_features_lexical_syntactic_*.parquet"
        features_files = glob.glob(os.path.join(PARQUET_DIR, features_pattern))

        if not features_files:
            logger.error(f"No feature files found for {dataset}_{split}")
            return {}

        # Load the features
        feature_file = features_files[0]
        df_features = pd.read_parquet(feature_file)
        logger.info(f"Loaded features from {os.path.basename(feature_file)}")

        # Check if label column exists directly in the features file
        if 'label' not in df_features.columns:
            logger.error("Missing label column in features file")
            return {}

        try:
            # Extract premise and hypothesis features for syntactic processing
            premise_cols = [col for col in df_features.columns if
                            col.startswith('premise_') and not col.startswith('premise_bert_')]
            hypothesis_cols = [col for col in df_features.columns if
                               col.startswith('hypothesis_') and not col.startswith('hypothesis_bert_')]

            # Create synthetic input_ids - we need to create integer indices for BERT
            # Instead of using bert embeddings as input_ids, we'll create sequential token indices
            batch_size = len(df_features)
            seq_length = 10  # Assuming a sequence length of 10 based on your column structure

            # Create placeholder tensors that BERT can process
            # These are dummy values since we're not actually using BERT for encoding
            input_ids = torch.arange(seq_length).unsqueeze(0).repeat(batch_size, 1)
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
            token_type_ids = torch.zeros((batch_size, seq_length), dtype=torch.long)

            # Now process the actual embeddings as features for your model
            # Extract BERT embedding columns
            premise_bert_cols = [col for col in df_features.columns if col.startswith('premise_bert_')]
            hypothesis_bert_cols = [col for col in df_features.columns if col.startswith('hypothesis_bert_')]

            # Create feature tensors from actual data
            premise_features = torch.tensor(df_features[premise_cols].values, dtype=torch.float)
            hypothesis_features = torch.tensor(df_features[hypothesis_cols].values, dtype=torch.float)

            # Create BERT embedding tensors
            premise_bert_features = torch.tensor(df_features[premise_bert_cols].values, dtype=torch.float)
            hypothesis_bert_features = torch.tensor(df_features[hypothesis_bert_cols].values, dtype=torch.float)

            # Extract labels
            labels = torch.tensor(df_features['label'].values, dtype=torch.long)

            return {
                # Standard BERT inputs (with correct data types)
                "input_ids": input_ids.long(),  # Must be long for embedding lookup
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,

                # Linguistic features
                "syntax_features_premise": premise_features,
                "syntax_features_hypothesis": hypothesis_features,

                # Pre-computed BERT embeddings
                "premise_bert_features": premise_bert_features,
                "hypothesis_bert_features": hypothesis_bert_features,

                "labels": labels
            }

        except Exception as e:
            logger.error(f"Error processing features: {str(e)}")
            return {}

    def clear_cache(self):
        """Clear the dataset cache."""
        self.dataset_cache = {}
        logger.info("Dataset cache cleared")

    def close(self):
        """Placeholder for API compatibility."""
        self.clear_cache()
