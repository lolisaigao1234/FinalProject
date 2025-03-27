# data/data_loader.py
import os
import logging
from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from datasets import load_dataset
from pandas import DataFrame
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from config import DATASETS, DATA_DIR

logger = logging.getLogger(__name__)


def _convert_hf_to_dataframe(hf_dataset, dataset_name: str) -> pd.DataFrame:
    """Convert HuggingFace dataset to pandas DataFrame with standardized columns."""
    # Handle DatasetDict vs Dataset
    if hasattr(hf_dataset, 'features'):
        # It's a Dataset object
        df = pd.DataFrame(hf_dataset)
    else:
        # It's a DatasetDict object, we need to merge all splits
        dfs = []
        for split_name, split_dataset in hf_dataset.items():
            split_df = pd.DataFrame(split_dataset)
            split_df['split'] = split_name
            dfs.append(split_df)
        df = pd.concat(dfs, ignore_index=True)

    df = df[df['label'] != -1]

    # Standardize column names
    if "premise" in df.columns and "hypothesis" in df.columns:
        df = df.rename(columns={
            "premise": "premise_text",
            "hypothesis": "hypothesis_text"
        })
    elif "sentence1" in df.columns and "sentence2" in df.columns:
        df = df.rename(columns={
            "sentence1": "premise_text",
            "sentence2": "hypothesis_text"
        })

    # Add unique IDs if not present
    if "id" not in df.columns:
        df["id"] = [f"{dataset_name}_{i}" for i in range(len(df))]

    # Standardize label format
    if "label" in df.columns:
        if df["label"].dtype == object:
            label_map = {"entailment": 0, "contradiction": 1, "neutral": 2}
            df["label"] = df["label"].map(lambda x: label_map.get(x, x))

    return df


class DatasetLoader:
    """Handles loading and preprocessing of NLI datasets from HuggingFace."""

    def __init__(self, db_handler=None):
        """Initialize the dataset loader."""
        self.db_handler = db_handler
        self.datasets = {}

        # Define dataset mapping
        self.dataset_mapping = {
            "SNLI": "stanfordnlp/snli",
            "MNLI": "multi_nli",
            "ANLI": "facebook/anli"
        }

        # Define data directory
        self.data_dir = DATA_DIR
        os.makedirs(self.data_dir, exist_ok=True)

    def load_dataset(self, dataset_name, split=None, sample_size=None):
        """Load dataset by name or list of names."""
        # Handle case when dataset_name is a list
        if isinstance(dataset_name, list):
            logger.info(f"Loading multiple datasets: {', '.join(dataset_name)}")

            # Use list comprehension to process each dataset
            dataframes = [
                self._load_single_dataset(name, split, sample_size)
                for name in dataset_name
            ]

            # Combine all dataframes
            return pd.concat(dataframes, ignore_index=True)
        else:
            # Original single dataset case
            return self._load_single_dataset(dataset_name, split, sample_size)

    def _load_single_dataset(self, dataset_name, split=None, sample_size=None):
        """Load a single dataset by name."""
        try:
            # Load from HuggingFace
            if dataset_name in self.dataset_mapping:
                hf_dataset = load_dataset(self.dataset_mapping[dataset_name], split=split)
                df = _convert_hf_to_dataframe(hf_dataset, dataset_name)

                # Store in database
                if self.db_handler:
                    # if sample_size:
                    #     self.db_handler.store_dataframe(df, dataset_name, split or "all")
                    # else:
                    self.db_handler.store_dataframe(df, dataset_name, split or "all")

                return df
            else:
                raise ValueError(f"Dataset {dataset_name} not configured")

        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {str(e)}")
            raise

    def _handle_dataset_specific_splits(self, dataset_name, split):
        """Handle dataset-specific split naming conventions."""
        # Handle ANLI's specific split naming
        if dataset_name == "ANLI" and split in ["train", "dev", "validation", "test"]:
            split_mapping = {
                "train": ["train_r1", "train_r2", "train_r3"],
                "dev": ["dev_r1", "dev_r2", "dev_r3"],
                "validation": ["dev_r1", "dev_r2", "dev_r3"],
                "test": ["test_r1", "test_r2", "test_r3"]
            }

            specific_splits = split_mapping[split]
            logger.info(f"Loading {dataset_name} {split} splits: {', '.join(specific_splits)}")

            return pd.concat([
                _convert_hf_to_dataframe(
                    load_dataset(self.dataset_mapping[dataset_name], split=s),
                    dataset_name
                )
                for s in specific_splits
            ], ignore_index=True)

        # Add handlers for other datasets with special split naming here

        # Return None if no special handling needed
        return None