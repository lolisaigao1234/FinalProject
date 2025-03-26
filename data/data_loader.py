import os
import logging
from typing import Tuple
import pandas as pd
from datasets import load_dataset

from config import DATASETS, DATA_DIR

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Handles loading and preprocessing of NLI datasets from HuggingFace or local files."""

    def __init__(self, db_handler=None):
        """Initialize the dataset loader.

        Args:
            db_handler: Database handler for storing processed data
        """
        self.db_handler = db_handler
        self.datasets = {}

    def load_dataset(self, dataset_name: str, split: str = None, force_reload: bool = False) -> pd.DataFrame:
        # Check if already loaded and cached
        if not force_reload and self.db_handler and self.db_handler.check_exists(dataset_name, split or "all"):
            logger.info(f"Loading {dataset_name} {split or 'all'} from database")
            return self.db_handler.load_dataframe(dataset_name, split or "all")

        logger.info(f"Loading {dataset_name} dataset from source")

        try:
            # Map dataset names to HuggingFace dataset IDs
            dataset_mapping = {
                "SNLI": "stanfordnlp/snli",
                "MNLI": "nyu-mll/multi_nli",
                "ANLI": "facebook/anli"
            }

            if dataset_name in dataset_mapping:
                # Load from HuggingFace
                hf_dataset = load_dataset(dataset_mapping[dataset_name], split=split)
                df = self._convert_hf_to_dataframe(hf_dataset, dataset_name)

                # Store in database if handler is provided
                if self.db_handler:
                    self.db_handler.store_dataframe(df, dataset_name, split or "all")

                return df
            else:
                raise ValueError(f"Dataset {dataset_name} not configured for HuggingFace loading")

        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {str(e)}")
            raise  # Re-raise the exception instead of falling back to local files

    def _convert_hf_to_dataframe(self, hf_dataset, dataset_name: str) -> pd.DataFrame:
        """Convert HuggingFace dataset to pandas DataFrame with standardized columns."""

        """Convert HuggingFace dataset to pandas DataFrame with standardized columns.

        Args:
            hf_dataset: HuggingFace dataset object
            dataset_name: Name of the dataset

        Returns:
            Standardized pandas DataFrame
        """

        # Convert to pandas DataFrame - handle HuggingFace dataset structure
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

        print("Printing out the df head")
        print(df.head())

        # Standardize column names based on dataset
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

        # Ensure we have the required columns
        required_columns = ["premise_text", "hypothesis_text"]
        for col in required_columns:
            if col not in df.columns:
                logger.warning(f"Required column {col} not found in dataset {dataset_name}")

        # Add unique IDs if not present
        if "id" not in df.columns:
            df["id"] = [f"{dataset_name}_{i}" for i in range(len(df))]

        # Standardize label format if present
        if "label" in df.columns:
            # Ensure label is numeric
            if df["label"].dtype == object:
                # Map string labels to integers
                label_map = {"entailment": 0, "contradiction": 1, "neutral": 2}
                df["label"] = df["label"].map(lambda x: label_map.get(x, x))

        return df

    def _load_from_local(self, dataset_name: str, split: str = None) -> pd.DataFrame:
        """Load dataset from local files specified in config.

        Args:
            dataset_name: Name of the dataset
            split: Dataset split to load

        Returns:
            Pandas DataFrame containing the dataset
        """
        if dataset_name not in DATASETS:
            raise ValueError(f"Dataset {dataset_name} not found in config")

        dataset_config = DATASETS[dataset_name]

        if split:
            split_path = dataset_config.get(f"{split}_path")
            if not split_path or not os.path.exists(split_path):
                raise FileNotFoundError(f"Split {split} not found for dataset {dataset_name}")

            return self._load_file(split_path, dataset_name)
        else:
            # Load all splits
            dfs = []
            for split_key in ["train_path", "dev_path", "test_path"]:
                if split_key in dataset_config and os.path.exists(dataset_config[split_key]):
                    split_name = split_key.replace("_path", "")
                    df = self._load_file(dataset_config[split_key], dataset_name)
                    df["split"] = split_name
                    dfs.append(df)

            if not dfs:
                raise FileNotFoundError(f"No valid splits found for dataset {dataset_name}")

            return pd.concat(dfs, ignore_index=True)

    def _load_file(self, file_path: str, dataset_name: str) -> pd.DataFrame:
        """Load a dataset file based on its extension.

        Args:
            file_path: Path to the dataset file
            dataset_name: Name of the dataset

        Returns:
            Pandas DataFrame containing the dataset
        """
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".jsonl" or ext == ".json":
            df = pd.read_json(file_path, lines=(ext == ".jsonl"))
        elif ext == ".csv":
            df = pd.read_csv(file_path)
        elif ext == ".tsv" or ext == ".txt":
            df = pd.read_csv(file_path, sep="\t")
        elif ext == ".parquet":
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        # Standardize columns
        return self._convert_hf_to_dataframe(df, dataset_name)

    def prepare_sentence_pairs(self, dataset_name: str, split: str = None) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Prepare sentence pairs and individual sentences for a dataset.

        Args:
            dataset_name: Name of the dataset
            split: Dataset split to prepare

        Returns:
            Tuple of (pairs_df, sentences_df, pairs_with_text_df)
        """
        # Load the dataset
        df = self.load_dataset(dataset_name, split)

        # Extract unique sentences
        sentences = []
        sentence_ids = {}

        # Process premises
        for idx, premise in enumerate(df["premise_text"].unique()):
            sentence_id = f"{dataset_name}_p_{idx}"
            sentence_ids[premise] = sentence_id
            sentences.append({"id": sentence_id, "text": premise})

        # Process hypotheses
        for idx, hypothesis in enumerate(df["hypothesis_text"].unique()):
            if hypothesis not in sentence_ids:
                sentence_id = f"{dataset_name}_h_{idx}"
                sentence_ids[hypothesis] = sentence_id
                sentences.append({"id": sentence_id, "text": hypothesis})

        # Create sentences dataframe
        sentences_df = pd.DataFrame(sentences)

        # Create pairs dataframe
        pairs = []
        for idx, row in df.iterrows():
            pair_id = row.get("id", f"{dataset_name}_pair_{idx}")
            premise_id = sentence_ids[row["premise_text"]]
            hypothesis_id = sentence_ids[row["hypothesis_text"]]

            pair = {
                "id": pair_id,
                "premise_id": premise_id,
                "hypothesis_id": hypothesis_id
            }

            if "label" in row:
                pair["label"] = row["label"]

            pairs.append(pair)

        pairs_df = pd.DataFrame(pairs)

        # Create a joined dataframe with text
        pairs_with_text_df = pairs_df.merge(
            sentences_df, left_on="premise_id", right_on="id", how="left"
        ).rename(columns={"text": "premise_text"})

        pairs_with_text_df = pairs_with_text_df.merge(
            sentences_df, left_on="hypothesis_id", right_on="id", how="left", suffixes=("", "_hyp")
        ).rename(columns={"text": "hypothesis_text"})

        # Store in database if handler is provided
        if self.db_handler:
            self.db_handler.store_dataframe(pairs_df, dataset_name, split or "all", "pairs")
            self.db_handler.store_dataframe(sentences_df, dataset_name, split or "all", "sentences")
            self.db_handler.store_dataframe(pairs_with_text_df, dataset_name, split or "all", "pairs_with_text")

        return pairs_df, sentences_df, pairs_with_text_df
