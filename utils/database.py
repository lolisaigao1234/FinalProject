# utils/database.py
import os
import logging
from typing import Dict, List, Union, Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config import PARQUET_DIR

logger = logging.getLogger(__name__)


class DatabaseHandler:
    """Efficient database handler using Parquet for preprocessed data storage."""

    def __init__(self, db_type: str = "parquet"):
        """Initialize the database handler."""
        self.db_type = db_type
        os.makedirs(PARQUET_DIR, exist_ok=True)

    def store_dataframe(self, df: pd.DataFrame, dataset: str, split: str, table: str = None):
        """Store a dataframe in Parquet format."""
        filename = f"{dataset}_{split}_{table if table else 'data'}.parquet"
        filepath = os.path.join(PARQUET_DIR, filename)
        df.to_parquet(filepath, index=False)
        logger.info(f"Saved dataframe to {filepath}")

    def load_dataframe(self, dataset: str, split: str, table: str = None) -> pd.DataFrame:
        """Load a dataframe from Parquet storage."""
        filename = f"{dataset}_{split}_{table if table else 'data'}.parquet"
        filepath = os.path.join(PARQUET_DIR, filename)

        if os.path.exists(filepath):
            return pd.read_parquet(filepath)
        else:
            logger.warning(f"No data found at {filepath}")
            return pd.DataFrame()

    def check_exists(self, dataset: str, split: str, table: str = None) -> bool:
        """Check if data exists in storage."""
        filename = f"{dataset}_{split}_{table if table else 'data'}.parquet"
        filepath = os.path.join(PARQUET_DIR, filename)
        return os.path.exists(filepath)

    def close(self):
        """Placeholder for API compatibility."""
        pass
