# config.py
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.absolute()
DATA_DIR = os.path.join(PROJECT_ROOT, "data_files")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
MODELS_DIR = os.path.join(PROJECT_ROOT, "saved_models")

# Create directories
for directory in [DATA_DIR, CACHE_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Dataset settings
DATASETS = {
    "SNLI": {
        "train_path": os.path.join(DATA_DIR, "snli_1.0_train.jsonl"),
        "dev_path": os.path.join(DATA_DIR, "snli_1.0_dev.jsonl"),
        "test_path": os.path.join(DATA_DIR, "snli_1.0_test.jsonl"),
    },
    "MNLI": {
        "train_path": os.path.join(DATA_DIR, "multinli_1.0_train.jsonl"),
        "dev_path": os.path.join(DATA_DIR, "multinli_1.0_dev.jsonl"),
        "test_path": os.path.join(DATA_DIR, "multinli_1.0_test.jsonl"),
    },
    "SICK": {
        "train_path": os.path.join(DATA_DIR, "SICK_train.txt"),
        "dev_path": os.path.join(DATA_DIR, "SICK_trial.txt"),
        "test_path": os.path.join(DATA_DIR, "SICK_test.txt"),
    },
}

# Preprocessing settings
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 32
NUM_WORKERS = 4

# Stanza settings
STANZA_PROCESSORS = "tokenize,pos,lemma,depparse,constituency"
STANZA_LANG = "en"

# Model settings
MODEL_NAME = "bert-base-uncased"
HIDDEN_SIZE = 768
NUM_CLASSES = 3  # entailment, contradiction, neutral
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
EPOCHS = 5

# Database settings
DB_TYPE = "parquet"
PARQUET_DIR = os.path.join(CACHE_DIR, "parquet")
os.makedirs(PARQUET_DIR, exist_ok=True)
