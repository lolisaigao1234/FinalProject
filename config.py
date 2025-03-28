# config.py
import os
from pathlib import Path
import argparse
import torch

# Project paths
PROJECT_ROOT = Path(__file__).parent.absolute()
DATA_DIR = os.path.join(PROJECT_ROOT, "data_files")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
MODELS_DIR = os.path.join(PROJECT_ROOT, "saved_models")

# # Create directories
for directory in [DATA_DIR, CACHE_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Dataset settings for HuggingFace datasets
DATASETS = {
    "SNLI": {
        "hf_name": "stanfordnlp/snli",
        "splits": {
            "train": "train",
            "dev": "validation",
            "test": "test"
        }
    },
    "MNLI": {
        "hf_name": "nyu-mll/multi_nli",
        "splits": {
            "train": "train",
            "dev": "validation_matched",
            "test": "test_matched"
        }
    },
    "ANLI": {
        "hf_name": "facebook/anli",
        "splits": {
            "train": "train",
            "dev": "validation",
            "test": "test"
        }
    }
}

# Performance-critical settings
MAX_SEQ_LENGTH = 128  # Consider 256 if model supports it
BATCH_SIZE = 64       # Increased for A100's 40GB VRAM
GRAD_ACCUM_STEPS = 2  # For larger effective batch sizes
NUM_WORKERS = 8       # Match --cpus-per-task=8 in sbatch
USE_FP16 = True       # Enable mixed precision training
CUDA_BENCHMARK = True # Enable cuDNN auto-tuner
PIN_MEMORY = True     # Faster data transfer to GPU

# Stanza settings
STANZA_PROCESSORS = "tokenize,pos,lemma,depparse,constituency"
STANZA_LANG = "en"

# Model optimization
MODEL_NAME = "bert-base-uncased"
HIDDEN_SIZE = 768
NUM_CLASSES = 3
LEARNING_RATE = 3e-5  # Adjusted for larger batch size
WEIGHT_DECAY = 0.01
EPOCHS = 5

# A100-specific settings
TORCH_COMPILE = True  # Enable PyTorch 2.0 compiler if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Add environment variables for HuggingFace
HF_CACHE_DIR = os.path.join(CACHE_DIR, "huggingface")
os.environ["HF_HOME"] = HF_CACHE_DIR  # Set huggingface cache location
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizer parallelism

# Database settings
DB_TYPE = "parquet"
PARQUET_DIR = os.path.join(CACHE_DIR, "parquet")
os.makedirs(PARQUET_DIR, exist_ok=True)

def parse_args():
    """Parse command line arguments with performance-related options"""
    parser = argparse.ArgumentParser(description="NLI Pipeline")
    parser.add_argument("--dataset", default="SNLI", choices=["SNLI", "MNLI", "ANLI"])
    parser.add_argument("--mode", default="train", choices=["preprocess", "train", "evaluate", "predict"])
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                       help=f"Batch size (default: {BATCH_SIZE})")
    parser.add_argument("--grad_accum", type=int, default=GRAD_ACCUM_STEPS,
                       help="Gradient accumulation steps")
    parser.add_argument("--fp16", action="store_true", default=USE_FP16,
                       help="Enable mixed precision training")
    parser.add_argument("--no_compile", action="store_false", dest="compile",
                       help="Disable PyTorch compilation")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--force_reprocess", action="store_true")
    parser.add_argument("--sample_size", type=int, default=300,
                        help="Sample size for preprocessing")
    return parser.parse_args()