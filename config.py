# config.py
# Add 'mnb_bow' to choices for --model_type
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
            "dev": "validation", # standard name 'validation' maps to HF 'validation'
            "test": "test"
        }
    },
    "MNLI": {
        "hf_name": "nyu-mll/multi_nli",
        "splits": {
            "train": "train",
            "dev": "validation_matched", # standard name 'validation' maps to HF 'validation_matched'
            "test": "validation_mismatched" # standard name 'test' maps to HF 'validation_mismatched'
            # Add validation_mismatched if needed, map to e.g., "dev_mismatched"
        }
    },
    "ANLI": {
        "hf_name": "facebook/anli",
        "splits": {
            # Combine rounds for standard splits or handle rounds separately
            # Example: Combining R1, R2, R3 for train
            # You'll need custom logic in data_loader to handle this if using HF directly
            # Or preprocess ANLI into standard train/dev/test parquet files first
            "train": "train_r1+train_r2+train_r3", # Example syntax if HF load_dataset supports it
            "dev": "dev_r1+dev_r2+dev_r3",
            "test": "test_r1+test_r2+test_r3"
            # Simpler approach might be needed depending on data_loader implementation
        }
    }
}

# Performance-critical settings
MAX_SEQ_LENGTH = 128  # Consider 256 if model supports it
BATCH_SIZE = 64  # Increased for A100's 40GB VRAM
GRAD_ACCUM_STEPS = 2  # For larger effective batch sizes
NUM_WORKERS = 8  # Match --cpus-per-task=8 in sbatch
USE_FP16 = True  # Enable mixed precision training
CUDA_BENCHMARK = True  # Enable cuDNN auto-tuner
PIN_MEMORY = True  # Faster data transfer to GPU
TORCH_LOGS = "+dynamo"
TORCHDYNAMO_VERBOSE = 1

# MODEL IDENTIFICATION
HF_MODEL_IDENTIFIERS = {
    "bert-base-uncased": "bert-base-uncased",
    "roberta-base": "roberta-base",
    "deberta-base": "microsoft/deberta-base"  # Example identifier for DeBERTa V3 base
    # Add other models as needed
}

# You can keep MODEL_NAME for default or specific models
MODEL_NAME = "bert-base-uncased"
ROBERTA_NAME = "roberta-base"
DEBERTA_NAME = "microsoft/deberta-base"

# Ensure NUM_CLASSES is defined (it seems to be 3 already)
NUM_CLASSES = 3
HIDDEN_SIZE = 768  # This might vary slightly for different models, but often 768 for base

# Model optimization
LEARNING_RATE = 3e-5  # Adjusted for larger batch size
WEIGHT_DECAY = 0.01
EPOCHS = 5
# Dimension of syntactic features extracted from Stanza parse trees
# Note: This is specific to the syntax-aware models, not TF-IDF
SYNTACTIC_FEATURE_DIM = 200

# Stanza settings
STANZA_PROCESSORS = "tokenize,pos,lemma,depparse,constituency"
STANZA_LANG = "en"

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
    parser.add_argument("--mode", default="preprocess", choices=["preprocess", "train", "evaluate", "predict"])
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
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Total sample size across splits for preprocessing & training. Processes full dataset if omitted.")
    # --- MODIFIED model_type ---
    parser.add_argument("--model_type", default="svm_syntactic_exp1",  # Changed default for testing
                        choices=["svm", "logistic_tfidf", "mnb_bow", "svm_syntactic_exp1"],  # Added svm_syntactic_exp1
                        help="Model type to use: 'svm', 'logistic_tfidf', 'mnb_bow', 'svm_syntactic_exp1'")
    # ---------------------------
    # --- SVM/Logistic/MNB Args ---
    parser.add_argument("--kernel", default="linear", choices=["linear", "rbf", "poly"], help="SVM kernel type")
    parser.add_argument("--C", type=float, default=1.0, help="SVM/Logistic Regression regularization parameter C")
    parser.add_argument("--max_features", type=int, default=10000, help="Max features for TF-IDF/BoW")
    # Add alpha for Naive Bayes if needed:
    # parser.add_argument("--alpha", type=float, default=1.0, help="Smoothing parameter alpha for MNB")
    # -------------------------
    parser.add_argument("--cross_evaluate", action="store_true", help="Perform cross-dataset evaluation (SVM/Logistic/MNB only)")
    parser.add_argument("--baseline_model_name", type=str, default=None,
                        choices=list(HF_MODEL_IDENTIFIERS.keys()),  # Use keys from config
                        help="Specify a baseline transformer model (bert-base, roberta-base, etc.) for --model_type neural")
    return parser.parse_args()