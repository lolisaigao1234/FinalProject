# Modify file: IS567FP/config.py
# Add 'cross_validate_syntactic_experiment_8' to model_type choices
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
            "dev": "validation",  # standard name 'validation' maps to HF 'validation'
            "test": "test"
        },
        # Example: Define premise/hypothesis column names if they differ
        "text_cols": {"premise": "premise", "hypothesis": "hypothesis"}
    },
    "MNLI": {
        "hf_name": "nyu-mll/multi_nli",
        "splits": {
            "train": "train",
            "dev": "validation_matched",  # standard name 'validation' maps to HF 'validation_matched'
            "test": "validation_mismatched"  # standard name 'test' maps to HF 'validation_mismatched'
            # Add validation_mismatched if needed, map to e.g., "dev_mismatched"
        },
        "text_cols": {"premise": "premise", "hypothesis": "hypothesis"}
    },
    "ANLI": {
        "hf_name": "facebook/anli",
        "splits": {
            # Combine rounds for standard splits or handle rounds separately
            # Example: Combining R1, R2, R3 for train
            # You'll need custom logic in data_loader to handle this if using HF directly
            # Or preprocess ANLI into standard train/dev/test parquet files first
            "train": "train_r1",  # Example: Using only R1 for simplicity. Adjust as needed.
            "dev": "dev_r1",
            "test": "test_r1"
            # Combine with '+' if HF loader supports it and if desired: "train_r1+train_r2+train_r3"
        },
        "text_cols": {"premise": "premise", "hypothesis": "hypothesis"}
    }
}

# Performance-critical settings
MAX_SEQ_LENGTH = 128  # Consider 256 if model supports it
BATCH_SIZE = 64  # Increased for A100's 40GB VRAM
GRAD_ACCUM_STEPS = 2  # For larger effective batch sizes
NUM_WORKERS = 64  # Number of workers for data loading (adjust based on system)
USE_FP16 = True  # Enable mixed precision training (relevant for neural models)
CUDA_BENCHMARK = True  # Enable cuDNN auto-tuner
PIN_MEMORY = True  # Faster data transfer to GPU (relevant for neural models)
TORCH_LOGS = "+dynamo"
TORCHDYNAMO_VERBOSE = 1

# Model optimization
MODEL_NAME = "bert-base-uncased"
HIDDEN_SIZE = 768
NUM_CLASSES = 3
LEARNING_RATE = 3e-5  # Adjusted for larger batch size
WEIGHT_DECAY = 0.01
EPOCHS = 5
# Dimension of syntactic features extracted from Stanza parse trees
SYNTACTIC_FEATURE_DIM = 200

STANZA_PROCESSORS = "tokenize,pos,lemma,depparse,constituency"
STANZA_LANG = "en"

# Compute settings
TORCH_COMPILE = False  # Default to False unless explicitly needed/tested for baselines
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Add environment variables for HuggingFace
HF_CACHE_DIR = os.path.join(CACHE_DIR, "huggingface")
os.environ["HF_HOME"] = HF_CACHE_DIR  # Set huggingface cache location
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizer parallelism

# Database settings
DB_TYPE = "parquet"
PARQUET_DIR = os.path.join(CACHE_DIR, "parquet")  # Main directory for Parquet files
os.makedirs(PARQUET_DIR, exist_ok=True)

# <<< HF_MODEL_IDENTIFIERS Placeholder (Replace or remove if not used for baselines) >>>
# Example: Define if you plan to load HuggingFace models directly by name
HF_MODEL_IDENTIFIERS = {
    "bert-base-uncased": "bert-base-uncased",
    # "roberta-base": "roberta-base",
    # Add other identifiers if needed
}

# Add Decision Tree settings
DECISION_TREE_MAX_DEPTH = 10
DECISION_TREE_MIN_SAMPLES_SPLIT = 2
DECISION_TREE_MIN_SAMPLES_LEAF = 1
RANDOM_SEED = 42  # Set a random seed for reproducibility

# If only using baseline models defined in this project, you might not need this dict.
try:
    # Attempt to import dynamically if structure allows, otherwise define statically.
    # This assumes config.py might be imported before models initializes fully in some contexts.
    # Safest is often to define choices statically based on __init__.py or have main.py pass them.
    # For this fix, we'll use the keys directly based on models/__init__.py
    MODEL_CHOICES = [
        "baseline-1", "baseline-2", "baseline-3",
        "experiment-1", "experiment-2", "experiment-3", "experiment-4",
        "experiment-5", "experiment-6", "experiment-7", "experiment-8"
    ]
    DEFAULT_MODEL = "baseline-1"  # Changed default from 'svm'
except ImportError:
    print("Warning: Could not dynamically import MODEL_REGISTRY from models. Using predefined list.")
    # Define statically as a fallback, ensure this list matches models/__init__.py
    MODEL_CHOICES = [
        "baseline-1", "baseline-2", "baseline-3",
        "experiment-1", "experiment-2", "experiment-3", "experiment-4",
        "experiment-5", "experiment-6", "experiment-7", "experiment-8"
    ]
    DEFAULT_MODEL = "baseline-1"


def parse_args():
    """Parse command line arguments with performance-related options"""
    parser = argparse.ArgumentParser(description="NLI Pipeline")
    parser.add_argument("--dataset", default="SNLI", choices=["SNLI", "MNLI", "ANLI"])
    parser.add_argument("--mode", default="preprocess", choices=["preprocess", "train", "evaluate", "predict"])
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help=f"Batch size (default: {BATCH_SIZE}) - primarily for neural models/feature extraction")
    parser.add_argument("--grad_accum", type=int, default=GRAD_ACCUM_STEPS,
                        help="Gradient accumulation steps (for neural models)")
    parser.add_argument("--fp16", action="store_true", default=USE_FP16,
                        help="Enable mixed precision training (for neural models)")
    parser.add_argument("--no_compile", action="store_false", dest="compile", default=not TORCH_COMPILE,
                        help="Disable PyTorch compilation (for neural models)")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Epochs for neural models")
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE,
                        help="Learning rate for neural/GB models")
    parser.add_argument("--force_reprocess", action="store_true",
                        help="Force reprocessing/recomputation of data/features/models")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Total sample size across splits for preprocessing & training. Processes full dataset if omitted.")

    # --- MODIFIED model_type ---
    # Choices now directly reflect the keys in MODEL_REGISTRY from models/__init__.py
    parser.add_argument("--model_type", default=DEFAULT_MODEL, # Use the new default
                        choices=MODEL_CHOICES, # Use the keys from MODEL_REGISTRY
                        help="Model type to train/evaluate, corresponding to keys in models.MODEL_REGISTRY.")
    # ---------------------------

    # --- Baseline Model Hyperparameters ---
    # parser.add_argument("--kernel", default="linear", choices=["linear", "rbf", "poly"], help="SVM kernel type") # REMOVED - Specific to SVM
    parser.add_argument("--C", type=float, default=1.0, help="Logistic Regression regularization parameter C") # Kept for Logistic Regression
    parser.add_argument("--max_features", type=int, default=10000, help="Max features for TF-IDF/BoW")
    parser.add_argument("--alpha", type=float, default=1.0, help="Smoothing parameter alpha for MNB")

    # --- Tree/Ensemble Model Hyperparameters ---
    parser.add_argument("--n_estimators", type=int, default=100,
                        help="Number of trees for Random Forest / Gradient Boosting")
    parser.add_argument("--max_depth", type=int, default=None, help="Max depth for trees (None for no limit)")
    parser.add_argument("--max_iter", type=int, default=1000,
                        help="Max iterations for Logistic Regression.")

    # --- KNN Hyperparameters (Added for Experiment 2) ---
    parser.add_argument("--n_neighbors", type=int, default=5, help="Number of neighbors for KNN (Exp 2)")

    # --- Cross-Evaluation ---
    # Note: The cross_evaluate argument seems tied only to Experiment 7 logic previously.
    # Ensure Experiment 7 ('cross_eval_syntactic_experiment_7.py') correctly uses this flag if needed.
    # Or remove if Experiment 7's logic changed. For now, keeping it as defined previously.
    parser.add_argument("--cross_evaluate", action="store_true",
                        help="Perform cross-dataset evaluation (used by Exp 7)")

    # --- Neural Model Selection (If using neural models - kept for potential future use) ---
    # Assuming HF_MODEL_IDENTIFIERS is defined earlier in config.py
    # HF_MODEL_IDENTIFIERS = { "bert-base-uncased": "bert-base-uncased"} # Example placeholder
    # parser.add_argument("--baseline_model_name", type=str, default=None,
    #                     choices=list(HF_MODEL_IDENTIFIERS.keys()) if 'HF_MODEL_IDENTIFIERS' in globals() and HF_MODEL_IDENTIFIERS else [],
    #                     help="Specify a baseline transformer model if using a neural model type")

    return parser.parse_args()
