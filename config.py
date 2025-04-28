# Modify file: IS567FP/config.py
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
            "dev": "validation", # standard name 'validation' maps to HF 'validation'
            "test": "test"
        },
         # Example: Define premise/hypothesis column names if they differ
         "text_cols": {"premise": "premise", "hypothesis": "hypothesis"}
    },
    "MNLI": {
        "hf_name": "nyu-mll/multi_nli",
        "splits": {
            "train": "train",
            "dev": "validation_matched", # standard name 'validation' maps to HF 'validation_matched'
            "test": "validation_mismatched" # standard name 'test' maps to HF 'validation_mismatched'
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
            "train": "train_r1", # Example: Using only R1 for simplicity. Adjust as needed.
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
NUM_WORKERS = 8  # Match --cpus-per-task=8 in sbatch
USE_FP16 = True  # Enable mixed precision training (relevant for neural models)
CUDA_BENCHMARK = True  # Enable cuDNN auto-tuner
PIN_MEMORY = True  # Faster data transfer to GPU (relevant for neural models)
TORCH_LOGS = "+dynamo"
TORCHDYNAMO_VERBOSE = 1


# Ensure NUM_CLASSES is defined
NUM_CLASSES = 3
HIDDEN_SIZE = 768  # Typical BERT-base hidden size

# Baseline Model optimization/parameters
LEARNING_RATE = 3e-5  # Adjusted for larger batch size (for potential future neural models)
WEIGHT_DECAY = 0.01
EPOCHS = 5
# Dimension of syntactic features extracted from Stanza parse trees
# Note: This is the raw dimension before potential selection/scaling
SYNTACTIC_FEATURE_DIM = 200 # This seems like a placeholder, the actual number of features generated might be different

# Stanza settings
STANZA_PROCESSORS = "tokenize,pos,lemma,depparse,constituency"
STANZA_LANG = "en"

# Compute settings
TORCH_COMPILE = False # Default to False unless explicitly needed/tested for baselines
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Add environment variables for HuggingFace
HF_CACHE_DIR = os.path.join(CACHE_DIR, "huggingface")
os.environ["HF_HOME"] = HF_CACHE_DIR  # Set huggingface cache location
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizer parallelism

# Database settings
DB_TYPE = "parquet"
PARQUET_DIR = os.path.join(CACHE_DIR, "parquet") # Main directory for Parquet files
os.makedirs(PARQUET_DIR, exist_ok=True)


# <<< HF_MODEL_IDENTIFIERS Placeholder (Replace or remove if not used for baselines) >>>
# Example: Define if you plan to load HuggingFace models directly by name
HF_MODEL_IDENTIFIERS = {
    "bert-base-uncased": "bert-base-uncased",
    # "roberta-base": "roberta-base",
    # Add other identifiers if needed
}
# If only using baseline models defined in this project, you might not need this dict.

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
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE, help="Learning rate for neural/GB models") # Updated help text
    parser.add_argument("--force_reprocess", action="store_true", help="Force reprocessing/recomputation of data/features/models")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Total sample size across splits for preprocessing & training. Processes full dataset if omitted.")
    # --- MODIFIED model_type ---
    parser.add_argument("--model_type", default="svm",
                        choices=[
                            "svm", # Trains BoW, Syntax, Combined SVM variants
                            "logistic_tfidf",
                            "mnb_bow",
                            "svm_syntactic_exp1", # SVM with only syntactic features
                            "svm_bow_syntactic_exp2", # SVM with BoW + syntactic features
                            "logistic_tfidf_syntactic_exp3", # Logistic Regression with TFIDF + Syntactic
                            "mnb_bow_syntactic_exp4", # MNB with BoW + Syntactic
                            "random_forest_bow_syntactic_exp5", # Random Forest with BoW + Syntactic
                            "gradient_boosting_tfidf_syntactic_exp6" # <-- ADDED Experiment 6
                            ],
                        help="Model type to train/evaluate.")
    # ---------------------------
    # --- Baseline Model Hyperparameters ---
    parser.add_argument("--kernel", default="linear", choices=["linear", "rbf", "poly"], help="SVM kernel type")
    parser.add_argument("--C", type=float, default=1.0, help="SVM/Logistic Regression regularization parameter C")
    parser.add_argument("--max_features", type=int, default=10000, help="Max features for TF-IDF/BoW")
    parser.add_argument("--alpha", type=float, default=1.0, help="Smoothing parameter alpha for MNB")
    # --- Random Forest / Gradient Boosting Hyperparameters ---
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees for Random Forest (Exp 5) / Gradient Boosting (Exp 6)")
    parser.add_argument("--max_depth", type=int, default=None, help="Max depth for trees (Exp 5/6, None for no limit)")
    # Note: GradientBoostingClassifier has its own learning_rate parameter, which reuses the --learning_rate argument above.
    # ------------------------------------------------------
    # --- Cross-Evaluation ---
    parser.add_argument("--cross_evaluate", action="store_true", help="Perform cross-dataset evaluation (check implementation compatibility)")
    # --- Neural Model Selection (If using neural models not covered here) ---
    parser.add_argument("--baseline_model_name", type=str, default=None,
                        choices=list(HF_MODEL_IDENTIFIERS.keys()) if HF_MODEL_IDENTIFIERS else [],
                        help="Specify a baseline transformer model if using a neural model type")
    return parser.parse_args()