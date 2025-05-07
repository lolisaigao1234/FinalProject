# Modify file: IS567FP/config.py
# Update args for 'predict' mode to use test files

import argparse
import logging
import os
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

import torch

# Project paths
PROJECT_ROOT = Path(__file__).parent.absolute()
DATA_DIR = os.path.join(PROJECT_ROOT, "data_files")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
MODELS_DIR = os.path.join(PROJECT_ROOT, "saved_models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")  # Added for predictions

# # Create directories
for directory in [DATA_DIR, CACHE_DIR, MODELS_DIR, OUTPUT_DIR]:  # Added OUTPUT_DIR
    os.makedirs(directory, exist_ok=True)

# Dataset settings for HuggingFace datasets
DATASETS = {
    "SNLI": {
        "hf_name": "stanfordnlp/snli",
        "splits": {"train": "train", "dev": "validation", "test": "test"},
        "text_cols": {"premise": "premise", "hypothesis": "hypothesis"}
    },
    "MNLI": {
        "hf_name": "nyu-mll/multi_nli",
        "splits": {"train": "train", "dev": "validation_matched", "test": "validation_mismatched"},
        "text_cols": {"premise": "premise", "hypothesis": "hypothesis"}
    },
    "ANLI": {
        "hf_name": "facebook/anli",
        # Change this line:
        "splits": {"train": "train", "dev": "dev", "test": "test"},  # Use base names
        "text_cols": {"premise": "premise", "hypothesis": "hypothesis"}
    }
}

# Performance-critical settings (Defaults, can be overridden by args)
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 64
GRAD_ACCUM_STEPS = 2
NUM_WORKERS = 4  # Adjusted default based on common local setups
USE_FP16 = False  # Default to False for broader CPU/GPU compatibility
CUDA_BENCHMARK = True
PIN_MEMORY = True
TORCH_LOGS = "+dynamo"
TORCHDYNAMO_VERBOSE = 1

# Model optimization (Defaults for relevant models)
MODEL_NAME = "bert-base-uncased"  # Still here, but less relevant for non-neural baselines
HIDDEN_SIZE = 768
NUM_CLASSES = 3
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01
EPOCHS = 5
SYNTACTIC_FEATURE_DIM = 200  # Used by feature extractor

STANZA_PROCESSORS = "tokenize,pos,lemma,depparse,constituency"
STANZA_LANG = "en"

# Compute settings
TORCH_COMPILE = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Add environment variables for HuggingFace
HF_CACHE_DIR = os.path.join(CACHE_DIR, "huggingface")
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Database settings
DB_TYPE = "parquet"
PARQUET_DIR = os.path.join(CACHE_DIR, "parquet")
os.makedirs(PARQUET_DIR, exist_ok=True)

# Baseline/Experiment Hyperparameters (Defaults)
DECISION_TREE_MAX_DEPTH = 10
DECISION_TREE_MIN_SAMPLES_SPLIT = 2
DECISION_TREE_MIN_SAMPLES_LEAF = 1
RANDOM_SEED = 42
KNN_N_NEIGHBORS = 5
KNN_WEIGHTS = "uniform"
KNN_METRIC = "cosine"
LOGISTIC_C = 1.0
LOGISTIC_MAX_ITER = 1000
MNB_ALPHA = 1.0
TFIDF_BOW_MAX_FEATURES = 10000  # Common max_features for both
RF_GB_N_ESTIMATORS = 100
RF_GB_MAX_DEPTH = None

# --- Define Model Choices based on models/__init__.py ---
try:
    # Statically define based on the previous file structure provided
    MODEL_CHOICES = [
        "baseline-1", "baseline-2", "baseline-3",
        "experiment-1", "experiment-2", "experiment-3", "experiment-4",
        "experiment-5", "experiment-6", "experiment-7", "experiment-8"
    ]
    # Define models that can be used for prediction mode in main.py
    PREDICT_MODEL_CHOICES = [
        "baseline-1", "baseline-2", "baseline-3",
        "experiment-1", "experiment-2", "experiment-3", "experiment-4",
        "experiment-5", "experiment-6"
        # Exclude 7 and 8 as requested
    ]
    DEFAULT_MODEL = "baseline-1"
except Exception as e:
    print(f"Warning: Error defining model choices: {e}. Using fallback.")
    MODEL_CHOICES = ["baseline-1", "baseline-2", "baseline-3", "experiment-1", "experiment-2", "experiment-3",
                     "experiment-4", "experiment-5", "experiment-6", "experiment-7", "experiment-8"]
    PREDICT_MODEL_CHOICES = ["baseline-1", "baseline-2", "baseline-3", "experiment-1", "experiment-2", "experiment-3",
                             "experiment-4", "experiment-5", "experiment-6"]
    DEFAULT_MODEL = "baseline-1"

# Reverse label map for prediction output
LABEL_MAP_REVERSE = {0: 'entailment', 1: 'contradiction', 2: 'neutral'}


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="NLI Pipeline")

    # --- MODES ---
    parser.add_argument(
        "--mode",
        default="preprocess",
        choices=["preprocess", "extract_features", "train", "evaluate", "predict"],  # 'predict' is now a valid mode
        help="Pipeline mode to run."
    )

    # --- General Arguments ---
    parser.add_argument("--dataset", default="SNLI", choices=["SNLI", "MNLI", "ANLI"])
    parser.add_argument("--force_reprocess", "--force", action="store_true",
                        help="Force reprocessing/recomputation of data/features/models")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Total sample size across splits. Processes full dataset if omitted.")
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"],
        help="Device for computation ('auto', 'cuda', 'mps', 'cpu')"
    )

    # --- Model Selection ---
    parser.add_argument(
        "--model_type", default=DEFAULT_MODEL, choices=MODEL_CHOICES,
        help="Model type key from models.MODEL_REGISTRY to train/evaluate/predict."
    )

    # --- Training Arguments ---
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Batch size (for neural models/feature extraction)")
    parser.add_argument("--grad_accum", type=int, default=GRAD_ACCUM_STEPS, help="Gradient accumulation steps (neural)")
    parser.add_argument("--fp16", action="store_true", default=USE_FP16,
                        help="Enable mixed precision training (neural)")
    parser.add_argument("--no_compile", action="store_false", dest="compile", default=not TORCH_COMPILE,
                        help="Disable PyTorch compilation (neural)")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Epochs (neural models)")
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE, help="Learning rate (neural/GB models)")
    parser.add_argument("--evaluate_after_train", action='store_true', default=False,
                        help="Evaluate on eval_split after training")

    # --- Evaluation Arguments ---
    parser.add_argument("--eval_split", type=str, default="test", choices=["train", "dev", "test"])
    parser.add_argument("--cross_evaluate", action="store_true",
                        help="Perform cross-dataset evaluation (used by Exp 7)")

    # --- Prediction Arguments ---
    # Removed --premise, --hypothesis
    parser.add_argument("--predict_input_dataset", type=str, default="SNLI", choices=["SNLI", "MNLI", "ANLI"],
                        help="Dataset split to use as input for prediction (e.g., predict on SNLI test set).")
    parser.add_argument("--predict_input_suffix", type=str, default="full",
                        help="Suffix of the input Parquet file (e.g., 'full', 'sampleXXX'). Should contain features.")
    parser.add_argument("--predict_model_dataset", type=str, default="SNLI", choices=["SNLI", "MNLI", "ANLI"],
                        help="Dataset the prediction model was trained on (used for loading the model).")
    parser.add_argument("--predict_model_suffix", type=str, default="full",
                        help="Suffix the prediction model was trained with (e.g., 'full', 'sampleXXX').")
    parser.add_argument("--predict_output_file", type=str, default=None,
                        help="Optional path to save prediction results (e.g., 'output/predictions.csv').")
    parser.add_argument("--predict_limit", type=int, default=None,
                        help="Optional: Limit prediction to the first N rows of the input file.")

    # --- Baseline/Experiment Hyperparameters ---
    parser.add_argument("--C", type=float, default=LOGISTIC_C, help="Regularization parameter C")
    parser.add_argument("--max_iter", type=int, default=LOGISTIC_MAX_ITER,
                        help="Max iterations for Logistic Regression.")
    parser.add_argument("--max_features", type=int, default=TFIDF_BOW_MAX_FEATURES, help="Max features for TF-IDF/BoW")
    parser.add_argument("--alpha", type=float, default=MNB_ALPHA, help="Smoothing parameter alpha for MNB")
    parser.add_argument("--n_estimators", type=int, default=RF_GB_N_ESTIMATORS, help="Number of trees for RF/GB")
    parser.add_argument("--max_depth", type=int, default=RF_GB_MAX_DEPTH,
                        help="Max depth for trees (None for no limit)")
    parser.add_argument("--min_samples_split", type=int, default=DECISION_TREE_MIN_SAMPLES_SPLIT,
                        help="Min samples to split node (Tree/RF)")
    parser.add_argument("--min_samples_leaf", type=int, default=DECISION_TREE_MIN_SAMPLES_LEAF,
                        help="Min samples per leaf node (Tree/RF)")
    parser.add_argument("--n_neighbors", type=int, default=KNN_N_NEIGHBORS, help="Number of neighbors for KNN")
    parser.add_argument("--no_scale_syntactic", action="store_false", dest="scale_syntactic", default=True,
                        help="Disable scaling of syntactic features (if model supports it)")

    parsed_args = parser.parse_args()

    # --- Post-processing / Validation ---
    if parsed_args.device == "auto":
        if torch.cuda.is_available():
            parsed_args.device = "cuda"
        elif torch.backends.mps.is_available():
            parsed_args.device = "mps"
        else:
            parsed_args.device = "cpu"

    # Default output filename if not provided
    if parsed_args.mode == 'predict' and not parsed_args.predict_output_file:
        parsed_args.predict_output_file = os.path.join(
            OUTPUT_DIR,
            f"predictions_{parsed_args.model_type}_{parsed_args.predict_input_dataset}_on_{parsed_args.predict_model_dataset}_{parsed_args.predict_model_suffix}.csv"
        )
        logger.info(f"Defaulting prediction output file to: {parsed_args.predict_output_file}")

    return parsed_args
