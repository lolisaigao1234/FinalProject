# IS567FP/main.py
import os # Added import
from torch.utils.data import DataLoader

from config import parse_args, DEVICE, EPOCHS, SYNTACTIC_FEATURE_DIM, MODEL_NAME, MODELS_DIR # Added MODELS_DIR
from data.preprocessor import TextPreprocessor
from data.data_loader import DatasetLoader # Added import
from models.NeuroTrainer import ModelTrainer, NLIDataset
from models.SVMTrainer import SVMTrainer, SVMWithBagOfWords, SVMWithSyntax, SVMWithBothFeatures, _evaluate_model, clean_dataset # Import helpers
from models.logistic_tf_idf_baseline import LogisticTFIDFBaseline, LogisticRegressionTrainer # Import new baseline
from utils.common import logging, torch
from utils.database import DatabaseHandler
from models.baseline_transformer import BaselineTransformerNLI
from models.transformer_model import BERTWithSyntacticAttention  # Your existing model
from config import HF_MODEL_IDENTIFIERS, NUM_CLASSES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


def printing_cuda_info():
    """
    Checks and logs CUDA availability, GPU information, and relevant versions.
    """
    cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA Available: {cuda_available}")

    if cuda_available:
        num_gpus = torch.cuda.device_count()
        logger.info(f"Number of GPUs Available: {num_gpus}")

        if num_gpus > 0:
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"Using GPU: {gpu_name}")
            current_device = torch.cuda.current_device()
            logger.info(f"Current CUDA Device Index: {current_device}")
        else:
            logger.warning("No CUDA-enabled GPUs found.")

        cuda_version = torch.version.cuda
        cudnn_version = torch.backends.cudnn.version()
        logger.info(f"torch.version.cuda: {cuda_version}")
        logger.info(f"torch.backends.cudnn.version(): {cudnn_version}")
    else:
        logger.info("CUDA is not available. Using CPU for computations.")


def preprocess_data(args):
    """Runs the preprocessing pipeline based on arguments."""
    logger.info(f"Preprocessing {args.dataset} dataset with total sample size {args.sample_size}")
    db_handler = DatabaseHandler()
    # TextPreprocessor handles both SVM/Logistic features (lexical/syntactic) and intermediate steps
    # The FeatureExtractor called by the preprocessor pipeline creates the features needed by SVMs.
    # TF-IDF baseline does *not* need these specific features, but it *does* need the intermediate
    # 'pairs' and 'sentences' files which TextPreprocessor creates.
    preprocessor = TextPreprocessor(db_handler) # Pass sample size to pipeline method instead
    preprocessor.preprocess_dataset_pipeline(
        dataset_name=args.dataset,
        total_sample_size=args.sample_size, # Pass the total size
        force_reprocess=args.force_reprocess
    )
    logger.info("Preprocessing pipeline complete (intermediate data created)")


def main():
    """Main entry point with performance monitoring"""

    printing_cuda_info()
    args = parse_args()

    # Set deterministic algorithms for reproducibility if needed (can slow down training)
    # torch.backends.cudnn.benchmark = not args.deterministic # Set benchmark based on deterministic flag
    # torch.backends.cudnn.deterministic = args.deterministic # Add a --deterministic flag to args?
    torch.backends.cudnn.benchmark = True # Keep benchmark enabled for speed
    torch.backends.cudnn.deterministic = False

    logger.info(f"Running in {args.mode} mode on {args.dataset} dataset")
    logger.info(f"Using device: {DEVICE}")
    logger.info(f"Selected model type: {args.model_type}")

    if args.mode == "preprocess":
        preprocess_data(args) # Pass args to preprocessor
    elif args.mode == "train":
        if args.model_type == "svm":
            # --- SVM Training ---
            logger.info(f"Starting SVM training with Kernel: {args.kernel}, C: {args.C}")
            svm_trainer = SVMTrainer() # Uses precomputed lexical/syntactic features
            svm_trainer.run_training(args)
            # --------------------
        elif args.model_type == "logistic_tfidf":
            # --- Logistic Regression + TF-IDF Training ---
            logger.info(f"Starting Logistic Regression + TF-IDF training with C: {args.C}, Max Features: {args.max_features}")
            # This trainer needs access to intermediate 'pairs' and 'sentences' data
            logistic_trainer = LogisticRegressionTrainer()
            logistic_trainer.run_training(args)
            # -----------------------------------------
        else:
            logger.error(f"Unknown model type for training: {args.model_type}")
            return

    elif args.mode == "evaluate":
        logger.warning("Evaluation mode not fully implemented yet.")
        # Add evaluation logic here - needs to load a trained model and test data
        if args.model_type == "svm":
             # Load SVM models and evaluate on test set (similar to cross_evaluate logic in SVMTrainer)
             logger.info("Evaluating SVM models...")
             svm_trainer = SVMTrainer()
             # Need to load test data (lexical/syntactic features)
             test_data = svm_trainer._load_datasets(args.dataset, split='test') # Adapt internal method or reimplement
             if test_data is not None:
                  # Assuming test_data is a tuple (test_df, None) if validation split is not used in _load_datasets
                  test_df, _ = test_data if isinstance(test_data, tuple) else (test_data, None)

                  if test_df is not None and not test_df.empty:
                       svm_results = {}
                       for model_cls, name in [(SVMWithBagOfWords, "SVMWithBagOfWords"),
                                            (SVMWithSyntax, "SVMWithSyntax"),
                                            (SVMWithBothFeatures, "SVMWithBothFeatures")]:
                            model_path = os.path.join(svm_trainer.save_dir, f"{name}.joblib")
                            if os.path.exists(model_path):
                                 logger.info(f"Loading {name} from {model_path}")
                                 model = model_cls.load(model_path) # Use appropriate FeatureExtractor in load if needed
                                 test_df_clean, y_test = clean_dataset(test_df.copy()) # Clean copy
                                 X_test = model.extract_features(test_df_clean)
                                 eval_time, metrics = _evaluate_model(model, X_test, y_test)
                                 logger.info(f"{name} Test Results - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
                                 svm_results[name] = metrics
                            else:
                                 logger.warning(f"Model file not found, cannot evaluate: {model_path}")
                  else:
                       logger.error(f"Could not load test feature data for {args.dataset} evaluation.")
             else:
                  logger.error(f"Could not load test feature data for {args.dataset} evaluation.")

        elif args.model_type == "logistic_tfidf":
            logger.info("Evaluating Logistic Regression TF-IDF model...")
            logistic_trainer = LogisticRegressionTrainer()
            model_dir = logistic_trainer.save_dir
            try:
                model = LogisticTFIDFBaseline.load(model_dir)
                # Load RAW test text data
                suffix = f"sample{args.sample_size}" if args.sample_size else "full"
                test_data = LogisticTFIDFBaseline.load_raw_text_data(args.dataset, 'test', suffix, logistic_trainer.db_handler)
                if test_data is not None and not test_data.empty:
                     test_data_clean, y_test = clean_dataset(test_data)
                     X_test = model.extract_features(test_data_clean) # Use loaded vectorizer
                     eval_time, metrics = _evaluate_model(model, X_test, y_test)
                     logger.info(f"Logistic TF-IDF Test Results - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
                else:
                     logger.error(f"Could not load test raw text data for {args.dataset} evaluation.")
            except FileNotFoundError:
                logger.error(f"Could not find saved Logistic Regression model/vectorizer in {model_dir}")
            except Exception as e:
                 logger.error(f"Error during Logistic Regression evaluation: {e}")

        else:
            logger.error(f"Unknown model type for evaluation: {args.model_type}")

    elif args.mode == "predict":
        logger.warning("Prediction mode not implemented yet.")
        # Add prediction logic here - needs loading model, data, and running predict
        pass
    else:
        logger.error(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()