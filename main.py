# IS567FP/main.py
# Add handling for 'mnb_bow' model type
import glob
import os # Added import

import pandas as pd
from torch.utils.data import DataLoader

from config import parse_args, DEVICE
from data.preprocessor import TextPreprocessor
from models.svm_bow_baseline import SVMTrainer, SVMWithBagOfWords, SVMWithSyntax, SVMWithBothFeatures, _evaluate_model, clean_dataset # Import helpers
from models.logistic_tf_idf_baseline import LogisticTFIDFBaseline, LogisticRegressionTrainer # Import new baseline
from models.multinomial_naive_bayes_bow_baseline import MultinomialNaiveBayesBaseline, MultinomialNaiveBayesTrainer # Import MNB
from utils.common import logging, torch
from utils.database import DatabaseHandler

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
    # TextPreprocessor handles both SVM/Logistic/MNB features (lexical/syntactic) and intermediate steps
    # The FeatureExtractor called by the preprocessor pipeline creates the features needed by SVMs.
    # TF-IDF/BoW baselines do *not* need these specific features, but they *do* need the intermediate
    # 'pairs' and 'sentences' files which TextPreprocessor creates.
    # Pass the total sample size, TextPreprocessor will calculate per-split sizes
    preprocessor = TextPreprocessor(db_handler)
    preprocessor.preprocess_dataset_pipeline(
        dataset_name=args.dataset,
        total_sample_size=args.sample_size, # Pass the total size
        # train_ratio=0.8, # Default ratio if not needed elsewhere, preprocessor uses internal defaults
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
        elif args.model_type == "mnb_bow": # <<< ADDED MNB BoW Training >>>
            # --- Multinomial Naive Bayes + BoW Training ---
            logger.info(f"Starting MNB + BoW training with Max Features: {args.max_features}")
            # This trainer also needs raw text data (handled internally)
            mnb_trainer = MultinomialNaiveBayesTrainer()
            mnb_trainer.run_training(args)
            # --------------------------------------------
        else: # Assuming 'neural' or unknown
            logger.warning(f"Neural network training path not fully implemented in this version of main.py.")
            # Add your neural network training call here if applicable
            # Example:
            # logger.info("Starting Neural Network training...")
            # neural_trainer = ModelTrainer(...) # Initialize appropriately
            # neural_trainer.train(...)
            if args.model_type != "neural": # Error only if not 'neural'
                 logger.error(f"Unknown model type for training: {args.model_type}")
            return

    elif args.mode == "evaluate":
        logger.info(f"Starting evaluation for model type: {args.model_type} on dataset: {args.dataset}")
        suffix = f"sample{args.sample_size}" if args.sample_size else "full"

        if args.model_type == "svm":
             logger.info("Evaluating SVM models...")
             svm_trainer = SVMTrainer()
             svm_save_dir = svm_trainer.save_dir # Get the save directory
             # Load test data (lexical/syntactic features)
             # The loading function needs the correct pattern now
             try:
                 test_feature_pattern = DatabaseHandler._get_final_features_pattern(args.dataset, 'test') # Get pattern
                 test_files = glob.glob(test_feature_pattern)
                 if not test_files:
                     logger.error(f"No final SVM feature files found for {args.dataset}/test matching pattern: {test_feature_pattern}")
                     return
                 # Load the first file found (assuming one file per dataset/split/suffix)
                 test_data_df = pd.read_parquet(test_files[0])
                 logger.info(f"Loaded test features from {test_files[0]}")

                 if test_data_df is not None and not test_data_df.empty:
                      test_df_clean, y_test = clean_dataset(test_data_df.copy()) # Clean copy
                      if test_df_clean is not None and y_test is not None and len(y_test) > 0:
                           svm_results = {}
                           for model_cls, name in [(SVMWithBagOfWords, "SVMWithBagOfWords"),
                                                (SVMWithSyntax, "SVMWithSyntax"),
                                                (SVMWithBothFeatures, "SVMWithBothFeatures")]:
                                model_path = os.path.join(svm_save_dir, f"{name}.joblib")
                                if os.path.exists(model_path):
                                     logger.info(f"Loading {name} from {model_path}")
                                     # Pass the appropriate extractor when loading
                                     if name == "SVMWithBagOfWords": extractor = LexicalFeatureExtractor()
                                     elif name == "SVMWithSyntax": extractor = SyntacticFeatureExtractor()
                                     else: extractor = CombinedFeatureExtractor()
                                     model = model_cls.load(model_path, feature_extractor=extractor) # Pass extractor
                                     X_test = model.extract_features(test_df_clean) # Use loaded model's extractor logic
                                     eval_time, metrics = _evaluate_model(model, X_test, y_test)
                                     logger.info(f"{name} Test Results - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
                                     svm_results[name] = metrics
                                else:
                                     logger.warning(f"Model file not found, cannot evaluate: {model_path}")
                      else:
                           logger.error("SVM Test data became invalid after cleaning.")
                 else:
                      logger.error(f"Could not load valid test feature data for {args.dataset} SVM evaluation.")
             except FileNotFoundError:
                 logger.error(f"Could not find test feature files for {args.dataset} SVM evaluation.")
             except Exception as e:
                 logger.error(f"Error during SVM evaluation: {e}", exc_info=True)

        elif args.model_type == "logistic_tfidf":
            logger.info("Evaluating Logistic Regression TF-IDF model...")
            logistic_trainer = LogisticRegressionTrainer()
            model_dir = logistic_trainer.save_dir
            try:
                # Load model (implicitly loads vectorizer via LogisticTFIDFBaseline.load)
                model = LogisticTFIDFBaseline.load(model_dir)
                # Load RAW test text data
                test_data_df = LogisticTFIDFBaseline.load_raw_text_data(args.dataset, 'test', suffix, logistic_trainer.db_handler)
                if test_data_df is not None and not test_data_df.empty:
                     test_data_clean, y_test = clean_dataset(test_data_df)
                     if test_data_clean is not None and y_test is not None and len(y_test) > 0:
                         # The model's extract_features method uses the loaded vectorizer
                         X_test = model.extract_features(test_data_clean)
                         eval_time, metrics = _evaluate_model(model, X_test, y_test)
                         logger.info(f"Logistic TF-IDF Test Results - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
                     else:
                         logger.error("Logistic TF-IDF Test data became invalid after cleaning.")
                else:
                     logger.error(f"Could not load test raw text data for {args.dataset} Logistic TF-IDF evaluation.")
            except FileNotFoundError:
                logger.error(f"Could not find saved Logistic Regression model/vectorizer in {model_dir}")
            except Exception as e:
                 logger.error(f"Error during Logistic Regression evaluation: {e}", exc_info=True)

        elif args.model_type == "mnb_bow": # <<< ADDED MNB BoW Evaluation >>>
            logger.info("Evaluating Multinomial Naive Bayes (BoW) model...")
            mnb_trainer = MultinomialNaiveBayesTrainer()
            model_dir = mnb_trainer.save_dir
            try:
                # Load model (implicitly loads vectorizer via load method)
                model = MultinomialNaiveBayesBaseline.load(model_dir)
                # Load RAW test text data (re-use logic from logistic)
                test_data_df = LogisticTFIDFBaseline.load_raw_text_data(args.dataset, 'test', suffix, mnb_trainer.db_handler)
                if test_data_df is not None and not test_data_df.empty:
                     test_data_clean, y_test = clean_dataset(test_data_df)
                     if test_data_clean is not None and y_test is not None and len(y_test) > 0:
                         # The model's extract_features uses the loaded BoW vectorizer
                         X_test = model.extract_features(test_data_clean)
                         eval_time, metrics = _evaluate_model(model, X_test, y_test)
                         logger.info(f"MNB BoW Test Results - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
                     else:
                          logger.error("MNB BoW Test data became invalid after cleaning.")
                else:
                     logger.error(f"Could not load test raw text data for {args.dataset} MNB BoW evaluation.")
            except FileNotFoundError:
                logger.error(f"Could not find saved MNB model/vectorizer in {model_dir}")
            except Exception as e:
                 logger.error(f"Error during MNB BoW evaluation: {e}", exc_info=True)
            # --------------------------------------------------------------------

        else: # Assuming 'neural' or unknown
             logger.warning("Neural network evaluation path not fully implemented in this version of main.py.")
             # Add evaluation logic for neural models here
             if args.model_type != "neural":
                  logger.error(f"Unknown model type for evaluation: {args.model_type}")

    elif args.mode == "predict":
        logger.warning("Prediction mode not implemented yet.")
        # Add prediction logic here - needs loading model, data, and running predict
        pass
    else:
        logger.error(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()