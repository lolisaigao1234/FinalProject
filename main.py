# Modify file: IS567FP/main.py
import logging

from config import parse_args, DEVICE
from data.preprocessor import TextPreprocessor
from models.baseline_trainer import BaselineTrainer # Use the unified trainer
from utils.database import DatabaseHandler
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


def printing_cuda_info():
    """Checks and logs CUDA availability, GPU information, and relevant versions."""
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
        try:
            cuda_version = torch.version.cuda
            cudnn_version = torch.backends.cudnn.version()
            logger.info(f"torch.version.cuda: {cuda_version}")
            logger.info(f"torch.backends.cudnn.version(): {cudnn_version}")
        except AttributeError:
            logger.warning("Could not retrieve CUDA/CuDNN versions.")
    else:
        logger.info("CUDA is not available. Using CPU for computations.")


def preprocess_data(args):
    """Runs the preprocessing pipeline based on arguments."""
    logger.info(f"Preprocessing {args.dataset} dataset. Sample size: {args.sample_size or 'Full'}.")
    db_handler = DatabaseHandler()
    # Pass sample_size to preprocessor if it uses it internally during pipeline setup
    # The main logic now uses total_sample_size within preprocess_dataset_pipeline
    preprocessor = TextPreprocessor(db_handler, sample_size=args.sample_size) # Pass sample_size for potential internal use
    # Call the pipeline method which now handles sampling internally based on total_sample_size
    preprocessor.preprocess_dataset_pipeline(
        dataset_name=args.dataset,
        total_sample_size=args.sample_size, # Pass the total size argument
        force_reprocess=args.force_reprocess
    )
    logger.info("Preprocessing pipeline complete.")


def main():
    """Main entry point."""
    printing_cuda_info()
    args = parse_args()

    # Set benchmark/deterministic (optional, adjust based on needs)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False

    logger.info(f"Running in {args.mode} mode on {args.dataset} dataset")
    logger.info(f"Using device: {DEVICE}")
    logger.info(f"Selected model type: {args.model_type}")
    if args.sample_size:
        logger.info(f"Using sample size: {args.sample_size}")
    else:
        logger.info("Processing full dataset (no sample size specified).")


    if args.mode == "preprocess":
        preprocess_data(args)

    elif args.mode == "train":
        # Define the list of all baseline/experiment model types handled by BaselineTrainer
        # <<< ADDED gradient_boosting_tfidf_syntactic_exp6 >>>
        baseline_model_types = [
            "svm", # Handles BoW, Syntax, Combined SVM variants internally
            "logistic_tfidf",
            "mnb_bow",
            "svm_syntactic_exp1",
            "svm_bow_syntactic_exp2",
            "logistic_tfidf_syntactic_exp3",
            "mnb_bow_syntactic_exp4",
            "random_forest_bow_syntactic_exp5",
            "gradient_boosting_tfidf_syntactic_exp6" # Added Exp 6
            ]
        # ----------------------------------------------------

        # Use the unified BaselineTrainer
        if args.model_type in baseline_model_types:
            logger.info(f"Initializing BaselineTrainer for model: {args.model_type}, dataset: {args.dataset}")
            # Pass all arguments to the trainer
            trainer = BaselineTrainer(
                model_type=args.model_type,
                dataset_name=args.dataset,
                args=args # Pass the full args object
            )
            logger.info(f"Starting training process...")
            # run_training handles the specific logic based on model_type
            results = trainer.run_training()
            if results:
                 logger.info(f"Training finished. Results: {results}")
            else:
                 logger.error("Training run failed or produced no results.")

            # Cross-evaluation logic (remains the same)
            if hasattr(args, 'cross_evaluate') and args.cross_evaluate:
                 logger.info("Cross-evaluation requested. Note: Ensure models are trained on the intended source dataset.")
                 logger.warning("Cross-evaluation logic might need specific implementation/mode.")

        # Add handling for other potential model types (e.g., neural) if needed
        # elif args.model_type == "neural":
        #      logger.warning("Neural network training path not fully implemented.")
        else:
            logger.error(f"Unknown or unsupported model type for training: '{args.model_type}'")


    elif args.mode == "evaluate":
        logger.info(f"Starting evaluation for model type: {args.model_type} on dataset: {args.dataset}")
        # Initialize trainer - evaluation logic is handled within BaselineTrainer
        trainer = BaselineTrainer(
            model_type=args.model_type,
            dataset_name=args.dataset,
            args=args
        )

        # Load test data - Trainer's run_evaluation method handles data loading strategy
        # For models like Exp3, Exp4, Exp5, Exp6 eval_data can be None as they load internally
        # For others, the trainer's _load_data is used by run_evaluation implicitly.
        # We call run_evaluation, it figures out if data is needed based on model type.
        # For simplicity, we attempt to load test data here for models that DON'T load internally.
        # Models that load internally will ignore the passed data if it's not None.
        test_data = None
        # <<< ADDED gradient_boosting_tfidf_syntactic_exp6 to list >>>
        models_loading_internally = [
             'logistic_tfidf_syntactic_exp3',
             'mnb_bow_syntactic_exp4',
             'random_forest_bow_syntactic_exp5',
             'gradient_boosting_tfidf_syntactic_exp6'
             ]
        # ----------------------------------------------------------
        if args.model_type not in models_loading_internally:
             _, _, test_data = trainer._load_data() # Attempt to load test data
             if test_data is None or test_data.empty:
                  logger.warning(f"Failed to load explicit test data for {args.dataset}. Model might load its own if applicable.")
                  # Do not return yet, let run_evaluation handle internal loading if possible

        # run_evaluation within the trainer handles loading the correct model and evaluating
        eval_metrics = trainer.run_evaluation(test_data) # Pass loaded test data (or None)
        if eval_metrics:
              logger.info(f"Evaluation metrics on test set: {eval_metrics}")
        else:
              logger.error("Evaluation run failed or produced no metrics.")


    elif args.mode == "predict":
        logger.warning("Prediction mode not fully implemented yet.")
        # Add prediction logic here if needed
        pass
    else:
        logger.error(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()