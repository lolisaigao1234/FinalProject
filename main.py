# IS567FP/main.py
import logging

from config import parse_args, DEVICE
from data.preprocessor import TextPreprocessor
# Import the new unified trainer and base helpers
from models.baseline_trainer import BaselineTrainer
from utils.database import DatabaseHandler
from utils.common import torch # Keep torch if used for CUDA info

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


def printing_cuda_info():
    """Checks and logs CUDA availability, GPU information, and relevant versions."""
    # (Keep this function as is)
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
    logger.info(f"Preprocessing {args.dataset} dataset. Target sample size: {args.sample_size or 'Full'}.")
    db_handler = DatabaseHandler()
    # Pass sample_size=None to TextPreprocessor init, pipeline handles sampling logic
    preprocessor = TextPreprocessor(db_handler, sample_size=None)
    # Pass total sample size to the pipeline method
    preprocessor.preprocess_dataset_pipeline(
        dataset_name=args.dataset,
        total_sample_size=args.sample_size, # Pass the arg directly
        force_reprocess=args.force_reprocess
    )
    logger.info("Preprocessing pipeline complete.")


def main():
    """Main entry point."""
    printing_cuda_info()
    args = parse_args()

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    logger.info(f"Running in {args.mode} mode on {args.dataset} dataset")
    logger.info(f"Using device: {DEVICE}") # DEVICE from config
    logger.info(f"Selected model type: {args.model_type}")

    if args.mode == "preprocess":
        preprocess_data(args)
    elif args.mode == "train":
        # Use the unified BaselineTrainer for specified baseline types
        # << --- Updated condition --- >>
        if args.model_type in ["svm", "logistic_tfidf", "mnb_bow", "svm_syntactic_exp1"]:
            logger.info(f"Initializing BaselineTrainer for model: {args.model_type}, dataset: {args.dataset}")
            trainer = BaselineTrainer(
                model_type=args.model_type, # Pass the selected model type
                dataset_name=args.dataset,
                args=args # Pass all args for hyperparameter access
            )
            logger.info(f"Starting training process...")
            results = trainer.run_training() # run_training now handles the logic based on model_type
            if results:
                 logger.info(f"Training finished. Results: {results}")
            else:
                 logger.error("Training run failed.")

            # Cross-evaluation logic (check if needed and how it interacts with specific model types)
            # The original cross-eval was tied to 'svm' type. Adapt if needed for 'svm_syntactic_exp1'.
            if args.model_type == 'svm' and hasattr(args, 'cross_evaluate') and args.cross_evaluate:
                logger.warning("Cross-evaluation logic might need adjustment for specific SVM experiments.")
                # trainer._run_cross_evaluation(args.dataset) # Assuming this method exists and handles loading correctly

        # << --- End Update --- >>
        else:
            logger.warning(f"Neural network training path or unknown model type '{args.model_type}' not explicitly handled by BaselineTrainer's main training logic.")
            # Add separate logic for neural models if needed

    elif args.mode == "evaluate":
        logger.info(f"Starting evaluation for model type: {args.model_type} on dataset: {args.dataset}")
        # Initialize trainer - evaluation logic is now within BaselineTrainer
        trainer = BaselineTrainer(
            model_type=args.model_type,
            dataset_name=args.dataset,
            args=args
        )
        # Load test data - trainer's _load_data can handle this
        _, _, test_data = trainer._load_data() # Load data appropriate for the model type

        if test_data is not None and not test_data.empty:
             # run_evaluation within the trainer handles loading the correct model and evaluating
             eval_metrics = trainer.run_evaluation(test_data) # Pass test data
             if eval_metrics:
                  logger.info(f"Evaluation metrics on test set: {eval_metrics}")
             else:
                  logger.error("Evaluation run failed or produced no metrics.")
        else:
            logger.error(f"Failed to load test data for {args.dataset} ({trainer.suffix}). Cannot evaluate.")


    elif args.mode == "predict":
        logger.warning("Prediction mode not fully implemented yet.")
        # Add prediction logic here
        pass
    else:
        logger.error(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()