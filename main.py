# Modify file: IS567FP/main.py
# Add logic to handle experiment 7
import logging

from config import parse_args, DEVICE
from data.preprocessor import TextPreprocessor
from models.baseline_trainer import BaselineTrainer # Use the unified trainer
# <<< Import Experiment 7 class >>>
from models.cross_eval_syntactic_experiment_7 import CrossEvalSyntacticExperiment7
# ---------------------------------
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
    preprocessor = TextPreprocessor(db_handler) # Pass db_handler, sample_size handled inside pipeline now
    # Call the pipeline method which now handles sampling internally
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
    torch.backends.cudnn.benchmark = False # Default False for baselines unless needed
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
        # <<< ADDED Experiment 7 to list handled by BaselineTrainer >>>
        # Note: This is a slight simplification. Exp7 orchestrates multiple models.
        # If BaselineTrainer cannot handle this structure, a separate elif block is needed.
        # For now, let's assume BaselineTrainer can be adapted or Exp7 handles its internal logic.
        baseline_model_types_handled_by_trainer = [
            "svm",
            "logistic_tfidf",
            "mnb_bow",
            "svm_syntactic_exp1",
            "svm_bow_syntactic_exp2",
            "logistic_tfidf_syntactic_exp3",
            "mnb_bow_syntactic_exp4",
            "random_forest_bow_syntactic_exp5",
            "gradient_boosting_tfidf_syntactic_exp6",
            # "cross_eval_syntactic_exp7" # Add here IF BaselineTrainer handles it
            ]
        # ----------------------------------------------------

        # Use the unified BaselineTrainer for most experiments
        if args.model_type in baseline_model_types_handled_by_trainer:
            logger.info(f"Initializing BaselineTrainer for model: {args.model_type}, dataset: {args.dataset}")
            trainer = BaselineTrainer(
                model_type=args.model_type,
                dataset_name=args.dataset,
                args=args
            )
            logger.info(f"Starting training process...")
            results = trainer.run_training()
            if results:
                 logger.info(f"Training finished. Results: {results}")
            else:
                 logger.error("Training run failed or produced no results.")

        # <<< Separate handler for Experiment 7 >>>
        elif args.model_type == "cross_eval_syntactic_exp7":
             logger.info(f"Initializing CrossEvalSyntacticExperiment7 for dataset: {args.dataset}")
             # Experiment 7 class takes args and runs its specific comparisons
             exp7_runner = CrossEvalSyntacticExperiment7(args)
             logger.info(f"Starting Experiment 7 run...")
             results = exp7_runner.run_experiment()
             if results:
                  logger.info(f"Experiment 7 finished. Overall Results Summary: {results}")
                  # Log detailed results (assuming 'results' is a dictionary)
                  for config, metrics in results.items():
                      logger.info(f"  - Config: {config}, Metrics: {metrics}")
             else:
                  logger.error("Experiment 7 run failed or produced no results.")
        # ----------------------------------------

        # Add handling for other potential model types (e.g., neural) if needed
        # elif args.model_type == "neural":
        #      logger.warning("Neural network training path not fully implemented.")
        else:
            logger.error(f"Unknown or unsupported model type for training: '{args.model_type}'")


    elif args.mode == "evaluate":
        logger.info(f"Starting evaluation for model type: {args.model_type} on dataset: {args.dataset}")

        # <<< ADDED Experiment 7 to evaluation list >>>
        models_loading_internally = [
             'logistic_tfidf_syntactic_exp3',
             'mnb_bow_syntactic_exp4',
             'random_forest_bow_syntactic_exp5',
             'gradient_boosting_tfidf_syntactic_exp6',
             'cross_eval_syntactic_exp7' # Experiment 7 also loads its own data
             ]
        # ----------------------------------------------------------

        if args.model_type in models_loading_internally:
            logger.info(f"{args.model_type} handles data loading internally for evaluation.")
            # For Exp7, evaluation might be integrated within its `run_experiment`
            # Or we might need a separate `evaluate_experiment` method if called separately.
            # Let's assume for now the `train` mode runs the full Exp7 including eval logging.
            # If evaluate mode needs to re-run Exp7 eval, need specific logic here.
            logger.warning(f"Evaluation mode for {args.model_type} might require specific implementation or rely on 'train' mode output.")
            # Example: Reload and evaluate if needed
            if args.model_type == 'cross_eval_syntactic_exp7':
                 logger.info("Evaluate mode for Exp7: Attempting to load and re-evaluate (requires models to be saved).")
                 try:
                     exp7_runner = CrossEvalSyntacticExperiment7(args)
                     # TODO: Add a method to CrossEvalSyntacticExperiment7 like `evaluate_on_test()`
                     # which loads models and evaluates on the test set.
                     # For now, just logging a message.
                     logger.warning("Exp7 evaluation logic in 'evaluate' mode not fully implemented. Run in 'train' mode.")
                 except Exception as e:
                     logger.error(f"Failed to set up Exp7 evaluation: {e}")
            else:
                 # For Exp 3, 4, 5, 6 - Use BaselineTrainer as before
                trainer = BaselineTrainer(
                    model_type=args.model_type,
                    dataset_name=args.dataset,
                    args=args
                )
                eval_metrics = trainer.run_evaluation(None) # Data loaded internally
                if eval_metrics: logger.info(f"Evaluation metrics on test set: {eval_metrics}")
                else: logger.error("Evaluation run failed.")
        else:
             # For other models handled by BaselineTrainer (SVM variants, TFIDF, MNB)
            trainer = BaselineTrainer(
                model_type=args.model_type,
                dataset_name=args.dataset,
                args=args
            )
            # Attempt to load explicit test data if needed by the model
            test_data = None
            _, _, test_data = trainer._load_data() # Trainer handles loading strategy
            if test_data is None or test_data.empty:
                 logger.warning(f"Could not load explicit test data for {args.dataset} for {args.model_type}.")

            # run_evaluation uses loaded data or loads internally if needed
            eval_metrics = trainer.run_evaluation(test_data)
            if eval_metrics: logger.info(f"Evaluation metrics on test set: {eval_metrics}")
            else: logger.error("Evaluation run failed or produced no metrics.")


    elif args.mode == "predict":
        logger.warning("Prediction mode not fully implemented yet.")
        # Add prediction logic here if needed
        pass
    else:
        logger.error(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()