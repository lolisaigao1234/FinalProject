# Modified file: IS567FP/main.py
# Reflects new experiment names and corrected class names for Exp 7/8

import logging
import torch

# Assuming config.py defines parse_args() and DEVICE
from config import parse_args, DEVICE
from data.preprocessor import TextPreprocessor
from models.baseline_trainer import BaselineTrainer # Use the unified trainer
# --- Import corrected Experiment 7 & 8 classes ---
# Make sure these class names match exactly what's defined in the files
try:
    from models.cross_eval_syntactic_experiment_7 import CrossEvalSyntacticExperiment7
except ImportError:
    logging.error("Failed to import SyntacticFeatureEvaluator from models.cross_eval_syntactic_experiment_7")
    # Define a dummy class to avoid crashing if the file is missing/incorrect
    class CrossEvalSyntacticExperiment7: pass
try:
    from models.cross_validate_syntactic_experiment_8 import CrossValidateSyntacticExperiment8
except ImportError:
    logging.error("Failed to import CrossValidator from models.cross_validate_syntactic_experiment_8")
    # Define a dummy class to avoid crashing if the file is missing/incorrect
    class CrossValidateSyntacticExperiment8: pass
# --------------------------------------------------
from utils.database import DatabaseHandler


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


def preprocess_data(args):
    """Runs the preprocessing pipeline based on arguments."""
    logger.info(f"Preprocessing {args.dataset} dataset. Sample size: {args.sample_size or 'Full'}.")
    db_handler = DatabaseHandler()
    preprocessor = TextPreprocessor(db_handler)
    preprocessor.preprocess_dataset_pipeline(
        dataset_name=args.dataset,
        total_sample_size=args.sample_size,
        force_reprocess=args.force_reprocess
    )
    logger.info("Preprocessing pipeline complete.")


def main():
    """Main entry point."""
    args = parse_args()

    # Optional: Set benchmark/deterministic based on needs
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False # Set True for full reproducibility if needed

    logger.info(f"Running in {args.mode} mode on {args.dataset} dataset")
    logger.info(f"Using device: {DEVICE}") # Assuming DEVICE is correctly defined in config
    logger.info(f"Selected model type / experiment key: {args.model_type}") # Updated terminology
    if args.sample_size:
        logger.info(f"Using sample size: {args.sample_size}")
    else:
        logger.info("Processing full dataset (no sample size specified).")

    if args.mode == "preprocess":
        preprocess_data(args)

    elif args.mode == "train":
        # --- List of models/experiments handled by the standard BaselineTrainer ---
        # Uses the NEW keys from MODEL_REGISTRY
        baseline_model_types_handled_by_trainer = [
            "baseline-1", # New DT + BoW
            "baseline-2", # Logistic Regression + TF-IDF
            "baseline-3", # Multinomial NB + BoW
            "experiment-1", # New DT + Syntactic
            "experiment-2", # New KNN + BoW + Syntactic
            "experiment-3", # Logistic Regression + TF-IDF + Syntactic
            "experiment-4", # Multinomial NB + BoW + Syntactic
            "experiment-5", # Random Forest + BoW + Syntactic
            "experiment-6", # Gradient Boosting + TF-IDF + Syntactic
        ]
        # --- Special experiment runners ---
        # Uses the NEW keys and CORRECTED class names
        special_experiment_runners = {
            "experiment-7": CrossEvalSyntacticExperiment7, # CORRECTED Class Name
            "experiment-8": CrossValidateSyntacticExperiment8           # CORRECTED Class Name
        }
        # --------------------------------------------------------------------

        # Use the unified BaselineTrainer for most experiments
        if args.model_type in baseline_model_types_handled_by_trainer:
            logger.info(f"Initializing BaselineTrainer for experiment: {args.model_type}, dataset: {args.dataset}")
            try:
                trainer = BaselineTrainer(
                    model_type=args.model_type, # BaselineTrainer uses this key to get class from registry
                    dataset_name=args.dataset,
                    args=args # Pass other args like sample_size if needed by trainer/model
                )
                logger.info(f"Starting training process for {args.model_type}...")
                results = trainer.run_training()
                if results:
                    logger.info(f"Training for {args.model_type} finished. Results: {results}")
                else:
                    logger.error(f"Training run for {args.model_type} failed or produced no results.")
            except Exception as e:
                 logger.error(f"Failed to initialize or run BaselineTrainer for {args.model_type}: {e}", exc_info=True)


        # Handle Special Experiments (Exp7, Exp8)
        elif args.model_type in special_experiment_runners:
            logger.info(f"Initializing special experiment runner for: {args.model_type}, dataset: {args.dataset}")
            try:
                ExperimentRunnerClass = special_experiment_runners[args.model_type]
                # Check if the class is a dummy placeholder due to import error
                if ExperimentRunnerClass.__name__ in ['SyntacticFeatureEvaluator', 'CrossValidator'] and not hasattr(ExperimentRunnerClass, 'run_experiment'):
                     logger.error(f"Cannot run {args.model_type} because its class failed to import correctly.")
                else:
                    experiment_runner = ExperimentRunnerClass(args) # Assuming constructor takes args
                    logger.info(f"Starting {args.model_type} run...")
                    results = experiment_runner.run_experiment() # Assuming this method exists
                    if results:
                        logger.info(f"{args.model_type} finished. Overall Results Summary:")
                        # Log detailed results (assuming 'results' is a dictionary or similar)
                        if isinstance(results, dict):
                            for config, metrics in results.items():
                                if isinstance(metrics, dict):
                                    metrics_str = ", ".join(
                                        [f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()])
                                    logger.info(f"  - Config: {config}, Metrics: {metrics_str}")
                                else:
                                    logger.info(f"  - Config: {config}, Result: {metrics}")
                        else:
                            # Log results directly if not a dictionary (e.g., path to results file)
                            logger.info(f"  Results: {results}")
                    else:
                        logger.error(f"{args.model_type} run failed or produced no results.")
            except Exception as e:
                logger.error(f"Failed to initialize or run special experiment {args.model_type}: {e}", exc_info=True)

        else:
            logger.error(f"Unknown or unsupported model type/experiment key for training: '{args.model_type}'")


    elif args.mode == "evaluate":
        logger.info(f"Starting evaluation for experiment key: {args.model_type} on dataset: {args.dataset}")

        # --- Update models with potentially different evaluation logic ---
        # Uses the NEW keys
        models_with_internal_eval = [
            "experiment-3", # Logistic Regression + TF-IDF + Syntactic
            "experiment-4", # Multinomial NB + BoW + Syntactic
            "experiment-5", # Random Forest + BoW + Syntactic
            "experiment-6", # Gradient Boosting + TF-IDF + Syntactic
            "experiment-7", # Cross-eval happens in 'train' mode
            "experiment-8"  # Cross-validation happens in 'train' mode
        ]
        # Baselines and Exp 1, 2 are assumed standard here
        standard_eval_models = [
            "baseline-1",
            "baseline-2",
            "baseline-3",
            "experiment-1",
            "experiment-2",
        ]
        # -------------------------------------------------------------

        if args.model_type in ["experiment-7", "experiment-8"]:
             logger.warning(f"Evaluation for {args.model_type} is performed during its 'train' mode execution. Please run with --mode train.")

        elif args.model_type in models_with_internal_eval:
            # For Exp 3, 4, 5, 6 - Use BaselineTrainer to trigger evaluation
            logger.info(f"Attempting evaluation for {args.model_type} via BaselineTrainer (may rely on internal data loading).")
            try:
                trainer = BaselineTrainer(
                    model_type=args.model_type,
                    dataset_name=args.dataset,
                    args=args
                )
                # run_evaluation now expects optional data, relies on model's internal loading if data is None
                eval_metrics = trainer.run_evaluation(None) # Pass None for test_data
                if eval_metrics:
                    logger.info(f"Evaluation metrics for {args.model_type} on test set: {eval_metrics}")
                else:
                    logger.error(f"Evaluation run failed for {args.model_type} or model doesn't support this mode well.")
            except Exception as e:
                 logger.error(f"Failed to initialize or run evaluation via BaselineTrainer for {args.model_type}: {e}", exc_info=True)

        elif args.model_type in standard_eval_models:
            # For other models assumed to be handled by BaselineTrainer with explicit test data
            logger.info(f"Attempting evaluation for {args.model_type} via BaselineTrainer with explicit test data loading.")
            try:
                trainer = BaselineTrainer(
                    model_type=args.model_type,
                    dataset_name=args.dataset,
                    args=args
                )
                # BaselineTrainer's _load_data handles loading strategy
                # Request only test data for evaluation mode if possible
                # Modify _load_data or add a specific method if needed, otherwise load all
                _, _, test_data = trainer._load_data() # Assumes this loads train/val/test
                if test_data is None or (hasattr(test_data, 'empty') and test_data.empty):
                     logger.warning(f"Could not load explicit test data for {args.dataset} test split. Evaluation might fail if {args.model_type} requires it.")
                     eval_metrics = trainer.run_evaluation(None) # Try anyway, maybe model loads internally
                else:
                     logger.info(f"Test data loaded successfully for {args.model_type} evaluation.")
                     eval_metrics = trainer.run_evaluation(test_data) # Pass explicit test data

                if eval_metrics:
                    logger.info(f"Evaluation metrics for {args.model_type} on test set: {eval_metrics}")
                else:
                    logger.error(f"Evaluation run failed for {args.model_type} or produced no metrics.")
            except Exception as e:
                 logger.error(f"Failed to initialize or run evaluation via BaselineTrainer for {args.model_type}: {e}", exc_info=True)

        else:
             logger.error(f"Unknown or unsupported model type/experiment key for evaluation: '{args.model_type}'")


    elif args.mode == "predict":
        logger.warning("Prediction mode not fully implemented yet.")
        # Add prediction logic here if needed
        pass
    else:
        logger.error(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()