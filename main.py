# File: IS567FP/main.py
# Updated to use ExperimentTrainer for experiments 1-6

import logging
import torch

# Assuming config.py defines parse_args() and DEVICE
# DEVICE is likely determined later in the script, using args.device
from config import parse_args #, DEVICE <-- Removed DEVICE import here
from data.preprocessor import TextPreprocessor
# --- MODIFICATION START: Import both trainers ---
from models.baseline_trainer import BaselineTrainer
from models.base_experiment_trainer import ExperimentTrainer # Import the new trainer
# --- MODIFICATION END ---

# --- Import corrected Experiment 7 & 8 classes ---
# Make sure these class names match exactly what's defined in the files
try:
    # Note: Class names in import should match the actual classes defined
    # E.g., if the file defines SyntacticFeatureEvaluator, import that.
    from models.cross_eval_syntactic_experiment_7 import CrossEvalSyntacticExperiment7 # Or SyntacticFeatureEvaluator if that's the class name
except ImportError:
    logging.error("Failed to import runner from models.cross_eval_syntactic_experiment_7")
    # Define a dummy class to avoid crashing if the file is missing/incorrect
    class CrossEvalSyntacticExperiment7: pass
try:
    # E.g., if the file defines CrossValidator, import that.
    from models.cross_validate_syntactic_experiment_8 import CrossValidateSyntacticExperiment8 # Or CrossValidator if that's the class name
except ImportError:
    logging.error("Failed to import runner from models.cross_validate_syntactic_experiment_8")
    # Define a dummy class to avoid crashing if the file is missing/incorrect
    class CrossValidateSyntacticExperiment8: pass
# --------------------------------------------------
from utils.database import DatabaseHandler
# --- MODIFICATION START: Import model registry ---
# Needed to check if model_type is valid before dispatching
from models import MODEL_REGISTRY
# --- MODIFICATION END ---


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


def preprocess_data(args):
    """Runs the preprocessing pipeline based on arguments."""
    logger.info(f"Preprocessing {args.dataset} dataset. Sample size: {args.sample_size or 'Full'}.")
    db_handler = DatabaseHandler()
    preprocessor = TextPreprocessor(db_handler) # Assuming constructor takes db_handler
    # Check if preprocess_dataset_pipeline exists and call it
    if hasattr(preprocessor, 'preprocess_dataset_pipeline'):
        preprocessor.preprocess_dataset_pipeline(
            dataset_name=args.dataset,
            total_sample_size=args.sample_size, # Match arg name used in function
            force_reprocess=args.force_reprocess
        )
    elif hasattr(preprocessor, 'preprocess_dataset'):
         logger.warning("preprocess_dataset_pipeline not found, falling back to preprocess_dataset for train split only.")
         # Fallback or adjust based on TextPreprocessor's available methods
         # This might need adjustment depending on TextPreprocessor implementation
         preprocessor.preprocess_dataset(
             dataset_name=args.dataset,
             split='train', # Example: only process train split
             sample_size=args.sample_size,
             force_reprocess=args.force_reprocess
         )
    else:
        logger.error("TextPreprocessor does not have a known preprocessing method ('preprocess_dataset_pipeline' or 'preprocess_dataset').")

    logger.info("Preprocessing complete (or attempted).")


def main():
    """Main entry point."""
    args = parse_args()

    # --- Determine Device (Moved from potential config import) ---
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA device requested but not available. Falling back to CPU.")
        args.device = 'cpu'
    elif args.device == 'mps' and hasattr(torch.backends, 'mps') and not torch.backends.mps.is_available():
        logger.warning("MPS device requested but not available. Falling back to CPU.")
        args.device = 'cpu'
    elif not args.device or args.device == 'auto':
        if torch.cuda.is_available():
            args.device = 'cuda'
        # Check MPS availability properly
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'
    # --- Store determined device for logging ---
    determined_device = args.device
    # ------------------------------------------------------------

    # Optional: Set benchmark/deterministic based on needs
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True # Set True for full reproducibility if needed

    logger.info(f"Running in {args.mode} mode on {args.dataset} dataset")
    logger.info(f"Using device: {determined_device}") # Use the determined device
    logger.info(f"Selected model type / experiment key: {args.model_type}")
    if args.sample_size:
        logger.info(f"Using sample size: {args.sample_size}")
    else:
        logger.info("Processing full dataset (no sample size specified).")

    # --- Validate model_type globally if needed ---
    if args.mode in ['train', 'evaluate'] and args.model_type not in MODEL_REGISTRY:
         # Check if it's one of the special runners if not in registry (though they should be in registry ideally)
         special_keys = ["experiment-7", "experiment-8"] # Match keys used later
         if args.model_type not in special_keys:
            logger.error(f"Invalid model_type '{args.model_type}'. Not found in MODEL_REGISTRY or special runners. Available: {list(MODEL_REGISTRY.keys())}")
            return # Exit if invalid model type

    # --- Mode Dispatch ---
    if args.mode == "preprocess":
        preprocess_data(args)

    elif args.mode == "train":
        # --- MODIFICATION START: Define model categories ---
        baseline_model_types = [
            "baseline-1",
            "baseline-2",
            "baseline-3",
        ]
        experiment_model_types_via_trainer = [
            "experiment-1",
            "experiment-2",
            "experiment-3",
            "experiment-4",
            "experiment-5",
            "experiment-6",
        ]
        # Special experiment runners (using potentially corrected class names)
        special_experiment_runners = {
            "experiment-7": CrossEvalSyntacticExperiment7, # Or SyntacticFeatureEvaluator
            "experiment-8": CrossValidateSyntacticExperiment8 # Or CrossValidator
        }
        # --- MODIFICATION END ---

        trainer = None
        results = None

        # --- MODIFICATION START: Instantiate correct trainer ---
        if args.model_type in baseline_model_types:
            logger.info(f"Initializing BaselineTrainer for baseline: {args.model_type}, dataset: {args.dataset}")
            try:
                trainer = BaselineTrainer(
                    model_type=args.model_type,
                    dataset_name=args.dataset,
                    args=args
                )
            except Exception as e:
               logger.error(f"Failed to initialize BaselineTrainer for {args.model_type}: {e}", exc_info=True)

        elif args.model_type in experiment_model_types_via_trainer:
             logger.info(f"Initializing ExperimentTrainer for experiment: {args.model_type}, dataset: {args.dataset}")
             try:
                 trainer = ExperimentTrainer(
                     model_type=args.model_type,
                     dataset_name=args.dataset,
                     args=args
                 )
             except Exception as e:
                logger.error(f"Failed to initialize ExperimentTrainer for {args.model_type}: {e}", exc_info=True)
        # --- MODIFICATION END ---

        # --- Run standard trainers (if initialized) ---
        if trainer:
            try:
                logger.info(f"Starting training process via {trainer.__class__.__name__} for {args.model_type}...")
                results = trainer.run_training()
                if results:
                    logger.info(f"Training for {args.model_type} finished. Results: {results}")
                     # Optional evaluation after training (using the same trainer instance)
                    if args.evaluate_after_train:
                         logger.info(f"--- Starting Evaluation after Training ({args.eval_split} split) ---")
                         eval_results = trainer.run_evaluation(eval_split=args.eval_split)
                         if eval_results:
                             logger.info(f"Evaluation completed for {args.model_type}. Results: {eval_results}")
                         else:
                             logger.warning(f"Evaluation run after training failed or produced no results for {args.model_type}.")
                    else:
                         logger.info("Evaluation after training not requested.")
                else:
                    logger.error(f"Training run for {args.model_type} failed or produced no results.")
            except Exception as e:
                logger.error(f"Failed during run_training for {args.model_type} with {trainer.__class__.__name__}: {e}", exc_info=True)


        # --- Handle Special Experiments (Exp7, Exp8) ---
        elif args.model_type in special_experiment_runners:
            logger.info(f"Initializing special experiment runner for: {args.model_type}, dataset: {args.dataset}")
            try:
                ExperimentRunnerClass = special_experiment_runners[args.model_type]
                # Basic check if the class looks like a placeholder
                # A better check might be isinstance(ExperimentRunnerClass, type) and ExperimentRunnerClass != type(None) etc.
                if not hasattr(ExperimentRunnerClass, '__module__') or ExperimentRunnerClass.__name__ in ['CrossEvalSyntacticExperiment7', 'CrossValidateSyntacticExperiment8']:
                     # Check if the actual imported class has the run method
                     if hasattr(ExperimentRunnerClass, 'run_experiment'):
                          logger.info(f"Found runner class {ExperimentRunnerClass.__name__}")
                     else:
                          logger.error(f"Cannot run {args.model_type} because its class {ExperimentRunnerClass.__name__} failed to import correctly or lacks 'run_experiment' method.")
                          ExperimentRunnerClass = None # Prevent execution

                if ExperimentRunnerClass:
                    # Assuming constructor takes args or relevant parts like dataset, device etc.
                    # Adjust constructor call based on actual Exp7/8 runner needs
                    experiment_runner = ExperimentRunnerClass(args=args) # Pass args, runner extracts what it needs
                    logger.info(f"Starting {args.model_type} run...")
                    # Assuming this method exists and performs the experiment/training/evaluation
                    results = experiment_runner.run_experiment()
                    if results:
                        logger.info(f"{args.model_type} finished. Overall Results Summary:")
                        # Log detailed results (copied existing formatting logic)
                        if isinstance(results, dict):
                            for config_key, metrics in results.items(): # Renamed 'config' to 'config_key'
                                if isinstance(metrics, dict):
                                    metrics_str = ", ".join(
                                        [f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()])
                                    logger.info(f"  - Config: {config_key}, Metrics: {metrics_str}")
                                else:
                                    logger.info(f"  - Config: {config_key}, Result: {metrics}")
                        else:
                            logger.info(f"  Results: {results}")
                    else:
                        logger.error(f"{args.model_type} run failed or produced no results.")
            except Exception as e:
                logger.error(f"Failed to initialize or run special experiment {args.model_type}: {e}", exc_info=True)

        # --- Handle case where model_type is not recognized ---
        # This case might be redundant if initial validation catches it, but good as a fallback.
        elif trainer is None and args.model_type not in special_experiment_runners:
             logger.error(f"Unknown or unsupported model type/experiment key for training: '{args.model_type}'")


    elif args.mode == "evaluate":
        logger.info(f"Starting evaluation for model: {args.model_type} on dataset: {args.dataset}, split: {args.eval_split}")

        # --- MODIFICATION START: Define model categories for evaluation ---
        baseline_model_types = [ "baseline-1", "baseline-2", "baseline-3"]
        experiment_model_types_via_trainer = [
            "experiment-1", "experiment-2", "experiment-3",
            "experiment-4", "experiment-5", "experiment-6"
        ]
        eval_handled_in_train = ["experiment-7", "experiment-8"]
        # --- MODIFICATION END ---

        trainer = None
        eval_results = None

        # --- Warn for models evaluated during training ---
        if args.model_type in eval_handled_in_train:
            logger.warning(f"Evaluation for {args.model_type} is performed during its 'train' mode execution. Please run with --mode train.")

        # --- Instantiate correct trainer for evaluation ---
        elif args.model_type in baseline_model_types:
            logger.info(f"Initializing BaselineTrainer for evaluation: {args.model_type}")
            try:
                trainer = BaselineTrainer(
                    model_type=args.model_type,
                    dataset_name=args.dataset,
                    args=args
                )
            except Exception as e:
               logger.error(f"Failed to initialize BaselineTrainer for evaluation of {args.model_type}: {e}", exc_info=True)

        elif args.model_type in experiment_model_types_via_trainer:
            logger.info(f"Initializing ExperimentTrainer for evaluation: {args.model_type}")
            try:
                 trainer = ExperimentTrainer(
                     model_type=args.model_type,
                     dataset_name=args.dataset,
                     args=args
                 )
            except Exception as e:
                 logger.error(f"Failed to initialize ExperimentTrainer for evaluation of {args.model_type}: {e}", exc_info=True)

        # --- Run evaluation using the instantiated trainer ---
        if trainer:
             try:
                 logger.info(f"Starting evaluation process via {trainer.__class__.__name__} for {args.model_type}...")
                 # Call run_evaluation - it handles loading the MODEL.
                 # The model's evaluate method handles loading the required DATA.
                 eval_results = trainer.run_evaluation(eval_split=args.eval_split)

                 if eval_results:
                     # run_evaluation returns a dict like {model_key: metrics}
                     metrics = eval_results.get(args.model_type) # Extract metrics for the specific model
                     if metrics:
                          logger.info(f"Evaluation metrics for {args.model_type} on {args.eval_split} split: {metrics}")
                     else:
                          logger.error(f"Evaluation run completed but metrics not found in result for {args.model_type}.")
                 else:
                     logger.error(f"Evaluation run failed for {args.model_type} or produced no results.")
             except Exception as e:
                 logger.error(f"Failed during run_evaluation for {args.model_type} with {trainer.__class__.__name__}: {e}", exc_info=True)

        # --- Handle unknown models for evaluation ---
        elif args.model_type not in eval_handled_in_train: # Only show error if not handled in train mode
            logger.error(f"Unknown or unsupported model type/experiment key for evaluation: '{args.model_type}'")


    elif args.mode == "predict":
        logger.warning("Prediction mode not fully implemented yet.")
        # Add prediction logic here if needed
        pass
    else:
        logger.error(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()