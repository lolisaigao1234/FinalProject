# Modify file: IS567FP/main.py
# Add logic to handle experiment 8
import logging

from config import parse_args, DEVICE
from data.preprocessor import TextPreprocessor
from models.baseline_trainer import BaselineTrainer  # Use the unified trainer
# <<< Import Experiment 7 & 8 classes >>>
from models.cross_eval_syntactic_experiment_7 import CrossEvalSyntacticExperiment7
from models.cross_validate_syntactic_experiment_8 import CrossValidateSyntacticExperiment8
# ---------------------------------
from utils.database import DatabaseHandler
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


def preprocess_data(args):
    """Runs the preprocessing pipeline based on arguments."""
    logger.info(f"Preprocessing {args.dataset} dataset. Sample size: {args.sample_size or 'Full'}.")
    db_handler = DatabaseHandler()
    # PreprocessorInterface expects sample_size, but TextPreprocessor gets it from args later
    # Pass db_handler only
    preprocessor = TextPreprocessor(db_handler)
    # Call the pipeline method which now handles sampling internally
    preprocessor.preprocess_dataset_pipeline(
        dataset_name=args.dataset,
        total_sample_size=args.sample_size,  # Pass the total size argument
        force_reprocess=args.force_reprocess
        # train_ratio is not used directly here anymore if sampling is per split
    )
    logger.info("Preprocessing pipeline complete.")


def main():
    """Main entry point."""
    args = parse_args()

    # Set benchmark/deterministic (optional, adjust based on needs)
    torch.backends.cudnn.benchmark = False  # Default False for baselines unless needed
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
        # List of models handled by the standard BaselineTrainer
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
        ]
        # Special experiment runners
        special_experiment_runners = {
            "cross_eval_syntactic_exp7": CrossEvalSyntacticExperiment7,
            "cross_validate_syntactic_experiment_8": CrossValidateSyntacticExperiment8  # Added Exp 8
        }

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

        # <<< Handle Special Experiments (Exp7, Exp8) >>>
        elif args.model_type in special_experiment_runners:
            logger.info(f"Initializing special experiment runner for: {args.model_type}, dataset: {args.dataset}")
            ExperimentRunnerClass = special_experiment_runners[args.model_type]
            experiment_runner = ExperimentRunnerClass(args)
            logger.info(f"Starting {args.model_type} run...")
            results = experiment_runner.run_experiment()
            if results:
                logger.info(f"{args.model_type} finished. Overall Results Summary:")
                # Log detailed results (assuming 'results' is a dictionary)
                for config, metrics in results.items():
                    if isinstance(metrics, dict):  # Ensure metrics is a dict before logging details
                        metrics_str = ", ".join(
                            [f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()])
                        logger.info(f"  - Config: {config}, Metrics: {metrics_str}")
                    else:
                        logger.info(f"  - Config: {config}, Result: {metrics}")  # Log non-dict results directly
            else:
                logger.error(f"{args.model_type} run failed or produced no results.")
        # ----------------------------------------

        # Add handling for other potential model types (e.g., neural) if needed
        # elif args.model_type == "neural":
        #      logger.warning("Neural network training path not fully implemented.")
        else:
            logger.error(f"Unknown or unsupported model type for training: '{args.model_type}'")


    elif args.mode == "evaluate":
        logger.info(f"Starting evaluation for model type: {args.model_type} on dataset: {args.dataset}")

        # Models that load data internally or have special eval logic
        models_with_internal_eval = [
            'logistic_tfidf_syntactic_exp3',
            'mnb_bow_syntactic_exp4',
            'random_forest_bow_syntactic_exp5',
            'gradient_boosting_tfidf_syntactic_exp6',
            'cross_eval_syntactic_exp7',  # Experiment 7 runs comparisons, maybe no separate eval mode
            'cross_validate_syntactic_experiment_8'  # Experiment 8 is CV, eval is implicit in 'train' mode
        ]

        if args.model_type in models_with_internal_eval:
            logger.warning(f"Evaluation mode might not be applicable or is handled internally for {args.model_type}.")
            # For Exp 7 & 8, cross-evaluation/validation happens during their 'train' run.
            if args.model_type in ['cross_eval_syntactic_exp7', 'cross_validate_syntactic_experiment_8']:
                logger.warning(
                    f"For {args.model_type}, run in 'train' mode to perform the evaluation/cross-validation.")
            else:
                # For Exp 3, 4, 5, 6 - Use BaselineTrainer to trigger evaluation if model has predict_on_dataframe
                logger.info(
                    f"{args.model_type} likely handles data loading internally. Attempting evaluation via BaselineTrainer.")
                trainer = BaselineTrainer(
                    model_type=args.model_type,
                    dataset_name=args.dataset,
                    args=args
                )
                # run_evaluation now expects optional data, relies on model's internal loading if data is None
                eval_metrics = trainer.run_evaluation(None)
                if eval_metrics:
                    logger.info(f"Evaluation metrics on test set: {eval_metrics}")
                else:
                    logger.error("Evaluation run failed or model doesn't support this mode well.")

        else:
            # For other models handled by BaselineTrainer (SVM variants, base TFIDF, base MNB)
            trainer = BaselineTrainer(
                model_type=args.model_type,
                dataset_name=args.dataset,
                args=args
            )
            # Attempt to load explicit test data if needed by the model
            # BaselineTrainer's _load_data handles loading strategy based on model type
            _, _, test_data = trainer._load_data()
            if test_data is None or test_data.empty:
                logger.warning(
                    f"Could not load explicit test data for {args.dataset} test split for {args.model_type}. Evaluation might fail if model requires it.")

            # run_evaluation uses loaded test_data
            eval_metrics = trainer.run_evaluation(test_data)
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
