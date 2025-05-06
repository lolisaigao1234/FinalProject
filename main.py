# File: IS567FP/main.py
# Updated to handle 'predict' mode using test files as input

import logging
import time
from typing import Optional

import torch
import os
import pandas as pd
import gc  # For garbage collection

# Assuming config.py defines parse_args(), MODEL_CHOICES, PREDICT_MODEL_CHOICES, MODELS_DIR, LABEL_MAP_REVERSE, OUTPUT_DIR
from config import parse_args, MODEL_CHOICES, PREDICT_MODEL_CHOICES, MODELS_DIR, LABEL_MAP_REVERSE, OUTPUT_DIR

from data.preprocessor import TextPreprocessor
from features.feature_extractor import FeatureExtractor
from models.baseline_trainer import BaselineTrainer
from models.base_experiment_trainer import ExperimentTrainer

# --- Import helpers and model classes ---
from models import MODEL_REGISTRY
from utils.common import NLIModel
# Import SimpleParquetLoader and clean_dataset
from models.baseline_base import SimpleParquetLoader, clean_dataset
from models import (
    DecisionTreeBowBaseline, LogisticTFIDFBaseline, MultinomialNaiveBayesBaseline,
    DecisionTreeSyntacticExperiment1, KnnBowSyntacticExperiment2,
    LogisticTFIDFSyntacticExperiment3, MultinomialNaiveBayesBowSyntacticExperiment4,
    RandomForestBowSyntacticExperiment5, GradientBoostingTFIDFSyntacticExperiment6,
    CrossEvalSyntacticExperiment7, CrossValidateSyntacticExperiment8
)

# ----------------------------------------

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)


# --------------------


def main():
    args = parse_args()
    logger.info(f"Running mode: {args.mode}")
    logger.info(f"Selected model type: {args.model_type}")
    logger.info(f"Selected dataset: {args.dataset}")  # Note: this dataset is for train/eval mode by default
    logger.info(f"Using device: {args.device}")
    if args.sample_size:
        logger.info(f"Using sample size: {args.sample_size}")

    # Suffix for train/eval modes
    suffix = f"sample{args.sample_size}" if args.sample_size else "full"

    # --- Mode Handling ---
    if args.mode == "preprocess":
        # (Keep preprocess logic as before)
        logger.info("Starting preprocessing...")
        from utils.database import DatabaseHandler
        db_handler = DatabaseHandler()
        preprocessor = TextPreprocessor(db_handler)
        preprocessor.preprocess_dataset_pipeline(
            dataset_name=args.dataset,
            total_sample_size=args.sample_size,
            force_reprocess=args.force_reprocess,
        )
        logger.info("Preprocessing finished.")

    elif args.mode == "extract_features":
        # (Keep extract_features logic as before)
        logger.info("Starting feature extraction...")
        from utils.database import DatabaseHandler
        db_handler = DatabaseHandler()
        feature_extractor = FeatureExtractor(db_handler=db_handler)
        for split in ["train", "dev", "test"]:
            logger.info(f"Extracting features for split: {split} (suffix: {suffix})")
            try:
                feature_extractor.extract_features(
                    dataset_name=args.dataset,
                    split=split,
                    suffix=suffix,
                    force_recompute=args.force_reprocess
                )
            except Exception as e:
                logger.error(f"Failed to extract features for split {split}: {e}", exc_info=True)
        logger.info("Feature extraction finished.")

    elif args.mode == "train":
        # (Keep train logic as before)
        logger.info(f"Starting training for model: {args.model_type}")
        if args.model_type not in MODEL_REGISTRY:
            logger.error(f"Model type '{args.model_type}' not found.")
            return
        trainer = None
        model_class = MODEL_REGISTRY[args.model_type]
        if args.model_type.startswith('experiment-') and args.model_type not in ["experiment-7", "experiment-8"]:
            trainer = ExperimentTrainer(model_type=args.model_type, dataset_name=args.dataset, args=args)
        elif args.model_type.startswith('baseline-'):
            trainer = BaselineTrainer(model_type=args.model_type, dataset_name=args.dataset, args=args)
        elif args.model_type == "experiment-7":
            try:
                exp7_runner = CrossEvalSyntacticExperiment7(args=args);
                results = exp7_runner.run_experiment()
                logger.info(f"Experiment 7 Results: {results}")
            except Exception as e:
                logger.error(f"Error running Experiment 7: {e}", exc_info=True)
        elif args.model_type == "experiment-8":
            try:
                exp8_runner = CrossValidateSyntacticExperiment8(args=args);
                results = exp8_runner.run_experiment()
                logger.info(f"Experiment 8 Results: {results}")
            except Exception as e:
                logger.error(f"Error running Experiment 8: {e}", exc_info=True)
        else:
            logger.error(f"Cannot determine trainer for model type: {args.model_type}"); return

        if trainer:
            try:
                train_results = trainer.run_training()
                if train_results:
                    logger.info(f"Training completed. Results: {train_results}")
                    if args.evaluate_after_train:
                        logger.info(f"Evaluating model {args.model_type} on '{args.eval_split}' split...")
                        eval_results = trainer.run_evaluation(eval_split=args.eval_split)
                        if eval_results:
                            logger.info(f"Evaluation metrics: {eval_results.get(args.model_type)}")
                        else:
                            logger.error("Evaluation failed after training.")
                else:
                    logger.error(f"Training failed for model {args.model_type}")
            except Exception as e:
                logger.error(f"Error during training for {args.model_type}: {e}", exc_info=True)

    elif args.mode == "evaluate":
        # (Keep evaluate logic as before)
        logger.info(f"Starting evaluation for model: {args.model_type} on split: {args.eval_split}")
        if args.model_type not in MODEL_REGISTRY: logger.error(f"Model type '{args.model_type}' not found."); return
        eval_handled_in_train = ["experiment-7", "experiment-8"]
        trainer = None
        if args.model_type.startswith('experiment-') and args.model_type not in eval_handled_in_train:
            trainer = ExperimentTrainer(model_type=args.model_type, dataset_name=args.dataset, args=args)
        elif args.model_type.startswith('baseline-'):
            trainer = BaselineTrainer(model_type=args.model_type, dataset_name=args.dataset, args=args)

        if trainer:
            try:
                eval_results = trainer.run_evaluation(eval_split=args.eval_split)
                if eval_results:
                    logger.info(f"Evaluation metrics: {eval_results.get(args.model_type)}")
                else:
                    logger.error(f"Evaluation run failed or produced no results.")
            except Exception as e:
                logger.error(f"Failed during run_evaluation: {e}", exc_info=True)
        elif args.model_type not in eval_handled_in_train:
            logger.error(f"Unknown or unsupported model type for evaluation: '{args.model_type}'")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++                  PREDICTION MODE                   +++
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    elif args.mode == "predict":
        logger.info(f"Starting prediction mode for model: {args.model_type}")
        logger.info(f"Predicting on dataset: {args.predict_input_dataset}, suffix: {args.predict_input_suffix}")

        # --- Validate Arguments ---
        if args.model_type not in PREDICT_MODEL_CHOICES:
            logger.error(
                f"Error: Model type '{args.model_type}' is not supported for prediction (valid: {PREDICT_MODEL_CHOICES}).")
            return

        # --- Load Input Data ---
        input_df = None
        logger.info(
            f"Loading input data for prediction from {args.predict_input_dataset}/test/{args.predict_input_suffix}")
        try:
            loader = SimpleParquetLoader()
            # Load the 'test' split of the specified input dataset and suffix
            # The loader should find the features_stats_syntactic file ideally
            input_df = loader.load_data(loader, args.predict_input_dataset, 'test',
                                        args.predict_input_suffix)  # Pass loader instance

            if input_df is None or input_df.empty:
                raise FileNotFoundError("Loaded input data is None or empty.")

            logger.info(f"Loaded {len(input_df)} rows from input file.")

            # Apply limit if specified
            if args.predict_limit:
                logger.info(f"Applying prediction limit: {args.predict_limit} rows.")
                if args.predict_limit < len(input_df):
                    input_df = input_df.head(args.predict_limit)
                else:
                    logger.warning(
                        f"Prediction limit ({args.predict_limit}) is >= number of rows ({len(input_df)}). Predicting on all loaded rows.")

        except FileNotFoundError:
            logger.error(
                f"Error: Input feature file not found for dataset '{args.predict_input_dataset}', split 'test', suffix '{args.predict_input_suffix}'.")
            logger.error("Ensure features were extracted for the test set using '--mode extract_features'.")
            return
        except Exception as e:
            logger.error(f"Error loading input data: {e}", exc_info=True)
            return

        # --- Load Model ---
        loaded_model: Optional[NLIModel] = None
        model_class = MODEL_REGISTRY.get(args.model_type)
        if not model_class: logger.error(f"Internal Error: Model key '{args.model_type}' invalid."); return

        model_save_subdir = 'baseline_models'  # Assuming models saved here
        # Use predict_model_dataset and predict_model_suffix for loading path
        model_dir = os.path.join(MODELS_DIR, model_save_subdir, args.predict_model_dataset, args.model_type,
                                 args.predict_model_suffix)
        model_base_name = f"{args.predict_model_dataset}_{args.model_type}_{args.predict_model_suffix}"

        logger.info(
            f"Attempting to load model '{args.model_type}' trained on '{args.predict_model_dataset}/{args.predict_model_suffix}'")
        logger.info(f"Loading from directory: {model_dir} with base name: {model_base_name}")

        try:
            loaded_model = model_class.load(model_dir, model_base_name)
            if not loaded_model: raise ValueError("Model loading returned None.")
            logger.info(f"Model {args.model_type} loaded successfully.")
            if not getattr(loaded_model, 'is_trained', True): logger.warning("Loaded model is not marked as trained.")
        except FileNotFoundError:
            logger.error(f"Error: Model artifacts not found in '{model_dir}' for base name '{model_base_name}'.")
            return
        except Exception as e:
            logger.error(f"Error loading model '{args.model_type}': {e}", exc_info=True); return

        # --- Prepare Data for Model ---
        # Clean the loaded data (handles labels if present, ensures basic consistency)
        logger.info("Cleaning loaded input data...")
        cleaned_data_result = clean_dataset(input_df)
        if cleaned_data_result is None:
            logger.error("Input data became invalid after cleaning.")
            return
        df_cleaned, y_true_potential = cleaned_data_result  # y_true might be present, or None/empty
        logger.info(f"{len(df_cleaned)} rows remaining after cleaning.")

        if df_cleaned.empty:
            logger.error("No valid rows remaining in input data after cleaning.")
            return

        # The df_cleaned should contain all columns needed (text, features)
        # because it came from the feature extractor's output file.

        # --- Make Predictions ---
        predictions_int = None
        try:
            logger.info(f"Generating predictions for {len(df_cleaned)} samples...")
            start_pred_time = time.time()

            # Use the model's appropriate prediction method
            if hasattr(loaded_model, 'predict_on_dataframe'):
                predictions_int = loaded_model.predict_on_dataframe(df_cleaned)
            elif hasattr(loaded_model, 'predict') and hasattr(loaded_model, 'extract_features'):
                # Assumes extract_features correctly uses the already loaded+cleaned features
                # Needs to handle potential DataFrame vs numpy array input for predict
                features = loaded_model.extract_features(df_cleaned)
                if features is None or features.shape[0] == 0:
                    logger.error("Feature extraction for prediction failed or returned empty.")
                    raise RuntimeError("Feature extraction failed during prediction.")
                predictions_int = loaded_model.predict(features)
            else:
                logger.error(f"Model {args.model_type} lacks a suitable prediction method.")
                raise NotImplementedError("Prediction method not found.")

            pred_time = time.time() - start_pred_time
            logger.info(f"Prediction finished in {pred_time:.2f} seconds.")

            if predictions_int is None or len(predictions_int) != len(df_cleaned):
                logger.error(
                    f"Prediction resulted in unexpected number of labels (Expected: {len(df_cleaned)}, Got: {len(predictions_int) if predictions_int is not None else 'None'}).")
                raise RuntimeError("Prediction length mismatch.")

        except Exception as e:
            logger.error(f"An error occurred during prediction: {e}", exc_info=True)
            return

        # --- Format and Save Output ---
        try:
            logger.info("Formatting output...")
            # Map integer predictions to string labels
            predictions_str = [LABEL_MAP_REVERSE.get(p, "unknown") for p in predictions_int]

            # Create output DataFrame
            output_df = pd.DataFrame({
                'pair_id': df_cleaned['pair_id'],  # Assumes pair_id survived cleaning
                'premise': df_cleaned['premise_text'],  # Assumes text cols are present
                'hypothesis': df_cleaned['hypothesis_text'],
                'predicted_label_int': predictions_int,
                'predicted_label_str': predictions_str
            })

            # Include true label if available in the cleaned data
            true_label_col = 'gold_label' if 'gold_label' in df_cleaned.columns else 'label'
            if true_label_col in df_cleaned.columns:
                # Ensure alignment if clean_dataset modified indices (shouldn't if just cleaning labels)
                output_df['true_label_str'] = df_cleaned[true_label_col]
                if y_true_potential is not None and len(y_true_potential) == len(output_df):
                    output_df['true_label_int'] = y_true_potential
                else:
                    logger.warning("Could not align true integer labels from clean_dataset.")

            # Display head
            print("\n--- Prediction Results (Top 5) ---")
            print(output_df.head().to_string())
            print("-" * 30)

            # Save to file
            if args.predict_output_file:
                try:
                    output_dir = os.path.dirname(args.predict_output_file)
                    if output_dir:  # Ensure directory exists if specified in path
                        os.makedirs(output_dir, exist_ok=True)
                    output_df.to_csv(args.predict_output_file, index=False)
                    logger.info(f"Predictions saved successfully to: {args.predict_output_file}")
                except Exception as e:
                    logger.error(f"Failed to save predictions to {args.predict_output_file}: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"Error formatting or saving results: {e}", exc_info=True)

        # Cleanup
        del input_df, df_cleaned, output_df, loaded_model, predictions_int, predictions_str
        gc.collect()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++                END PREDICTION MODE                 +++
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    else:
        logger.error(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
