# File: IS567FP/models/base_experiment_trainer.py
import os
import logging
import time
from typing import Optional, Dict, Any
import pandas as pd
import gc # For memory management

# Configuration and Utilities
from config import MODELS_DIR
from utils.common import NLIModel # Base class/interface
# Import helpers, loaders, and registry
from .baseline_base import clean_dataset, SimpleParquetLoader # Use parquet loader and cleaner
from utils.database import DatabaseHandler
from . import MODEL_REGISTRY # Access to all registered models

logger = logging.getLogger(__name__)

# Define feature types potentially used by experiments
# FEATURE_TYPES_NEEDED = ["lexical", "syntactic", "embedding"] # Or load a combined file

class ExperimentTrainer:
    """
    Trainer specifically designed for 'experiment' models that rely on
    pre-computed features loaded from Parquet files.
    """
    def __init__(self, model_type: str, dataset_name: str, args: object):
        """
        Initializes the trainer for a specific experiment model type and dataset.

        Args:
            model_type (str): Model identifier key from MODEL_REGISTRY (e.g., 'experiment-1').
            dataset_name (str): Name of the dataset (e.g., 'SNLI').
            args (object): Command line arguments containing hyperparameters.
        """
        self.model_key = model_type
        self.dataset_name = dataset_name
        self.args = args
        self.sample_size = getattr(args, 'sample_size', None)
        self.suffix = f"sample{self.sample_size}" if self.sample_size else "full"

        # Get the corresponding model class from the registry
        self.model_cls = MODEL_REGISTRY.get(self.model_key)
        if not self.model_cls:
            raise ValueError(f"Model type key '{self.model_key}' not found in MODEL_REGISTRY.")
        # Check if it starts with 'experiment-' (basic sanity check)
        if not self.model_key.startswith('experiment-'):
             logger.warning(f"Instantiating ExperimentTrainer for a non-experiment model key: {self.model_key}")

        self.save_dir = self._get_save_directory()
        os.makedirs(self.save_dir, exist_ok=True)
        self.db_handler = DatabaseHandler() # May not be strictly needed if not loading raw text
        self.model: Optional[NLIModel] = None

        logger.info(f"Initialized ExperimentTrainer for model key: '{self.model_key}', class: {self.model_cls.__name__}, dataset: {self.dataset_name}, suffix: {self.suffix}")

    def _get_save_directory(self) -> str:
        """Determines the save directory based on model key."""
        # Keep the same structure as BaselineTrainer for consistency
        base_dir = os.path.join(MODELS_DIR, 'baseline_models', self.dataset_name, self.model_key, self.suffix)
        # Consider changing 'baseline_models' to 'experiment_models' if desired
        # base_dir = os.path.join(MODELS_DIR, 'experiment_models', self.dataset_name, self.model_key, self.suffix)
        return base_dir

    def _get_model_filename_base(self) -> str:
        """Generates a base filename for saving models/artifacts."""
        return f"{self.dataset_name}_{self.model_key}_{self.suffix}"

    def _initialize_model(self) -> Optional[NLIModel]:
        """Initializes the model instance using args."""
        logger.info(f"Initializing model instance for {self.model_key} ({self.model_cls.__name__})")
        model_instance: Optional[NLIModel] = None

        # Gather hyperparameters from args - specific models use specific ones
        hyperparams = {
            'args': self.args, # Pass the whole args object
            # Pass specific ones if needed, e.g.:
            # 'max_depth': getattr(self.args, 'max_depth', None),
            # 'min_samples_split': getattr(self.args, 'min_samples_split', 2),
            # 'n_neighbors': getattr(self.args, 'n_neighbors', 5),
        }

        try:
            model_instance = self.model_cls(**hyperparams)
        except Exception as e:
            logger.error(f"Error initializing model {self.model_key} ({self.model_cls.__name__}): {e}", exc_info=True)
            return None

        if not isinstance(model_instance, NLIModel):
             logger.error(f"Initialized object for {self.model_key} is not an instance of NLIModel.")
             return None

        logger.info(f"Model {self.model_key} initialized successfully.")
        return model_instance

    def run_training(self) -> Optional[Dict[str, Any]]:
        """
        Runs the training pipeline for experiment models: Initialize, Load Features, Clean, Extract, Train, Save.
        """
        results = {}
        self.model = self._initialize_model()

        if not self.model:
            logger.error(f"Failed to initialize model {self.model_key}. Aborting training.")
            return None

        logger.info(f"--- Loading Pre-computed Features for {self.model_key} on {self.dataset_name} (train/{self.suffix}) ---")
        start_load_time = time.time()
        df_cleaned: Optional[pd.DataFrame] = None
        X_train = None
        y_train = None

        try:
            # --- Load Pre-computed Features ---
            logger.info(f"Loading pre-computed features using SimpleParquetLoader...")

            # Attempt to load the combined feature file first, assuming FeatureExtractor saves it.
            # Example filename pattern from FeatureExtractor: {dataset}_{split}_features_all_{suffix}.parquet
            # SimpleParquetLoader might need adjustment to find this specific file pattern.
            # For now, we rely on SimpleParquetLoader's current logic which might look for `features_{suffix}.parquet`
            # or `raw_data_{suffix}.parquet` in the cache dir.
            # A more robust loader would take the feature_type(s) as input.
            loader = SimpleParquetLoader()
            features_df = None
            try:
                # Try loading based on dataset, split, suffix.
                # The loader should ideally find the comprehensive feature file.
                logger.info(f"Attempting to load feature file matching: dataset='{self.dataset_name}', split='train', suffix='{self.suffix}'")
                features_df = loader.load_data(self, self.dataset_name, 'train', self.suffix)

                # Check if essential columns exist (adapt based on actual needs)
                # 'pair_id' and 'label' are crucial. Feature columns depend on the experiment.
                required_base_cols = ['pair_id', 'label']
                if not all(col in features_df.columns for col in required_base_cols):
                    missing = [col for col in required_base_cols if col not in features_df.columns]
                    logger.error(f"Loaded feature DataFrame is missing essential columns: {missing}")
                    raise ValueError(f"Missing essential columns in feature file: {missing}")

                # Optional: Check if specific feature columns expected by the model exist.
                # This might be better handled inside the model's extract_features.
                # Example: if hasattr(self.model, 'REQUIRED_FEATURE_PREFIXES'): ... check columns ...

            except FileNotFoundError:
                logger.error(f"Pre-computed feature file not found for dataset '{self.dataset_name}', split 'train', suffix '{self.suffix}'. "
                             f"Ensure features were generated by feature_extractor.py and saved to the expected location/name.", exc_info=True)
                raise
            except Exception as e:
                logger.error(f"Error loading pre-computed features: {e}", exc_info=True)
                raise

            if features_df is None or features_df.empty:
                raise ValueError("Failed to load pre-computed features or data is empty.")
            logger.info(f"Loaded {len(features_df)} rows with {len(features_df.columns)} columns from pre-computed feature file.")
            # Log some column names for debugging
            logger.debug(f"Feature columns available: {features_df.columns.tolist()}")

            # --- Clean Data (mainly handles labels) ---
            # Pass the features_df here. clean_dataset prepares labels and handles label issues.
            cleaned_data = clean_dataset(features_df)
            if cleaned_data is None:
                raise ValueError("Feature data became empty or invalid after cleaning labels.")
            df_cleaned, y_train = cleaned_data # df_cleaned contains features AND label
            logger.info(f"Feature data cleaned. {len(df_cleaned)} samples remaining.")

            if len(df_cleaned) == 0:
                 logger.warning("No valid training samples left after cleaning.")
                 return {"error": "No valid training samples after cleaning."}

            # --- Extract/Select Features ---
            # The model's extract_features method should handle selecting the right columns
            # from df_cleaned and potentially handling NaNs within those feature columns.
            logger.info(f"Extracting/Selecting features using {self.model_key}.extract_features...")
            # This step also implicitly validates that the required feature columns are present in df_cleaned.
            X_train = self.model.extract_features(df_cleaned)

            # --- Common steps after data/features are loaded ---
            if X_train is None or y_train is None:
                 raise ValueError("Feature extraction failed or labels are missing.")
            if X_train.shape[0] != len(y_train):
                 raise ValueError(f"Feature/Label mismatch after loading/extraction: X={X_train.shape}, y={len(y_train)}")

            logger.info(f"Training features prepared. Shape: {X_train.shape if X_train is not None else 'None'}")
            logger.info(f"Training labels prepared. Length: {len(y_train) if y_train is not None else 'None'}")

            # Clean up large DataFrames
            del features_df, df_cleaned
            gc.collect()

        except Exception as e:
            logger.error(f"Error during data loading/preparation for {self.model_key}: {e}", exc_info=True)
            return None # Return None indicates failure
        finally:
            load_time = time.time() - start_load_time
            logger.info(f"Data loading and preparation phase finished in {load_time:.2f}s")
        # --- END: 数据加载和准备 ---

        # --- START: 模型训练 ---
        logger.info(f"--- Starting model training for {self.model_key} on {self.dataset_name} ({self.suffix}) ---")
        start_train_time = time.time()
        try:
            if X_train is None or y_train is None:
                 raise ValueError("Training features (X_train) or labels (y_train) are not available for training.")

            # Call the model's train method
            self.model.train(X_train, y_train) # Assumes train takes X, y

            results[self.model_key] = {'status': 'trained'}

        except NotImplementedError:
             logger.error(f"Model class {self.model_cls.__name__} does not implement the 'train' method correctly.")
             return None
        except Exception as e:
            logger.error(f"Error during model training call for {self.model_key}: {e}", exc_info=True)
            return None
        finally:
            train_time = time.time() - start_train_time
            logger.info(f"Model training phase for {self.model_key} finished in {train_time:.2f}s")
            if self.model_key in results:
                results[self.model_key]['train_time'] = train_time
            # Clean up training data if large
            del X_train, y_train
            gc.collect()
        # --- END: 模型训练 ---

        # --- Save the trained model ---
        logger.info(f"Saving model {self.model_key}...")
        model_filename_base = self._get_model_filename_base()
        try:
            # Model's save method needs directory and base name
            self.model.save(self.save_dir, model_filename_base)
            logger.info(f"Model {self.model_key} saved in directory: {self.save_dir} with base name: {model_filename_base}")
        except NotImplementedError:
             logger.error(f"Model class {self.model_cls.__name__} does not implement the 'save' method.")
        except Exception as e:
            logger.error(f"Error saving model {self.model_key}: {e}", exc_info=True)

        return results

    def run_evaluation(self, eval_split: str = 'test') -> Optional[Dict[str, Any]]:
        """
        Runs the evaluation pipeline for experiment models: Load Model, Evaluate.
        Relies on the model's evaluate method to handle its data loading.

        Args:
            eval_split (str): The data split to evaluate on (e.g., 'test', 'validation').

        Returns:
            dict: Evaluation metrics, or None if evaluation fails.
        """
        logger.info(f"--- Starting evaluation for {self.model_key} on {self.dataset_name} split '{eval_split}' ({self.suffix}) ---")

        # --- Load the saved model ---
        model_filename_base = self._get_model_filename_base()
        loaded_model: Optional[NLIModel] = None
        try:
            logger.info(f"Loading model {self.model_key} from directory: {self.save_dir} with base name {model_filename_base}")
            # Experiment models should implement a load method that restores the pipeline/model and necessary attributes (like feature_cols)
            loaded_model = self.model_cls.load(self.save_dir, model_filename_base)
            if loaded_model is None: raise FileNotFoundError

            # --- Sanity checks after loading ---
            if not hasattr(loaded_model, 'is_trained') or not loaded_model.is_trained:
                 logger.warning(f"Loaded model {self.model_key} doesn't seem to be marked as trained.")
            # Check for feature columns, as experiments rely on them
            if not hasattr(loaded_model, 'feature_cols') or loaded_model.feature_cols is None:
                 logger.warning(f"Loaded experiment model {self.model_key} is missing 'feature_cols' attribute after loading.")


        except FileNotFoundError:
            logger.error(f"Model artifacts not found for {self.model_key} in directory {self.save_dir} with base name {model_filename_base}. Cannot evaluate.")
            return None
        except NotImplementedError:
             logger.error(f"Model class {self.model_cls.__name__} does not implement the 'load' method.")
             return None
        except Exception as e:
            logger.error(f"Error loading model {self.model_key} from {self.save_dir}: {e}", exc_info=True)
            return None

        if not isinstance(loaded_model, NLIModel):
             logger.error(f"Loaded object for {self.model_key} is not an instance of NLIModel.")
             return None

        # --- Perform Evaluation ---
        logger.info(f"Evaluating loaded model {self.model_key}...")
        eval_results = {}
        start_time = time.time()
        try:
            # Call the loaded model's evaluate method.
            # Models like DecisionTreeSyntacticExperiment1 implement evaluate to load parquet data and predict.
            eval_metrics = loaded_model.evaluate(
                dataset_name=self.dataset_name,
                split=eval_split,
                suffix=self.suffix
            )
            eval_results = eval_metrics if eval_metrics else {}

        except NotImplementedError:
             logger.error(f"Model class {self.model_cls.__name__} does not implement the 'evaluate' method.")
             return None
        except Exception as e:
            logger.error(f"Error during evaluation for {self.model_key} on split '{eval_split}': {e}", exc_info=True)
            return None
        finally:
             eval_time = time.time() - start_time
             logger.info(f"Evaluation phase for {self.model_key} on split '{eval_split}' finished in {eval_time:.2f}s")
             if eval_results is None: eval_results = {}
             eval_results['eval_time'] = eval_time

        logger.info(f"Evaluation results for {self.model_key} on '{eval_split}': {eval_results}")
        return {self.model_key: eval_results}