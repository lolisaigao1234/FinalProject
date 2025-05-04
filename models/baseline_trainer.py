# File: IS567FP/models/baseline_trainer.py (Refactored)
import os
import logging
import time
from typing import Optional, Dict, Any

# Configuration and Utilities
from config import MODELS_DIR
from utils.common import NLIModel # Base class/interface (ensure models adhere to it)

# Import the registry and ALL model classes it refers to
from . import MODEL_REGISTRY

logger = logging.getLogger(__name__)

class BaselineTrainer:
    """
    Handles orchestration of training and evaluation for models defined in MODEL_REGISTRY.
    Relies on individual model classes to handle their specific data loading and logic.
    """
    def __init__(self, model_type: str, dataset_name: str, args: object):
        """
        Initializes the trainer for a specific model type and dataset.

        Args:
            model_type (str): Model identifier key from MODEL_REGISTRY (e.g., 'baseline-1', 'experiment-3').
            dataset_name (str): Name of the dataset (e.g., 'SNLI').
            args (object): Command line arguments containing hyperparameters.
        """
        self.model_key = model_type # Use 'key' to emphasize it's from the registry
        self.dataset_name = dataset_name
        self.args = args
        self.sample_size = getattr(args, 'sample_size', None)
        self.suffix = f"sample{self.sample_size}" if self.sample_size else "full"

        # Get the corresponding model class from the registry
        self.model_cls = MODEL_REGISTRY.get(self.model_key)
        if not self.model_cls:
            raise ValueError(f"Model type key '{self.model_key}' not found in MODEL_REGISTRY.")

        self.save_dir = self._get_save_directory()
        os.makedirs(self.save_dir, exist_ok=True)
        # Keep db_handler if models need it passed during init or methods
        # self.db_handler = DatabaseHandler()
        self.model: Optional[NLIModel] = None # To hold the instantiated model

        logger.info(f"Initialized Trainer for model key: '{self.model_key}', class: {self.model_cls.__name__}, dataset: {self.dataset_name}, suffix: {self.suffix}")

    def _get_save_directory(self) -> str:
        """Determines the save directory based on model key."""
        # Centralized directory, subfoldered by dataset/model_key/suffix
        base_dir = os.path.join(MODELS_DIR, 'baseline_models', self.dataset_name, self.model_key, self.suffix)
        return base_dir

    def _get_model_filename_base(self) -> str:
        """Generates a base filename for saving models/artifacts."""
        # Consistent naming using the model key
        return f"{self.dataset_name}_{self.model_key}_{self.suffix}"

    def _initialize_model(self) -> Optional[NLIModel]:
        """Initializes the model instance using args."""
        logger.info(f"Initializing model instance for {self.model_key} ({self.model_cls.__name__})")
        model_instance: Optional[NLIModel] = None

        # Prepare hyperparameters relevant to this model class
        # This requires knowing which args each model might need.
        # A more robust way is to inspect the model_cls.__init__ signature,
        # but a simpler way is to pass all args or select known ones.
        hyperparams = {
            'C': getattr(self.args, 'C', 1.0),
            'max_features': getattr(self.args, 'max_features', 10000), # For TFIDF/BoW based models
            'tfidf_max_features': getattr(self.args, 'max_features', 10000), # Explicit name if models use it
            'bow_max_features': getattr(self.args, 'max_features', 10000), # Explicit name if models use it
            'alpha': getattr(self.args, 'alpha', 1.0), # For MNB
            'n_estimators': getattr(self.args, 'n_estimators', 100), # For RF, GB
            'max_depth': getattr(self.args, 'max_depth', None), # For RF, GB, DT
            'learning_rate': getattr(self.args, 'learning_rate', 0.1), # For GB (check if others use it)
            'n_neighbors': getattr(self.args, 'n_neighbors', 5), # For KNN
            'max_iter': getattr(self.args, 'max_iter', 1000), # For Logistic Regression
            'random_state': 42, # Common random state
             # Add other relevant hyperparameters from args if needed by models
             # e.g., kernel was removed as it was SVM specific. DT/KNN/RF don't use it.
        }
        # Add args/suffix/db_handler if model constructors need them
        # hyperparams['args'] = self.args
        # hyperparams['suffix'] = self.suffix
        # hyperparams['db_handler'] = self.db_handler

        try:
            # Attempt to instantiate the model class with relevant args
            # Option 1: Pass all args (simpler, models ignore extras)
            # model_instance = self.model_cls(**vars(self.args))

            # Option 2: Pass selected hyperparams (cleaner, requires knowing what models need)
            # Filter hyperparams based on model needs - this is complex.
            # Simplest approach for now: instantiate with common params.
            # Models should ideally accept **kwargs or specific relevant params.
            # Assuming models take relevant args from the hyperparams dict:
            # This might require adjusting model __init__ signatures.
            # For now, let's try instantiating with no args, assuming defaults are handled in the class
            # OR that the class __init__ fetches from a shared config/args itself.
            # A common pattern is passing the `args` object directly:
            logger.warning("Attempting to initialize model with the 'args' object. Ensure model __init__ handles this.")
            model_instance = self.model_cls(args=self.args) # Pass the whole args object

            # --- OR --- If models take specific params:
            # Example: If KNN takes n_neighbors and RF takes n_estimators, max_depth:
            # if self.model_cls == KnnBowSyntacticExperiment2:
            #     model_instance = self.model_cls(n_neighbors=hyperparams['n_neighbors'])
            # elif self.model_cls == RandomForestBowSyntacticExperiment5:
            #     model_instance = self.model_cls(n_estimators=hyperparams['n_estimators'], max_depth=hyperparams['max_depth'], random_state=hyperparams['random_state'])
            # # ... This becomes complex quickly. Passing `args` is often easier.

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
        Runs the training pipeline: Initialize, Train, Save.
        Assumes the model's train method handles data loading.
        """
        results = {}
        self.model = self._initialize_model()

        if not self.model:
            logger.error(f"Failed to initialize model {self.model_key}. Aborting training.")
            return None

        logger.info(f"--- Starting training for {self.model_key} on {self.dataset_name} ({self.suffix}) ---")
        start_time = time.time()
        try:
            # Assume train method takes dataset info and handles loading/processing
            # Adjust signature based on actual model implementations
            train_results = self.model.train(
                train_dataset=self.dataset_name, train_split='train', train_suffix=self.suffix,
                val_dataset=self.dataset_name, val_split='validation', val_suffix=self.suffix
                # Pass other args if needed, e.g., db_handler=self.db_handler
            )
            # Store whatever results the train method returns (e.g., metrics, timings)
            results[self.model_key] = train_results if train_results else {}

        except NotImplementedError:
             logger.error(f"Model class {self.model_cls.__name__} does not implement the 'train' method.")
             return None
        except Exception as e:
            logger.error(f"Error during training for {self.model_key}: {e}", exc_info=True)
            return None
        finally:
            train_time = time.time() - start_time
            logger.info(f"Training phase for {self.model_key} finished in {train_time:.2f}s")
            if self.model_key in results:
                results[self.model_key]['train_time'] = train_time # Add train time to results

        # --- Save the trained model ---
        logger.info(f"Saving model {self.model_key}...")
        model_filename_base = self._get_model_filename_base()
        save_path_base = os.path.join(self.save_dir, model_filename_base)
        try:
            # Assume a save method exists that takes a base path
            self.model.save(save_path_base)
            logger.info(f"Model {self.model_key} saved with base path: {save_path_base}")
        except NotImplementedError:
             logger.error(f"Model class {self.model_cls.__name__} does not implement the 'save' method.")
             # Continue, but model won't be saved
        except Exception as e:
            logger.error(f"Error saving model {self.model_key}: {e}", exc_info=True)

        return results


    def run_evaluation(self, eval_split: str = 'test') -> Optional[Dict[str, Any]]:
        """
        Runs the evaluation pipeline: Load Model, Evaluate.
        Assumes the model's evaluate method handles data loading for the specified split.

        Args:
            eval_split (str): The data split to evaluate on (e.g., 'test', 'validation').

        Returns:
            dict: Evaluation metrics, or None if evaluation fails.
        """
        logger.info(f"--- Starting evaluation for {self.model_key} on {self.dataset_name} split '{eval_split}' ({self.suffix}) ---")

        # --- Load the saved model ---
        model_filename_base = self._get_model_filename_base()
        load_path_base = os.path.join(self.save_dir, model_filename_base)
        loaded_model: Optional[NLIModel] = None
        try:
            # Assume a static load method exists that takes the base path
            logger.info(f"Loading model {self.model_key} from base path: {load_path_base}")
            loaded_model = self.model_cls.load(load_path_base)
            if loaded_model is None: raise FileNotFoundError # If load returns None on failure
        except FileNotFoundError:
            logger.error(f"Model artifacts not found for {self.model_key} at base path {load_path_base}. Cannot evaluate.")
            return None
        except NotImplementedError:
             logger.error(f"Model class {self.model_cls.__name__} does not implement the 'load' method.")
             return None
        except Exception as e:
            logger.error(f"Error loading model {self.model_key} from {load_path_base}: {e}", exc_info=True)
            return None

        if not isinstance(loaded_model, NLIModel):
             logger.error(f"Loaded object for {self.model_key} is not an instance of NLIModel.")
             return None

        # --- Perform Evaluation ---
        logger.info(f"Evaluating loaded model {self.model_key}...")
        eval_results = {}
        start_time = time.time()
        try:
            # Assume evaluate method takes dataset info + split and handles loading/processing
            eval_metrics = loaded_model.evaluate(
                dataset_name=self.dataset_name,
                split=eval_split,
                suffix=self.suffix
                # Pass other args if needed, e.g., db_handler=self.db_handler
            )
            eval_results = eval_metrics if eval_metrics else {}

        except NotImplementedError:
             logger.error(f"Model class {self.model_cls.__name__} does not implement the 'evaluate' method.")
             return None
        except Exception as e:
            logger.error(f"Error during evaluation for {self.model_key} on split '{eval_split}': {e}", exc_info=True)
            return None # Indicate evaluation failure
        finally:
             eval_time = time.time() - start_time
             logger.info(f"Evaluation phase for {self.model_key} on split '{eval_split}' finished in {eval_time:.2f}s")
             eval_results['eval_time'] = eval_time # Add eval time to results

        logger.info(f"Evaluation results for {self.model_key} on '{eval_split}': {eval_results}")
        return {self.model_key: eval_results}