# models/baseline_trainer.py
import os
import logging
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Optional, Tuple

from config import MODELS_DIR, DATA_DIR # Assuming DATA_DIR is for parquet features
from utils.common import NLIModel
from utils.database import DatabaseHandler
# Import base model classes and helpers
from .baseline_base import TextBaselineModel, clean_dataset, _evaluate_model_performance
# Import specific model classes
from .logistic_tf_idf_baseline import LogisticTFIDFBaseline
from .multinomial_naive_bayes_bow_baseline import MultinomialNaiveBayesBaseline
from .svm_bow_baseline import SVMWithBagOfWords, SVMWithSyntax, SVMWithBothFeatures, load_parquet_data, \
    _handle_nan_values, SVMModel, LexicalFeatureExtractor, SyntacticFeatureExtractor, \
    CombinedFeatureExtractor  # Import SVM specifics
from .svm_hand_crafted_syntactic_features_experiment_1 import SVMHandcraftedSyntacticExperiment1

logger = logging.getLogger(__name__)

class BaselineTrainer:
    """Handles training and evaluation for various baseline models."""
    def __init__(self, model_type: str, dataset_name: str, args: object):
        """
        Initializes the trainer for a specific model type and dataset.

        Args:
            model_type (str): 'svm', 'logistic_tfidf', 'mnb_bow'.
            dataset_name (str): Name of the dataset (e.g., 'SNLI').
            args (object): Command line arguments containing model hyperparameters
                           (C, kernel, max_features, sample_size, etc.).
        """
        self.model_type = model_type
        self.dataset_name = dataset_name
        self.args = args
        self.sample_size = getattr(args, 'sample_size', None)
        self.suffix = f"sample{self.sample_size}" if self.sample_size else "full"
        self.save_dir = self._get_save_directory()
        os.makedirs(self.save_dir, exist_ok=True)
        self.db_handler = DatabaseHandler() # Needed for text baselines
        self.model: Optional[NLIModel] = None # Allow SVMModel or TextBaselineModel

    def _get_save_directory(self) -> str:
        """Determines the save directory based on the model type."""
        # Use a consistent naming scheme, possibly including dataset/suffix if needed
        base_dir = os.path.join(MODELS_DIR, f"{self.model_type}_baseline")
        # Example: Optionally add dataset/suffix subdirs if models are dataset/size specific
        # return os.path.join(base_dir, self.dataset_name, self.suffix)
        return base_dir

    def _get_model_filename_base(self) -> str:
        """Generates a base filename for saving models/extractors."""
        # Example: SNLI_logistic_tfidf_full
        return f"{self.dataset_name}_{self.model_type}_{self.suffix}"

    def _load_data(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Loads train, validation, and test data based on model type."""
        train_data, val_data, test_data = None, None, None
        logger.info(f"Loading data for {self.model_type} on {self.dataset_name} ({self.suffix})")

        if self.model_type == 'svm':
            # SVM loads precomputed parquet features
            feature_file_pattern = f"{self.dataset_name}_{{split}}_features_lexical_syntactic_{self.suffix}"
            parquet_feature_dir = os.path.join(DATA_DIR, 'parquet') # Adjust if path differs
            logger.info(f"SVM: Loading precomputed features from {parquet_feature_dir} using pattern base: {feature_file_pattern.format(split='*')}")
            try:
                train_data = load_parquet_data(self.dataset_name, 'train', feature_type=f'features_lexical_syntactic_{self.suffix}', cache_dir=parquet_feature_dir)
                train_data = _handle_nan_values(train_data, "training")
            except FileNotFoundError:
                logger.error(f"SVM training features not found for pattern: {feature_file_pattern.format(split='train')}")
            try:
                val_data = load_parquet_data(self.dataset_name, 'validation', feature_type=f'features_lexical_syntactic_{self.suffix}', cache_dir=parquet_feature_dir)
                val_data = _handle_nan_values(val_data, "validation")
            except FileNotFoundError:
                logger.warning("SVM validation features not found.")
            try:
                test_data = load_parquet_data(self.dataset_name, 'test', feature_type=f'features_lexical_syntactic_{self.suffix}', cache_dir=parquet_feature_dir)
                test_data = _handle_nan_values(test_data, "test")
            except FileNotFoundError:
                logger.warning("SVM test features not found.")

        elif self.model_type in ['logistic_tfidf', 'mnb_bow']:
            # Text baselines load raw text data
            try:
                train_data = TextBaselineModel.load_raw_text_data(self.dataset_name, 'train', self.suffix, self.db_handler)
            except Exception as e:
                 logger.error(f"Failed to load raw train data: {e}")
            try:
                val_data = TextBaselineModel.load_raw_text_data(self.dataset_name, 'validation', self.suffix, self.db_handler)
            except Exception as e:
                 logger.warning(f"Failed to load raw validation data: {e}")
            try:
                test_data = TextBaselineModel.load_raw_text_data(self.dataset_name, 'test', self.suffix, self.db_handler)
            except Exception as e:
                 logger.warning(f"Failed to load raw test data: {e}")
        else:
            logger.error(f"Unsupported model type for data loading: {self.model_type}")

        return train_data, val_data, test_data


    def _initialize_model(self) -> Optional[NLIModel]:
        """Initializes the correct model instance based on model_type and args."""
        logger.info(f"Initializing model: {self.model_type}")
        model = None
        try:
            if self.model_type == 'logistic_tfidf':
                model = LogisticTFIDFBaseline(
                    C=getattr(self.args, 'C', 1.0),
                    max_features=getattr(self.args, 'max_features', 10000),
                    # Add other relevant params like ngram_range if needed
                )
            elif self.model_type == 'mnb_bow':
                model = MultinomialNaiveBayesBaseline(
                    alpha=getattr(self.args, 'alpha', 1.0), # Example: get alpha if defined in args
                    max_features=getattr(self.args, 'max_features', 10000),
                )
            # << --- ADD THIS CASE --- >>
            elif self.model_type == 'svm_syntactic_exp1':
                 # Instantiate the specific Experiment 1 SVM model
                 model = SVMHandcraftedSyntacticExperiment1(
                      kernel=getattr(self.args, 'kernel', 'linear'),
                      C=getattr(self.args, 'C', 1.0)
                 )
            # << --- END ADDITION --- >>
            elif self.model_type == 'svm':
                # General SVM still handled later in run_training to train multiple variants
                logger.info("General 'svm' model type selected. Specific SVM variants (BoW, Syntax, Combined) will be trained.")
                return None # Return None, specific SVM models instantiated later
            else:
                logger.error(f"Unknown model type for initialization: {self.model_type}")
        except Exception as e:
            logger.error(f"Error initializing model {self.model_type}: {e}", exc_info=True)

        return model


    def run_training(self):
        """Runs the full training and evaluation pipeline."""
        train_data, val_data, test_data = self._load_data()

        if train_data is None or train_data.empty:
            logger.error("Training data could not be loaded. Aborting.")
            return None

        # --- Handle SVM Training (Multiple Models for type 'svm') ---
        if self.model_type == 'svm':
             # <<< Keep existing logic for training multiple SVM variants >>>
             logger.info("Starting SVM training for BoW, Syntax, and Combined features...")
             # ... (existing logic for training SVMWithBagOfWords, SVMWithSyntax, SVMWithBothFeatures) ...
             # ... This part remains unchanged ...
             svm_results = {}
             kernel = getattr(self.args, 'kernel', 'linear')
             C = getattr(self.args, 'C', 1.0)
             svm_save_dir = self.save_dir # Use the trainer's save dir

             if val_data is None or val_data.empty:
                  if len(train_data) < 5:
                       logger.error("Not enough training data to create a validation split for SVM.")
                       return None
                  logger.info("Splitting validation set from training data for SVM.")
                  train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42, stratify=train_data.get('label', None))

             clean_train_result = clean_dataset(train_data)
             clean_val_result = clean_dataset(val_data) if val_data is not None else None

             if not clean_train_result:
                  logger.error("Training data invalid after cleaning.")
                  return None
             if val_data is not None and not clean_val_result:
                  logger.warning("Validation data invalid after cleaning, skipping SVM validation.")
                  val_data_clean, y_val = None, None
             elif val_data is not None:
                  train_data_clean, y_train = clean_train_result
                  val_data_clean, y_val = clean_val_result
             else:
                  train_data_clean, y_train = clean_train_result
                  val_data_clean, y_val = None, None

             svm_model_configs = [
                 (SVMWithBagOfWords(kernel=kernel, C=C), "SVM_BoW"),
                 (SVMWithSyntax(kernel=kernel, C=C), "SVM_Syntax"),
                 (SVMWithBothFeatures(kernel=kernel, C=C), "SVM_Combined")
             ]

             for svm_instance, model_name_suffix in svm_model_configs:
                 logger.info(f"--- Training {model_name_suffix} ---")
                 model_filename_base = f"{self.dataset_name}_{model_name_suffix}_{self.suffix}"
                 logger.info("Extracting training features...")
                 X_train = svm_instance.extract_features(train_data_clean)
                 if X_train is None or X_train.size == 0:
                      logger.error(f"Feature extraction failed for training {model_name_suffix}. Skipping.")
                      continue

                 start_time = time.time()
                 svm_instance.train(X_train, y_train)
                 train_time = time.time() - start_time
                 logger.info(f"Training complete in {train_time:.2f}s")

                 eval_metrics = {}
                 eval_time = 0.0
                 if val_data_clean is not None and y_val is not None:
                     logger.info("Extracting validation features...")
                     X_val = svm_instance.extract_features(val_data_clean)
                     if X_val is not None and X_val.size > 0:
                          logger.info("Evaluating on validation set...")
                          eval_time, eval_metrics = _evaluate_model_performance(svm_instance, X_val, y_val)
                     else:
                          logger.warning("Feature extraction failed for validation. Skipping evaluation.")
                 else:
                      logger.info("Skipping validation evaluation.")

                 model_path = os.path.join(svm_save_dir, f"{model_filename_base}.joblib")
                 svm_instance.save(model_path)
                 svm_results[model_name_suffix] = {**eval_metrics, 'train_time': train_time, 'eval_time': eval_time}

             logger.info("Finished training all SVM variants.")
             return svm_results

        # --- Handle Text Baselines AND Specific SVM Experiment (Single Model per run) ---
        elif self.model_type in ['logistic_tfidf', 'mnb_bow', 'svm_syntactic_exp1']: # Add new type here
            # Initialize the single model instance
            self.model = self._initialize_model()
            if not self.model:
                 logger.error(f"Failed to initialize model {self.model_type}. Aborting.")
                 return None

            # Check if the model requires precomputed features (SVM) or raw text (TFIDF/BoW)
            if isinstance(self.model, SVMModel): # Check if it's an SVM type
                 logger.info(f"Handling SVM model type: {self.model_type}. Using precomputed features.")
                 # SVM loads precomputed features; data loading handled above.
                 # We need to clean the loaded feature data.
                 clean_train_result = clean_dataset(train_data)
                 if not clean_train_result:
                      logger.error("SVM Training data invalid after cleaning. Aborting.")
                      return None
                 train_data_clean, y_train = clean_train_result

                 logger.info(f"Extracting features for {self.model_type} from precomputed data...")
                 # The model's internal feature_extractor (SyntacticFeatureExtractor for Exp1)
                 # will select the correct columns from train_data_clean.
                 X_train = self.model.extract_features(train_data_clean) # Extract correct features

                 if X_train is None or X_train.shape[0] == 0:
                      logger.error("Feature extraction failed for training data. Aborting.")
                      return None

                 # Train the model
                 start_time = time.time()
                 self.model.train(X_train, y_train)
                 train_time = time.time() - start_time

                 # Evaluate on validation set
                 eval_results = {}
                 if val_data is not None and not val_data.empty:
                     clean_val_result = clean_dataset(val_data)
                     if clean_val_result:
                         val_data_clean, y_val = clean_val_result
                         logger.info("Evaluating on validation data...")
                         X_val = self.model.extract_features(val_data_clean) # Extract correct features
                         eval_time, eval_metrics = _evaluate_model_performance(self.model, X_val, y_val)
                         eval_results = {**eval_metrics, 'eval_time': eval_time}
                     else:
                         logger.warning("Validation data invalid after cleaning. Skipping validation.")
                 else:
                     logger.info("Skipping validation evaluation (no validation data).")

                 # Save model (SVMModel save handles internal state)
                 model_filename_base = self._get_model_filename_base()
                 model_path = os.path.join(self.save_dir, f"{model_filename_base}.joblib")
                 self.model.save(model_path) # Use SVMModel's save

                 self.run_evaluation(test_data)
                 return {**eval_results, 'train_time': train_time}

            elif isinstance(self.model, TextBaselineModel): # TFIDF or BoW
                 logger.info(f"Handling Text Baseline model type: {self.model_type}. Using raw text data.")
                 # Clean raw training data
                 clean_train_result = clean_dataset(train_data)
                 if not clean_train_result:
                      logger.error("Raw training data invalid after cleaning. Aborting.")
                      return None
                 train_data_clean, y_train = clean_train_result

                 # Fit extractor and transform training data
                 logger.info(f"Fitting feature extractor ({self.model.extractor.__class__.__name__}) on training data...")
                 self.model.extractor.fit(train_data_clean)
                 logger.info("Transforming training data...")
                 X_train = self.model.extract_features(train_data_clean)

                 if X_train is None or X_train.shape[0] == 0:
                      logger.error("Feature extraction failed for training data. Aborting.")
                      return None

                 # Train the model
                 start_time = time.time()
                 self.model.train(X_train, y_train)
                 train_time = time.time() - start_time

                 # Evaluate on validation set
                 eval_results = {}
                 if val_data is not None and not val_data.empty:
                     clean_val_result = clean_dataset(val_data)
                     if clean_val_result:
                         val_data_clean, y_val = clean_val_result
                         logger.info("Evaluating on validation data...")
                         X_val = self.model.extract_features(val_data_clean)
                         eval_time, eval_metrics = _evaluate_model_performance(self.model, X_val, y_val)
                         eval_results = {**eval_metrics, 'eval_time': eval_time}
                     else:
                         logger.warning("Validation data invalid after cleaning. Skipping validation.")
                 else:
                     logger.info("Skipping validation evaluation (no validation data).")

                 # Save model and extractor using TextBaselineModel's save
                 model_filename_base = self._get_model_filename_base()
                 self.model.save(self.save_dir, model_filename_base)

                 self.run_evaluation(test_data)
                 return {**eval_results, 'train_time': train_time}
            else:
                 logger.error(f"Initialized model is not a recognized baseline type (SVMModel or TextBaselineModel).")
                 return None
        else:
             logger.error(f"Unsupported model type in run_training: {self.model_type}")
             return None


    def run_evaluation(self, eval_data: Optional[pd.DataFrame], model_type: Optional[str] = None):
        """Evaluates the trained model on the provided data (e.g., test set)."""
        model_to_eval_type = model_type or self.model_type
        logger.info(f"Starting evaluation for model type: {model_to_eval_type} on dataset: {self.dataset_name}")
        # ... (data loading/validation check) ...
        if eval_data is None or eval_data.empty:
             logger.warning(f"No evaluation data provided for {self.dataset_name}/{model_to_eval_type}. Skipping evaluation.")
             return {}

        # Load the appropriate model(s)
        if model_to_eval_type == 'svm':
             # <<< Existing logic for evaluating multiple SVM variants >>>
             # ... (this part remains unchanged) ...
            svm_results = {}
            svm_model_configs = [
                (SVMWithBagOfWords, "SVM_BoW"),
                (SVMWithSyntax, "SVM_Syntax"),
                (SVMWithBothFeatures, "SVM_Combined")
            ]
            clean_eval_result = clean_dataset(eval_data)
            if not clean_eval_result:
                 logger.error("Evaluation data invalid after cleaning. Cannot evaluate SVMs.")
                 return {}
            eval_data_clean, y_eval = clean_eval_result

            for model_cls, model_name_suffix in svm_model_configs:
                model_filename_base = f"{self.dataset_name}_{model_name_suffix}_{self.suffix}"
                model_path = os.path.join(self.save_dir, f"{model_filename_base}.joblib")
                try:
                    logger.info(f"Loading SVM model: {model_path}")
                    # Determine extractor based on model_cls
                    extractor_instance = None
                    if model_cls == SVMWithBagOfWords: extractor_instance = LexicalFeatureExtractor()
                    elif model_cls == SVMWithSyntax: extractor_instance = SyntacticFeatureExtractor()
                    elif model_cls == SVMWithBothFeatures: extractor_instance = CombinedFeatureExtractor()

                    loaded_svm_model = model_cls.load(model_path, feature_extractor=extractor_instance)
                    logger.info("Extracting features for evaluation...")
                    X_eval = loaded_svm_model.extract_features(eval_data_clean)
                    if X_eval is not None and X_eval.shape[0] > 0:
                         _, eval_metrics = _evaluate_model_performance(loaded_svm_model, X_eval, y_eval)
                         svm_results[model_name_suffix] = eval_metrics
                    else:
                         logger.warning(f"Feature extraction failed for evaluation data on {model_name_suffix}")
                except FileNotFoundError:
                    logger.error(f"SVM model file not found, cannot evaluate: {model_path}")
                except Exception as e:
                    logger.error(f"Error during SVM evaluation for {model_name_suffix}: {e}", exc_info=True)
            return svm_results

        # <<< ADD svm_syntactic_exp1 to this block >>>
        elif model_to_eval_type in ['logistic_tfidf', 'mnb_bow', 'svm_syntactic_exp1']:
            model_loaded_for_eval = False
            if self.model is None or self.model.__class__.__name__.lower() != model_to_eval_type.replace('_', ''): # Check type match
                 model_filename_base = f"{self.dataset_name}_{model_to_eval_type}_{self.suffix}" # Use correct type in filename
                 logger.info(f"Loading {model_to_eval_type} model from {self.save_dir} for evaluation...")
                 try:
                      # Load the specific model class
                      if model_to_eval_type == 'logistic_tfidf':
                           self.model = LogisticTFIDFBaseline.load(self.save_dir, model_filename_base)
                      elif model_to_eval_type == 'mnb_bow':
                           self.model = MultinomialNaiveBayesBaseline.load(self.save_dir, model_filename_base)
                      elif model_to_eval_type == 'svm_syntactic_exp1':
                           # Pass the correct extractor instance needed by SVMModel.load
                           self.model = SVMHandcraftedSyntacticExperiment1.load(
                                os.path.join(self.save_dir, f"{model_filename_base}.joblib"),
                                feature_extractor=SyntacticFeatureExtractor()
                           )
                      model_loaded_for_eval = True
                 except FileNotFoundError:
                      logger.error(f"Model {model_filename_base} not found in {self.save_dir}. Cannot evaluate.")
                      return {}
                 except Exception as e:
                      logger.error(f"Error loading model {model_filename_base}: {e}", exc_info=True)
                      return {}

            if not model_loaded_for_eval and self.model is None:
                 logger.error(f"Model for {model_to_eval_type} is not loaded and not found. Cannot evaluate.")
                 return {}

            # --- Proceed with evaluation using self.model ---
            # Check if it's an SVM type or TextBaseline type for feature handling
            clean_eval_result = clean_dataset(eval_data)
            if not clean_eval_result:
                 logger.error("Evaluation data invalid after cleaning. Cannot evaluate.")
                 return {}
            eval_data_clean, y_eval = clean_eval_result

            logger.info("Extracting features for evaluation...")
            # Feature extraction depends on model type
            if isinstance(self.model, SVMModel):
                 X_eval = self.model.extract_features(eval_data_clean) # SVM extracts from precomputed features
            elif isinstance(self.model, TextBaselineModel):
                 if not self.model.extractor.is_fitted: # Check if extractor loaded correctly
                      logger.error("Feature extractor is not fitted/loaded. Cannot extract features for evaluation.")
                      return {}
                 X_eval = self.model.extract_features(eval_data_clean) # Text models extract from raw text
            else:
                 logger.error(f"Loaded model instance type {type(self.model)} not recognized for feature extraction.")
                 return {}

            if X_eval is None or X_eval.shape[0] == 0:
                 logger.error("Feature extraction failed for evaluation data.")
                 return {}

            _, eval_metrics = _evaluate_model_performance(self.model, X_eval, y_eval)
            return eval_metrics
        else:
            logger.error(f"Unsupported model type for evaluation: {model_to_eval_type}")
            return {}