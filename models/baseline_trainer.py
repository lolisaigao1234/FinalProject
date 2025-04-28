# Modify: IS567FP/models/baseline_trainer.py
import os
import logging
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Optional, Tuple, Dict, Any, Type # Added Type

from config import MODELS_DIR, DATA_DIR, CACHE_DIR  # Assuming DATA_DIR is for parquet features
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
# Import Experiment 2 model
from .svm_bow_hand_crafted_syntactic_features_experiment_2 import SVMBowHandCraftedSyntacticExperiment2
# Import Experiment 3 model
from .logistic_tfidf_syntactic_experiment_3 import LogisticTFIDFSyntacticExperiment3


logger = logging.getLogger(__name__)

class BaselineTrainer:
    """Handles training and evaluation for various baseline models."""
    def __init__(self, model_type: str, dataset_name: str, args: object):
        """
        Initializes the trainer for a specific model type and dataset.

        Args:
            model_type (str): 'svm', 'logistic_tfidf', 'mnb_bow', etc.
            dataset_name (str): Name of the dataset (e.g., 'SNLI').
            args (object): Command line arguments containing model hyperparameters
                           (C, kernel, max_features, sample_size, alpha, etc.).
        """
        self.model_type = model_type
        self.dataset_name = dataset_name
        self.args = args
        self.sample_size = getattr(args, 'sample_size', None)
        self.suffix = f"sample{self.sample_size}" if self.sample_size else "full"
        self.save_dir = self._get_save_directory()
        os.makedirs(self.save_dir, exist_ok=True)
        self.db_handler = DatabaseHandler() # Needed for text baselines and Exp3
        self.model: Optional[NLIModel] = None # Allow SVMModel or TextBaselineModel or Exp3 model

    def _get_save_directory(self) -> str:
        """Determines the save directory based on the model type."""
        # Consolidate saving under 'baseline_models' or similar, organized by dataset/model/suffix
        base_dir = os.path.join(MODELS_DIR, 'baseline_models', self.dataset_name, self.model_type, self.suffix)
        return base_dir

    def _get_model_filename_base(self) -> str:
        """Generates a base filename for saving models/extractors/pipelines."""
        # Keep consistent: dataset_modeltype_suffix
        return f"{self.dataset_name}_{self.model_type}_{self.suffix}"

    def _load_data(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Loads train, validation, and test data based on model type."""
        train_data, val_data, test_data = None, None, None
        logger.info(f"Loading data for {self.model_type} on {self.dataset_name} ({self.suffix})")

        # Determine the feature type pattern needed based on the model
        # SVM models (including Exp1, Exp2) need the precomputed lexical+syntactic features
        if self.model_type.startswith('svm'):
             # Define the specific feature file name base expected
             feature_type_base = f'features_lexical_syntactic_{self.suffix}'
             parquet_feature_dir = os.path.join(CACHE_DIR, 'parquet') # Use CACHE_DIR as base for parquet

             logger.info(f"SVM family: Loading precomputed features from {parquet_feature_dir} using base: {feature_type_base}")

             # Construct the full expected filename for each split
             # The db_handler expects the full table name which includes dataset and split
             train_table_name = f"{self.dataset_name}_train_{feature_type_base}"
             val_table_name = f"{self.dataset_name}_validation_{feature_type_base}"
             test_table_name = f"{self.dataset_name}_test_{feature_type_base}"

             try:
                  train_data = self.db_handler.load_dataframe(self.dataset_name, 'train', train_table_name)
                  if not train_data.empty: train_data = _handle_nan_values(train_data, "training")
                  else: logger.error(f"Loaded empty dataframe for train features: {train_table_name}")
             except Exception as e:
                  logger.error(f"Failed to load SVM train features ({train_table_name}): {e}", exc_info=True)

             try:
                  val_data = self.db_handler.load_dataframe(self.dataset_name, 'validation', val_table_name)
                  if not val_data.empty: val_data = _handle_nan_values(val_data, "validation")
                  else: logger.warning(f"Loaded empty dataframe for validation features: {val_table_name}")
             except Exception as e:
                  logger.warning(f"Could not load SVM validation features ({val_table_name}): {e}")

             try:
                  test_data = self.db_handler.load_dataframe(self.dataset_name, 'test', test_table_name)
                  if not test_data.empty: test_data = _handle_nan_values(test_data, "test")
                  else: logger.warning(f"Loaded empty dataframe for test features: {test_table_name}")
             except Exception as e:
                  logger.warning(f"Could not load SVM test features ({test_table_name}): {e}")

        # logistic_tfidf, mnb_bow load raw text via helper method
        elif self.model_type in ['logistic_tfidf', 'mnb_bow']:
            try:
                train_data = TextBaselineModel.load_raw_text_data(self.dataset_name, 'train', self.suffix, self.db_handler)
            except Exception as e:
                 logger.error(f"Failed to load raw train data for {self.model_type}: {e}", exc_info=True)
            try:
                val_data = TextBaselineModel.load_raw_text_data(self.dataset_name, 'validation', self.suffix, self.db_handler)
            except Exception as e:
                 logger.warning(f"Failed to load raw validation data for {self.model_type}: {e}")
            try:
                test_data = TextBaselineModel.load_raw_text_data(self.dataset_name, 'test', self.suffix, self.db_handler)
            except Exception as e:
                 logger.warning(f"Failed to load raw test data for {self.model_type}: {e}")

        # Experiment 3 handles its own complex data loading internally
        elif self.model_type == 'logistic_tfidf_syntactic_exp3':
             logger.info("Experiment 3 selected. Data loading will be handled internally by the model class.")
             # No data loading needed here for Exp3, return placeholders or signal to model
             # Returning None signals the model needs to load its own data based on info passed.
             return None, None, None

        else:
            logger.error(f"Unsupported model type for data loading: {self.model_type}")

        return train_data, val_data, test_data


    def _initialize_model(self) -> Optional[NLIModel]:
        """Initializes the correct model instance based on model_type and args."""
        logger.info(f"Initializing model: {self.model_type}")
        model: Optional[NLIModel] = None
        kernel = getattr(self.args, 'kernel', 'linear')
        C = getattr(self.args, 'C', 1.0)
        max_features = getattr(self.args, 'max_features', 10000)
        alpha = getattr(self.args, 'alpha', 1.0)

        try:
            if self.model_type == 'logistic_tfidf':
                model = LogisticTFIDFBaseline(C=C, max_features=max_features)
            elif self.model_type == 'mnb_bow':
                model = MultinomialNaiveBayesBaseline(alpha=alpha, max_features=max_features)
            elif self.model_type == 'svm_syntactic_exp1':
                 model = SVMHandcraftedSyntacticExperiment1(kernel=kernel, C=C)
            elif self.model_type == 'svm_bow_syntactic_exp2':
                 model = SVMBowHandCraftedSyntacticExperiment2(kernel=kernel, C=C)
            # <<< ADDED Exp3 Initialization >>>
            elif self.model_type == 'logistic_tfidf_syntactic_exp3':
                 model = LogisticTFIDFSyntacticExperiment3(
                      C=C,
                      tfidf_max_features=max_features # Reuse max_features for TF-IDF part
                 )
            # --------------------------------
            elif self.model_type == 'svm':
                # General SVM handled later
                logger.info("General 'svm' model type selected. Specific variants will be trained.")
                return None # Signal that multiple models will be handled
            else:
                logger.error(f"Unknown model type for initialization: {self.model_type}")
        except Exception as e:
            logger.error(f"Error initializing model {self.model_type}: {e}", exc_info=True)

        return model


    def run_training(self):
        """Runs the full training and evaluation pipeline."""
        results = {}

        # --- Handle Experiment 3 (loads its own data) ---
        if self.model_type == 'logistic_tfidf_syntactic_exp3':
            logger.info("Handling training for Experiment 3 (Logistic TFIDF + Syntactic)...")
            self.model = self._initialize_model()
            if not self.model or not isinstance(self.model, LogisticTFIDFSyntacticExperiment3):
                logger.error("Failed to initialize Experiment 3 model.")
                return None

            # Pass dataset info, model handles internal loading/prep
            try:
                 train_results = self.model.train(
                      train_dataset=self.dataset_name, train_split='train', train_suffix=self.suffix,
                      val_dataset=self.dataset_name, val_split='validation', val_suffix=self.suffix
                 )
                 results[self.model_type] = train_results
                 # Save the trained Exp3 model and its pipeline
                 model_filename_base = self._get_model_filename_base()
                 save_path_base = os.path.join(self.save_dir, model_filename_base)
                 self.model.save(save_path_base)

                 # Trigger evaluation on test set (model loads test data internally)
                 self.run_evaluation(None) # Pass None for data, model handles loading

            except Exception as e:
                 logger.error(f"Error during Experiment 3 training: {e}", exc_info=True)
                 return None
            return results

        # --- Load data for other models ---
        train_data, val_data, test_data = self._load_data()

        if train_data is None or train_data.empty:
            logger.error("Training data could not be loaded. Aborting.")
            return None

        # --- Handle SVM Training (Multiple Models for type 'svm') ---
        if self.model_type == 'svm':
             # ... (Keep existing SVM variant training logic as before) ...
             logger.info("Starting SVM training for BoW, Syntax, and Combined features...")
             svm_results = {}
             kernel = getattr(self.args, 'kernel', 'linear')
             C = getattr(self.args, 'C', 1.0)
             svm_save_dir = self.save_dir # Use the trainer's save dir

             # Prepare validation data
             if val_data is None or val_data.empty:
                  if len(train_data) < 5:
                       logger.error("Not enough training data to create a validation split for SVM.")
                       return None
                  logger.info("Splitting validation set from training data for SVM.")
                  train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42, stratify=train_data.get('label', None))

             # Clean dataframes
             clean_train_result = clean_dataset(train_data)
             clean_val_result = clean_dataset(val_data) if val_data is not None else None

             if not clean_train_result: logger.error("Training data invalid after cleaning."); return None
             train_data_clean, y_train = clean_train_result
             val_data_clean, y_val = clean_val_result if clean_val_result else (None, None)

             # Define SVM variants
             svm_model_configs = [
                 (SVMWithBagOfWords(kernel=kernel, C=C), "SVM_BoW", LexicalFeatureExtractor()),
                 (SVMWithSyntax(kernel=kernel, C=C), "SVM_Syntax", SyntacticFeatureExtractor()),
                 (SVMWithBothFeatures(kernel=kernel, C=C), "SVM_Combined", CombinedFeatureExtractor())
             ]

             for svm_instance, model_name_suffix, extractor in svm_model_configs:
                 logger.info(f"--- Training {model_name_suffix} ---")
                 model_filename_base = f"{self.dataset_name}_{model_name_suffix}_{self.suffix}"
                 svm_instance.feature_extractor = extractor # Ensure correct extractor

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
             self.run_evaluation(test_data, model_type='svm') # Evaluate all SVM variants on test data
             return svm_results


        # --- Handle Text Baselines AND Specific SVM Experiments (Single Model per run) ---
        # <<< MODIFIED THIS CONDITION (removed Exp3) >>>
        elif self.model_type in ['logistic_tfidf', 'mnb_bow', 'svm_syntactic_exp1', 'svm_bow_syntactic_exp2']:
            # Initialize the single model instance
            self.model = self._initialize_model()
            if not self.model:
                 logger.error(f"Failed to initialize model {self.model_type}. Aborting.")
                 return None

            # Branch based on whether it's an SVM type or a TextBaseline type
            if isinstance(self.model, SVMModel): # Covers Exp1 and Exp2
                 logger.info(f"Handling SVM model type: {self.model_type}. Using precomputed features.")
                 clean_train_result = clean_dataset(train_data)
                 if not clean_train_result:
                      logger.error("SVM Training data invalid after cleaning. Aborting.")
                      return None
                 train_data_clean, y_train = clean_train_result

                 logger.info(f"Extracting features for {self.model_type} from precomputed data...")
                 X_train = self.model.extract_features(train_data_clean)

                 if X_train is None or (hasattr(X_train, 'shape') and X_train.shape[0] == 0) or (not hasattr(X_train, 'shape') and len(X_train) == 0):
                     logger.error("Feature extraction failed for training data (returned None or empty). Aborting.")
                     return None


                 start_time = time.time()
                 self.model.train(X_train, y_train)
                 train_time = time.time() - start_time
                 logger.info(f"Training complete in {train_time:.2f}s")

                 eval_results = {}
                 eval_time = 0.0
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

                 model_filename_base = self._get_model_filename_base()
                 model_path = os.path.join(self.save_dir, f"{model_filename_base}.joblib")
                 self.model.save(model_path)

                 results[self.model_type] = {**eval_results, 'train_time': train_time}
                 self.run_evaluation(test_data) # Evaluate this specific model on test data


            elif isinstance(self.model, TextBaselineModel): # TFIDF or BoW
                 # ... (Keep existing TextBaseline training logic as before) ...
                 logger.info(f"Handling Text Baseline model type: {self.model_type}. Using raw text data.")
                 clean_train_result = clean_dataset(train_data)
                 if not clean_train_result:
                      logger.error("Raw training data invalid after cleaning. Aborting.")
                      return None
                 train_data_clean, y_train = clean_train_result

                 logger.info(f"Fitting feature extractor ({self.model.extractor.__class__.__name__}) on training data...")
                 self.model.extractor.fit(train_data_clean)
                 logger.info("Transforming training data...")
                 X_train = self.model.extract_features(train_data_clean)

                 if X_train is None or X_train.shape[0] == 0:
                      logger.error("Feature extraction failed for training data. Aborting.")
                      return None

                 start_time = time.time()
                 self.model.train(X_train, y_train)
                 train_time = time.time() - start_time
                 logger.info(f"Training complete in {train_time:.2f}s")

                 eval_results = {}
                 eval_time = 0.0
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

                 model_filename_base = self._get_model_filename_base()
                 self.model.save(self.save_dir, model_filename_base) # Pass dir and base name

                 results[self.model_type] = {**eval_results, 'train_time': train_time}
                 self.run_evaluation(test_data) # Evaluate this specific model on test data
            else:
                 logger.error(f"Initialized model is not a recognized baseline type (SVMModel or TextBaselineModel).")
                 return None
        else:
             logger.error(f"Unsupported model type in run_training: {self.model_type}")
             return None

        return results


    def run_evaluation(self, eval_data: Optional[pd.DataFrame], model_type: Optional[str] = None):
        """Evaluates the trained model(s) on the provided data (e.g., test set)."""
        model_to_eval_type = model_type or self.model_type
        eval_dataset_name = self.dataset_name # Assuming evaluation is on the same dataset for now
        logger.info(f"Starting evaluation for model type: {model_to_eval_type} on dataset: {eval_dataset_name} (Suffix: {self.suffix})")

        all_eval_metrics = {}

        # --- Special Handling for 'svm' type (evaluate all variants) ---
        if model_to_eval_type == 'svm':
            if eval_data is None or eval_data.empty:
                logger.warning(f"No evaluation data provided for SVM variants. Skipping evaluation.")
                return {}
            svm_variants_to_eval = [
                (SVMWithBagOfWords, "SVM_BoW", LexicalFeatureExtractor()),
                (SVMWithSyntax, "SVM_Syntax", SyntacticFeatureExtractor()),
                (SVMWithBothFeatures, "SVM_Combined", CombinedFeatureExtractor())
            ]
            clean_eval_result = clean_dataset(eval_data)
            if not clean_eval_result:
                 logger.error("Evaluation data invalid after cleaning. Cannot evaluate SVMs.")
                 return {}
            eval_data_clean, y_eval = clean_eval_result

            for model_cls, model_name_suffix, extractor_instance in svm_variants_to_eval:
                # Construct the specific filename base for this SVM variant
                svm_variant_filename_base = f"{eval_dataset_name}_{model_name_suffix}_{self.suffix}"
                # Note: We need the save_dir where *these specific variants* were saved during the 'svm' training run.
                # Assuming the main 'svm' run saved them in the trainer's self.save_dir.
                model_path = os.path.join(self.save_dir, f"{svm_variant_filename_base}.joblib")
                try:
                    logger.info(f"Loading SVM model for evaluation: {model_path}")
                    # Pass the correct extractor instance when loading
                    loaded_svm_model = model_cls.load(model_path, feature_extractor=extractor_instance)
                    logger.info("Extracting features for evaluation...")
                    X_eval = loaded_svm_model.extract_features(eval_data_clean)
                    if X_eval is not None and X_eval.shape[0] > 0:
                         _, eval_metrics = _evaluate_model_performance(loaded_svm_model, X_eval, y_eval)
                         all_eval_metrics[model_name_suffix] = eval_metrics # Store metrics keyed by variant name
                         logger.info(f"Evaluation results for {model_name_suffix}: {eval_metrics}")
                    else:
                         logger.warning(f"Feature extraction failed for evaluation data on {model_name_suffix}")
                except FileNotFoundError:
                    logger.error(f"SVM model file not found, cannot evaluate: {model_path}")
                except Exception as e:
                    logger.error(f"Error during SVM evaluation for {model_name_suffix}: {e}", exc_info=True)
            return all_eval_metrics

        # --- Handling for other specific model types ---
        # <<< MODIFIED THIS CONDITION (Added Exp3) >>>
        elif model_to_eval_type in ['logistic_tfidf', 'mnb_bow', 'svm_syntactic_exp1', 'svm_bow_syntactic_exp2', 'logistic_tfidf_syntactic_exp3']:
            # Load the single specified model
            model_filename_base = f"{eval_dataset_name}_{model_to_eval_type}_{self.suffix}"
            # Construct full base path for loading (model + pipeline/extractor + metadata)
            load_path_base = os.path.join(self.save_dir, model_filename_base)
            loaded_model: Optional[NLIModel] = None
            model_cls: Optional[Type[NLIModel]] = None # To hold the class type for loading

            try:
                logger.info(f"Loading {model_to_eval_type} model from base path {load_path_base} for evaluation...")
                # Determine the correct class to use for loading
                if model_to_eval_type == 'logistic_tfidf': model_cls = LogisticTFIDFBaseline
                elif model_to_eval_type == 'mnb_bow': model_cls = MultinomialNaiveBayesBaseline
                elif model_to_eval_type == 'svm_syntactic_exp1': model_cls = SVMHandcraftedSyntacticExperiment1
                elif model_to_eval_type == 'svm_bow_syntactic_exp2': model_cls = SVMBowHandCraftedSyntacticExperiment2
                elif model_to_eval_type == 'logistic_tfidf_syntactic_exp3': model_cls = LogisticTFIDFSyntacticExperiment3
                else: raise ValueError("Should not happen, model type check failed.")

                # Call the correct load method (might need adjustments based on model class)
                if issubclass(model_cls, TextBaselineModel):
                     loaded_model = model_cls.load(self.save_dir, model_filename_base)
                elif issubclass(model_cls, SVMModel): # Covers Exp1, Exp2
                     # SVM load needs extractor type hint, determine from class
                     extractor = None
                     if model_cls == SVMHandcraftedSyntacticExperiment1: extractor = SyntacticFeatureExtractor()
                     elif model_cls == SVMBowHandCraftedSyntacticExperiment2: extractor = CombinedFeatureExtractor()
                     # Construct the specific joblib path for SVM
                     svm_model_path = os.path.join(self.save_dir, f"{model_filename_base}.joblib")
                     loaded_model = model_cls.load(svm_model_path, extractor)
                elif model_cls == LogisticTFIDFSyntacticExperiment3:
                     loaded_model = model_cls.load(load_path_base) # Exp3 loads using base path
                else:
                     raise TypeError(f"Don't know how to load model type {model_to_eval_type}")

                if loaded_model is None: raise FileNotFoundError # Trigger except if load failed

            except FileNotFoundError:
                 logger.error(f"Model artifacts for {model_filename_base} not found at {load_path_base}. Cannot evaluate.")
                 return {}
            except Exception as e:
                 logger.error(f"Error loading model {model_filename_base}: {e}", exc_info=True)
                 return {}

            # --- Perform Evaluation ---
            # Experiment 3 handles its own data loading for evaluation
            if model_to_eval_type == 'logistic_tfidf_syntactic_exp3':
                 logger.info("Triggering internal evaluation for Experiment 3...")
                 try:
                      # Define a method in Exp3 model like `evaluate_on_split` if needed
                      # Or reuse predict_on_dataframe and evaluate externally
                      # Assuming Exp3 needs dataset/split/suffix info for its internal loader:
                      test_prep_result = loaded_model._load_and_prepare_data(self.dataset_name, 'test', self.suffix)
                      if test_prep_result:
                           X_test_df, y_test = test_prep_result
                           X_test_transformed = loaded_model.extract_features(X_test_df) # Transform using loaded pipeline
                           _, eval_metrics = _evaluate_model_performance(loaded_model, X_test_transformed, y_test)
                           all_eval_metrics[model_to_eval_type] = eval_metrics
                           logger.info(f"Evaluation results for {model_to_eval_type}: {eval_metrics}")
                      else:
                           logger.error("Failed to load/prepare test data within Experiment 3 model.")

                 except AttributeError:
                      logger.error("Experiment 3 model does not have the expected internal data loading/evaluation method.")
                 except Exception as e:
                      logger.error(f"Error during Experiment 3 evaluation: {e}", exc_info=True)

            # Evaluate other models using the provided eval_data
            else:
                 if eval_data is None or eval_data.empty:
                      logger.warning(f"No evaluation data provided for {model_to_eval_type}. Skipping.")
                      return {}

                 clean_eval_result = clean_dataset(eval_data)
                 if not clean_eval_result:
                      logger.error("Evaluation data invalid after cleaning. Cannot evaluate.")
                      return {}
                 eval_data_clean, y_eval = clean_eval_result

                 logger.info("Extracting features for evaluation...")
                 X_eval = None
                 try:
                    # Handle feature extraction based on loaded model type
                    if isinstance(loaded_model, SVMModel):
                         X_eval = loaded_model.extract_features(eval_data_clean)
                    elif isinstance(loaded_model, TextBaselineModel):
                         if not loaded_model.extractor.is_fitted:
                              logger.error("Feature extractor is not fitted/loaded. Cannot extract features.")
                              return {}
                         X_eval = loaded_model.extract_features(eval_data_clean)
                    else:
                         logger.error(f"Loaded model instance type {type(loaded_model)} not recognized for feature extraction.")
                         return {}
                 except Exception as e:
                    logger.error(f"Error during feature extraction for evaluation: {e}", exc_info=True)
                    return {}


                 if X_eval is None or X_eval.shape[0] == 0:
                      logger.error("Feature extraction failed for evaluation data.")
                      return {}

                 _, eval_metrics = _evaluate_model_performance(loaded_model, X_eval, y_eval)
                 all_eval_metrics[model_to_eval_type] = eval_metrics # Store results keyed by model type
                 logger.info(f"Evaluation results for {model_to_eval_type}: {eval_metrics}")

            return all_eval_metrics
        else:
            logger.error(f"Unsupported model type for evaluation: {model_to_eval_type}")
            return {}