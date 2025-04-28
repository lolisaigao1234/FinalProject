# Modify: IS567FP/models/baseline_trainer.py
import os
import logging
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Optional, Tuple, Dict, Any

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
# Import Experiment 2 model
from .svm_bow_hand_crafted_syntactic_features_experiment_2 import SVMBowHandCraftedSyntacticExperiment2

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
        self.db_handler = DatabaseHandler() # Needed for text baselines
        self.model: Optional[NLIModel] = None # Allow SVMModel or TextBaselineModel

    def _get_save_directory(self) -> str:
        """Determines the save directory based on the model type."""
        base_dir = os.path.join(MODELS_DIR, f"{self.model_type}_baseline")
        return base_dir

    def _get_model_filename_base(self) -> str:
        """Generates a base filename for saving models/extractors."""
        return f"{self.dataset_name}_{self.model_type}_{self.suffix}"

    def _load_data(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Loads train, validation, and test data based on model type."""
        train_data, val_data, test_data = None, None, None
        logger.info(f"Loading data for {self.model_type} on {self.dataset_name} ({self.suffix})")

        # Determine the feature type pattern needed based on the model
        # SVM models need the precomputed lexical+syntactic features
        if self.model_type.startswith('svm'):
            feature_type_suffix = f'features_lexical_syntactic_{self.suffix}'
            parquet_feature_dir = os.path.join(DATA_DIR, '..', 'cache', 'parquet') # Adjust base if needed
            logger.info(f"SVM family: Loading precomputed features from {parquet_feature_dir} with suffix: {feature_type_suffix}")
            try:
                # Construct the full feature filename base for load_parquet_data
                # e.g., "SNLI_train_features_lexical_syntactic_sample100"
                train_table_name = f"{self.dataset_name}_train_{feature_type_suffix}"
                train_data = load_parquet_data(self.dataset_name, 'train', feature_type=train_table_name, cache_dir=parquet_feature_dir)
                train_data = _handle_nan_values(train_data, "training")
            except FileNotFoundError:
                logger.error(f"SVM training features not found for: {train_table_name}")
            try:
                val_table_name = f"{self.dataset_name}_validation_{feature_type_suffix}"
                val_data = load_parquet_data(self.dataset_name, 'validation', feature_type=val_table_name, cache_dir=parquet_feature_dir)
                val_data = _handle_nan_values(val_data, "validation")
            except FileNotFoundError:
                logger.warning("SVM validation features not found.")
            try:
                test_table_name = f"{self.dataset_name}_test_{feature_type_suffix}"
                test_data = load_parquet_data(self.dataset_name, 'test', feature_type=test_table_name, cache_dir=parquet_feature_dir)
                test_data = _handle_nan_values(test_data, "test")
            except FileNotFoundError:
                logger.warning("SVM test features not found.")

        elif self.model_type in ['logistic_tfidf', 'mnb_bow']:
            # Text baselines load raw text data (intermediate pairs+sentences)
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
            elif self.model_type == 'svm_bow_syntactic_exp2': # Added Experiment 2
                 model = SVMBowHandCraftedSyntacticExperiment2(kernel=kernel, C=C)
            elif self.model_type == 'svm':
                # General SVM handled later
                logger.info("General 'svm' model type selected. Specific variants will be trained.")
                return None
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
             # This section trains the original 3 SVM variants if --model_type svm is specified.
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
        # <<< MODIFIED THIS CONDITION >>>
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
                 model_path = os.path.join(self.save_dir, f"{model_filename_base}.joblib")
                 self.model.save(model_path)

                 self.run_evaluation(test_data) # Evaluate this specific model on test data
                 return {**eval_results, 'train_time': train_time}

            elif isinstance(self.model, TextBaselineModel): # TFIDF or BoW
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
                 self.model.save(self.save_dir, model_filename_base)

                 self.run_evaluation(test_data) # Evaluate this specific model on test data
                 return {**eval_results, 'train_time': train_time}
            else:
                 logger.error(f"Initialized model is not a recognized baseline type (SVMModel or TextBaselineModel).")
                 return None
        else:
             logger.error(f"Unsupported model type in run_training: {self.model_type}")
             return None


    def run_evaluation(self, eval_data: Optional[pd.DataFrame], model_type: Optional[str] = None):
        """Evaluates the trained model(s) on the provided data (e.g., test set)."""
        model_to_eval_type = model_type or self.model_type
        eval_dataset_name = self.dataset_name # Assuming evaluation is on the same dataset for now
        logger.info(f"Starting evaluation for model type: {model_to_eval_type} on dataset: {eval_dataset_name}")

        if eval_data is None or eval_data.empty:
             logger.warning(f"No evaluation data provided for {eval_dataset_name}/{model_to_eval_type}. Skipping evaluation.")
             return {}

        # --- Load and Evaluate SVM Variants (if model_type is 'svm') ---
        if model_to_eval_type == 'svm':
            # This evaluates the original 3 SVM variants
            svm_results = {}
            svm_model_configs = [
                (SVMWithBagOfWords, "SVM_BoW", LexicalFeatureExtractor()),
                (SVMWithSyntax, "SVM_Syntax", SyntacticFeatureExtractor()),
                (SVMWithBothFeatures, "SVM_Combined", CombinedFeatureExtractor())
            ]
            clean_eval_result = clean_dataset(eval_data)
            if not clean_eval_result:
                 logger.error("Evaluation data invalid after cleaning. Cannot evaluate SVMs.")
                 return {}
            eval_data_clean, y_eval = clean_eval_result

            for model_cls, model_name_suffix, extractor in svm_model_configs:
                model_filename_base = f"{eval_dataset_name}_{model_name_suffix}_{self.suffix}"
                model_path = os.path.join(self.save_dir, f"{model_filename_base}.joblib") # Use base class save_dir
                try:
                    logger.info(f"Loading SVM model for evaluation: {model_path}")
                    # Pass the correct extractor instance when loading
                    loaded_svm_model = model_cls.load(model_path, feature_extractor=extractor)
                    logger.info("Extracting features for evaluation...")
                    X_eval = loaded_svm_model.extract_features(eval_data_clean)
                    if X_eval is not None and X_eval.shape[0] > 0:
                         _, eval_metrics = _evaluate_model_performance(loaded_svm_model, X_eval, y_eval)
                         svm_results[model_name_suffix] = eval_metrics
                         logger.info(f"Evaluation results for {model_name_suffix}: {eval_metrics}")
                    else:
                         logger.warning(f"Feature extraction failed for evaluation data on {model_name_suffix}")
                except FileNotFoundError:
                    logger.error(f"SVM model file not found, cannot evaluate: {model_path}")
                except Exception as e:
                    logger.error(f"Error during SVM evaluation for {model_name_suffix}: {e}", exc_info=True)
            return svm_results

        # --- Load and Evaluate Single Text Baseline or Specific SVM Experiment ---
        # <<< MODIFIED THIS CONDITION >>>
        elif model_to_eval_type in ['logistic_tfidf', 'mnb_bow', 'svm_syntactic_exp1', 'svm_bow_syntactic_exp2']:
            # Load the single specified model
            model_filename_base = f"{eval_dataset_name}_{model_to_eval_type}_{self.suffix}" # Use correct model type
            model_path = os.path.join(self.save_dir, f"{model_filename_base}.joblib")
            loaded_model: Optional[NLIModel] = None

            try:
                logger.info(f"Loading {model_to_eval_type} model from {model_path} for evaluation...")
                if model_to_eval_type == 'logistic_tfidf':
                     loaded_model = LogisticTFIDFBaseline.load(self.save_dir, model_filename_base)
                elif model_to_eval_type == 'mnb_bow':
                     loaded_model = MultinomialNaiveBayesBaseline.load(self.save_dir, model_filename_base)
                elif model_to_eval_type == 'svm_syntactic_exp1':
                     loaded_model = SVMHandcraftedSyntacticExperiment1.load(model_path, SyntacticFeatureExtractor())
                elif model_to_eval_type == 'svm_bow_syntactic_exp2':
                     loaded_model = SVMBowHandCraftedSyntacticExperiment2.load(model_path, CombinedFeatureExtractor())

                if loaded_model is None: raise FileNotFoundError # Trigger except if load failed internally

            except FileNotFoundError:
                 logger.error(f"Model {model_filename_base} not found in {self.save_dir}. Cannot evaluate.")
                 return {}
            except Exception as e:
                 logger.error(f"Error loading model {model_filename_base}: {e}", exc_info=True)
                 return {}

            # Evaluate the loaded model
            clean_eval_result = clean_dataset(eval_data)
            if not clean_eval_result:
                 logger.error("Evaluation data invalid after cleaning. Cannot evaluate.")
                 return {}
            eval_data_clean, y_eval = clean_eval_result

            logger.info("Extracting features for evaluation...")
            # Handle feature extraction based on loaded model type
            if isinstance(loaded_model, SVMModel):
                 X_eval = loaded_model.extract_features(eval_data_clean)
            elif isinstance(loaded_model, TextBaselineModel):
                 if not loaded_model.extractor.is_fitted:
                      logger.error("Feature extractor is not fitted/loaded. Cannot extract features.")
                      return {}
                 X_eval = loaded_model.extract_features(eval_data_clean)
            else:
                 logger.error(f"Loaded model instance type {type(loaded_model)} not recognized for evaluation.")
                 return {}

            if X_eval is None or X_eval.shape[0] == 0:
                 logger.error("Feature extraction failed for evaluation data.")
                 return {}

            _, eval_metrics = _evaluate_model_performance(loaded_model, X_eval, y_eval)
            logger.info(f"Evaluation results for {model_to_eval_type}: {eval_metrics}")
            return {model_to_eval_type: eval_metrics} # Return results keyed by model type
        else:
            logger.error(f"Unsupported model type for evaluation: {model_to_eval_type}")
            return {}