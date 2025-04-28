# Modify: IS567FP/models/baseline_trainer.py
import os
import logging
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Optional, Tuple, Dict, Any, Type # Added Type

from config import MODELS_DIR, CACHE_DIR # Use CACHE_DIR for parquet features
from utils.common import NLIModel
from utils.database import DatabaseHandler
# Import base model classes and helpers
from .baseline_base import TextBaselineModel, clean_dataset, _evaluate_model_performance
# Import specific model classes
from .logistic_tf_idf_baseline import LogisticTFIDFBaseline
from .multinomial_naive_bayes_bow_baseline import MultinomialNaiveBayesBaseline
from .svm_bow_baseline import SVMWithBagOfWords, SVMWithSyntax, SVMWithBothFeatures, _handle_nan_values, SVMModel, LexicalFeatureExtractor, SyntacticFeatureExtractor, CombinedFeatureExtractor # Import SVM specifics
from .svm_hand_crafted_syntactic_features_experiment_1 import SVMHandcraftedSyntacticExperiment1
from .svm_bow_hand_crafted_syntactic_features_experiment_2 import SVMBowHandCraftedSyntacticExperiment2
from .logistic_tfidf_syntactic_experiment_3 import LogisticTFIDFSyntacticExperiment3
from .multinomial_naive_bayes_bow_syntactic_experiment_4 import MultinomialNaiveBayesBowSyntacticExperiment4
# Import Experiment 5 model
from .random_forest_bow_syntactic_experiment_5 import RandomForestBowSyntacticExperiment5, CombinedBowSyntacticExtractor


logger = logging.getLogger(__name__)

class BaselineTrainer:
    """Handles training and evaluation for various baseline models."""
    def __init__(self, model_type: str, dataset_name: str, args: object):
        """
        Initializes the trainer for a specific model type and dataset.
        Args:
            model_type (str): Model identifier (e.g., 'svm', 'logistic_tfidf', 'random_forest_bow_syntactic_exp5').
            dataset_name (str): Name of the dataset (e.g., 'SNLI').
            args (object): Command line arguments containing hyperparameters.
        """
        self.model_type = model_type
        self.dataset_name = dataset_name
        self.args = args
        self.sample_size = getattr(args, 'sample_size', None)
        self.suffix = f"sample{self.sample_size}" if self.sample_size else "full"
        self.save_dir = self._get_save_directory()
        os.makedirs(self.save_dir, exist_ok=True)
        self.db_handler = DatabaseHandler()
        self.model: Optional[NLIModel] = None

    def _get_save_directory(self) -> str:
        """Determines the save directory."""
        base_dir = os.path.join(MODELS_DIR, 'baseline_models', self.dataset_name, self.model_type, self.suffix)
        return base_dir

    def _get_model_filename_base(self) -> str:
        """Generates a base filename for saving models."""
        return f"{self.dataset_name}_{self.model_type}_{self.suffix}"

    def _load_data(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Loads train, validation, and test data based on model type."""
        train_data, val_data, test_data = None, None, None
        logger.info(f"Loading data for {self.model_type} on {self.dataset_name} ({self.suffix})")

        # Models needing precomputed features (SVMs, RF_Exp5, MNB_Exp4)
        # <<< MODIFIED Condition >>>
        if self.model_type.startswith('svm') or self.model_type == 'random_forest_bow_syntactic_exp5' or self.model_type == 'mnb_bow_syntactic_exp4':
            feature_type_base = f'features_lexical_syntactic_{self.suffix}'
            parquet_feature_dir = os.path.join(CACHE_DIR, 'parquet')
            logger.info(f"Loading precomputed features from {parquet_feature_dir} using base: {feature_type_base}")

            train_table_name = f"{self.dataset_name}_train_{feature_type_base}"
            val_table_name = f"{self.dataset_name}_validation_{feature_type_base}"
            test_table_name = f"{self.dataset_name}_test_{feature_type_base}"

            try:
                train_data = self.db_handler.load_dataframe(self.dataset_name, 'train', train_table_name)
                if not train_data.empty: train_data = _handle_nan_values(train_data, "training")
                else: logger.error(f"Loaded empty dataframe for train features: {train_table_name}")
            except Exception as e: logger.error(f"Failed to load features ({train_table_name}): {e}", exc_info=True)

            try:
                val_data = self.db_handler.load_dataframe(self.dataset_name, 'validation', val_table_name)
                if not val_data.empty: val_data = _handle_nan_values(val_data, "validation")
                else: logger.warning(f"Loaded empty dataframe for validation features: {val_table_name}")
            except Exception as e: logger.warning(f"Could not load validation features ({val_table_name}): {e}")

            try:
                test_data = self.db_handler.load_dataframe(self.dataset_name, 'test', test_table_name)
                if not test_data.empty: test_data = _handle_nan_values(test_data, "test")
                else: logger.warning(f"Loaded empty dataframe for test features: {test_table_name}")
            except Exception as e: logger.warning(f"Could not load test features ({test_table_name}): {e}")

        # Models needing raw text (TFIDF, MNB_BoW)
        elif self.model_type in ['logistic_tfidf', 'mnb_bow']:
            try: train_data = TextBaselineModel.load_raw_text_data(self.dataset_name, 'train', self.suffix, self.db_handler)
            except Exception as e: logger.error(f"Failed to load raw train data for {self.model_type}: {e}", exc_info=True)
            try: val_data = TextBaselineModel.load_raw_text_data(self.dataset_name, 'validation', self.suffix, self.db_handler)
            except Exception as e: logger.warning(f"Failed to load raw validation data for {self.model_type}: {e}")
            try: test_data = TextBaselineModel.load_raw_text_data(self.dataset_name, 'test', self.suffix, self.db_handler)
            except Exception as e: logger.warning(f"Failed to load raw test data for {self.model_type}: {e}")

        # Experiment 3 handles data loading internally
        elif self.model_type == 'logistic_tfidf_syntactic_exp3':
            logger.info("Experiment 3 selected. Data loading handled internally by the model.")
            return None, None, None # Signal internal loading

        else:
            logger.error(f"Unsupported model type for data loading: {self.model_type}")

        return train_data, val_data, test_data

    def _initialize_model(self) -> Optional[NLIModel]:
        """Initializes the correct model instance."""
        logger.info(f"Initializing model: {self.model_type}")
        model: Optional[NLIModel] = None
        # Get hyperparameters from args with defaults
        kernel = getattr(self.args, 'kernel', 'linear')
        C = getattr(self.args, 'C', 1.0)
        max_features = getattr(self.args, 'max_features', 10000)
        alpha = getattr(self.args, 'alpha', 1.0)
        n_estimators = getattr(self.args, 'n_estimators', 100)
        max_depth = getattr(self.args, 'max_depth', None)
        random_state = 42 # Consistent random state

        try:
            if self.model_type == 'logistic_tfidf':
                model = LogisticTFIDFBaseline(C=C, max_features=max_features)
            elif self.model_type == 'mnb_bow':
                model = MultinomialNaiveBayesBaseline(alpha=alpha, max_features=max_features)
            elif self.model_type == 'svm_syntactic_exp1':
                 model = SVMHandcraftedSyntacticExperiment1(kernel=kernel, C=C)
            elif self.model_type == 'svm_bow_syntactic_exp2':
                 model = SVMBowHandCraftedSyntacticExperiment2(kernel=kernel, C=C)
            elif self.model_type == 'logistic_tfidf_syntactic_exp3':
                 model = LogisticTFIDFSyntacticExperiment3(C=C, tfidf_max_features=max_features)
            elif self.model_type == 'mnb_bow_syntactic_exp4':
                 model = MultinomialNaiveBayesBowSyntacticExperiment4(alpha=alpha, bow_max_features=max_features)
            # <<< ADDED Exp5 Initialization >>>
            elif self.model_type == 'random_forest_bow_syntactic_exp5':
                model = RandomForestBowSyntacticExperiment5(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=random_state
                )
            # --------------------------------
            elif self.model_type == 'svm': # General SVM trains multiple variants
                logger.info("General 'svm' type: Specific variants trained separately.")
                return None # Signal multiple models
            else:
                logger.error(f"Unknown model type for initialization: {self.model_type}")
        except Exception as e:
            logger.error(f"Error initializing model {self.model_type}: {e}", exc_info=True)

        return model

    def run_training(self):
        """Runs the full training and evaluation pipeline."""
        results = {}

        # --- Handle models loading their own data (Exp3, Exp4) ---
        # <<< MODIFIED CONDITION (added Exp4) >>>
        if self.model_type in ['logistic_tfidf_syntactic_exp3', 'mnb_bow_syntactic_exp4']:
            logger.info(f"Handling training for Experiment {self.model_type[-1]}...")
            self.model = self._initialize_model()
            if not self.model:
                logger.error(f"Failed to initialize model {self.model_type}.")
                return None
            try:
                train_results = self.model.train( # Assumes train method exists and handles data
                    train_dataset=self.dataset_name, train_split='train', train_suffix=self.suffix,
                    val_dataset=self.dataset_name, val_split='validation', val_suffix=self.suffix
                )
                results[self.model_type] = train_results
                model_filename_base = self._get_model_filename_base()
                save_path_base = os.path.join(self.save_dir, model_filename_base) # Use base path for saving these models
                self.model.save(save_path_base)
                self.run_evaluation(None) # Evaluate (model loads test data)
            except Exception as e:
                logger.error(f"Error during {self.model_type} training/saving: {e}", exc_info=True)
                return None
            return results

        # --- Load data for other models ---
        train_data, val_data, test_data = self._load_data()
        if train_data is None or train_data.empty:
            logger.error("Training data could not be loaded. Aborting.")
            return None

        # --- Handle SVM Training (Multiple Models for type 'svm') ---
        if self.model_type == 'svm':
            # ... (SVM variant training logic - remains the same) ...
            logger.info("Starting SVM training for BoW, Syntax, and Combined features...")
            svm_results = {}
            kernel = getattr(self.args, 'kernel', 'linear')
            C = getattr(self.args, 'C', 1.0)
            svm_save_dir = self.save_dir # Use the trainer's save dir

            # Prepare validation data
            if val_data is None or val_data.empty:
                 if len(train_data) < 5: logger.error("Not enough training data for validation split."); return None
                 logger.info("Splitting validation set from training data for SVM.")
                 train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42, stratify=train_data.get('label', None))

            clean_train_result = clean_dataset(train_data)
            clean_val_result = clean_dataset(val_data) if val_data is not None else None
            if not clean_train_result: logger.error("Training data invalid after cleaning."); return None
            train_data_clean, y_train = clean_train_result
            val_data_clean, y_val = clean_val_result if clean_val_result else (None, None)

            svm_model_configs = [
                (SVMWithBagOfWords(kernel=kernel, C=C), "SVM_BoW", LexicalFeatureExtractor()),
                (SVMWithSyntax(kernel=kernel, C=C), "SVM_Syntax", SyntacticFeatureExtractor()),
                (SVMWithBothFeatures(kernel=kernel, C=C), "SVM_Combined", CombinedFeatureExtractor())
            ]

            for svm_instance, model_name_suffix, extractor in svm_model_configs:
                logger.info(f"--- Training {model_name_suffix} ---")
                model_filename_base = f"{self.dataset_name}_{model_name_suffix}_{self.suffix}"
                svm_instance.feature_extractor = extractor
                X_train = svm_instance.extract_features(train_data_clean)
                if X_train is None or X_train.size == 0: logger.error(f"Feature extraction failed for {model_name_suffix}. Skipping."); continue

                start_time = time.time()
                svm_instance.train(X_train, y_train)
                train_time = time.time() - start_time; logger.info(f"Training complete in {train_time:.2f}s")

                eval_metrics = {}; eval_time = 0.0
                if val_data_clean is not None and y_val is not None:
                    X_val = svm_instance.extract_features(val_data_clean)
                    if X_val is not None and X_val.size > 0:
                         eval_time, eval_metrics = _evaluate_model_performance(svm_instance, X_val, y_val)
                    else: logger.warning("Validation feature extraction failed. Skipping eval.")
                else: logger.info("Skipping validation evaluation.")

                model_path = os.path.join(svm_save_dir, f"{model_filename_base}.joblib")
                svm_instance.save(model_path)
                svm_results[model_name_suffix] = {**eval_metrics, 'train_time': train_time, 'eval_time': eval_time}

            logger.info("Finished training all SVM variants.")
            self.run_evaluation(test_data, model_type='svm')
            return svm_results


        # --- Handle Text Baselines, Specific SVM Experiments, AND RF Exp5 ---
        # <<< MODIFIED Condition >>>
        elif self.model_type in ['logistic_tfidf', 'mnb_bow', 'svm_syntactic_exp1', 'svm_bow_syntactic_exp2', 'random_forest_bow_syntactic_exp5']:
            self.model = self._initialize_model()
            if not self.model: logger.error(f"Failed to initialize model {self.model_type}. Aborting."); return None

            eval_results = {}; train_time = 0.0; eval_time = 0.0

            # -- Training logic --
            if isinstance(self.model, (SVMModel, RandomForestBowSyntacticExperiment5)): # Use precomputed features
                logger.info(f"Handling {self.model_type}. Using precomputed features.")
                clean_train_result = clean_dataset(train_data)
                if not clean_train_result: logger.error("Training data invalid after cleaning."); return None
                train_data_clean, y_train = clean_train_result
                logger.info(f"Extracting features for {self.model_type}...")
                X_train = self.model.extract_features(train_data_clean)
                if X_train is None or X_train.size == 0: logger.error("Feature extraction failed for training."); return None

                start_time = time.time()
                self.model.train(X_train, y_train)
                train_time = time.time() - start_time; logger.info(f"Training complete in {train_time:.2f}s")

            elif isinstance(self.model, TextBaselineModel): # Use raw text
                logger.info(f"Handling {self.model_type}. Using raw text data.")
                clean_train_result = clean_dataset(train_data)
                if not clean_train_result: logger.error("Raw training data invalid after cleaning."); return None
                train_data_clean, y_train = clean_train_result
                logger.info(f"Fitting feature extractor ({self.model.extractor.__class__.__name__})...")
                self.model.extractor.fit(train_data_clean)
                logger.info("Transforming training data..."); X_train = self.model.extract_features(train_data_clean)
                if X_train is None or X_train.shape[0] == 0: logger.error("Feature extraction failed for training."); return None

                start_time = time.time()
                self.model.train(X_train, y_train)
                train_time = time.time() - start_time; logger.info(f"Training complete in {train_time:.2f}s")
            else:
                logger.error(f"Model instance is not a recognized type for training."); return None

            # -- Validation logic --
            if val_data is not None and not val_data.empty:
                clean_val_result = clean_dataset(val_data)
                if clean_val_result:
                    val_data_clean, y_val = clean_val_result
                    logger.info("Evaluating on validation data..."); X_val = self.model.extract_features(val_data_clean)
                    eval_time, eval_metrics = _evaluate_model_performance(self.model, X_val, y_val)
                    eval_results = {**eval_metrics, 'eval_time': eval_time}
                else: logger.warning("Validation data invalid after cleaning.")
            else: logger.info("Skipping validation evaluation (no validation data).")

            # -- Saving logic --
            model_filename_base = self._get_model_filename_base()
            save_path = os.path.join(self.save_dir, model_filename_base) # For RF, it's base path + .joblib; For Text, it's dir + base_name
            if isinstance(self.model, TextBaselineModel):
                 self.model.save(self.save_dir, model_filename_base)
            elif isinstance(self.model, (SVMModel, RandomForestBowSyntacticExperiment5)):
                 self.model.save(save_path) # Pass base path, .save() adds extension
            else:
                 logger.error("Cannot determine how to save this model type.")


            results[self.model_type] = {**eval_results, 'train_time': train_time}
            self.run_evaluation(test_data) # Evaluate this specific model

        else:
             logger.error(f"Unsupported model type in run_training: {self.model_type}"); return None

        return results

    def run_evaluation(self, eval_data: Optional[pd.DataFrame], model_type: Optional[str] = None):
        """Evaluates the trained model(s) on the provided data."""
        model_to_eval_type = model_type or self.model_type
        eval_dataset_name = self.dataset_name
        logger.info(f"Starting evaluation: Model={model_to_eval_type}, Dataset={eval_dataset_name}, Suffix={self.suffix}")

        all_eval_metrics = {}

        # --- Special Handling for 'svm' (evaluate all variants) ---
        if model_to_eval_type == 'svm':
            # ... (Keep existing SVM variant evaluation logic) ...
            if eval_data is None or eval_data.empty: logger.warning("No evaluation data for SVMs."); return {}
            svm_variants_to_eval = [
                (SVMWithBagOfWords, "SVM_BoW", LexicalFeatureExtractor()),
                (SVMWithSyntax, "SVM_Syntax", SyntacticFeatureExtractor()),
                (SVMWithBothFeatures, "SVM_Combined", CombinedFeatureExtractor())
            ]
            clean_eval_result = clean_dataset(eval_data)
            if not clean_eval_result: logger.error("Eval data invalid for SVMs."); return {}
            eval_data_clean, y_eval = clean_eval_result

            for model_cls, model_name_suffix, extractor_instance in svm_variants_to_eval:
                svm_variant_filename_base = f"{eval_dataset_name}_{model_name_suffix}_{self.suffix}"
                model_path = os.path.join(self.save_dir, f"{svm_variant_filename_base}.joblib")
                try:
                    loaded_svm_model = model_cls.load(model_path, extractor_instance)
                    X_eval = loaded_svm_model.extract_features(eval_data_clean)
                    if X_eval is not None and X_eval.shape[0] > 0:
                        _, eval_metrics = _evaluate_model_performance(loaded_svm_model, X_eval, y_eval)
                        all_eval_metrics[model_name_suffix] = eval_metrics
                        logger.info(f"Eval results for {model_name_suffix}: {eval_metrics}")
                    else: logger.warning(f"Feature extraction failed for eval on {model_name_suffix}")
                except FileNotFoundError: logger.error(f"SVM model not found: {model_path}")
                except Exception as e: logger.error(f"Error during SVM eval for {model_name_suffix}: {e}", exc_info=True)
            return all_eval_metrics

        # --- Handling for other specific model types ---
        # <<< MODIFIED Condition >>>
        elif model_to_eval_type in ['logistic_tfidf', 'mnb_bow', 'svm_syntactic_exp1', 'svm_bow_syntactic_exp2', 'logistic_tfidf_syntactic_exp3', 'mnb_bow_syntactic_exp4', 'random_forest_bow_syntactic_exp5']:
            # Load the single specified model
            model_filename_base = f"{eval_dataset_name}_{model_to_eval_type}_{self.suffix}"
            load_path_base = os.path.join(self.save_dir, model_filename_base) # Base path for loading
            loaded_model: Optional[NLIModel] = None
            model_cls: Optional[Type[NLIModel]] = None

            try:
                logger.info(f"Loading {model_to_eval_type} model from base path {load_path_base} for evaluation...")
                if model_to_eval_type == 'logistic_tfidf': model_cls = LogisticTFIDFBaseline
                elif model_to_eval_type == 'mnb_bow': model_cls = MultinomialNaiveBayesBaseline
                elif model_to_eval_type == 'svm_syntactic_exp1': model_cls = SVMHandcraftedSyntacticExperiment1
                elif model_to_eval_type == 'svm_bow_syntactic_exp2': model_cls = SVMBowHandCraftedSyntacticExperiment2
                elif model_to_eval_type == 'logistic_tfidf_syntactic_exp3': model_cls = LogisticTFIDFSyntacticExperiment3
                elif model_to_eval_type == 'mnb_bow_syntactic_exp4': model_cls = MultinomialNaiveBayesBowSyntacticExperiment4
                elif model_to_eval_type == 'random_forest_bow_syntactic_exp5': model_cls = RandomForestBowSyntacticExperiment5 # Added Exp5
                else: raise ValueError("Unknown model type for loading.")

                # Call the correct load method
                if issubclass(model_cls, TextBaselineModel):
                    loaded_model = model_cls.load(self.save_dir, model_filename_base)
                elif issubclass(model_cls, SVMModel): # Covers Exp1, Exp2
                     extractor = None
                     if model_cls == SVMHandcraftedSyntacticExperiment1: extractor = SyntacticFeatureExtractor()
                     elif model_cls == SVMBowHandCraftedSyntacticExperiment2: extractor = CombinedFeatureExtractor()
                     svm_model_path = os.path.join(self.save_dir, f"{model_filename_base}.joblib")
                     loaded_model = model_cls.load(svm_model_path, extractor)
                elif issubclass(model_cls, (LogisticTFIDFSyntacticExperiment3, MultinomialNaiveBayesBowSyntacticExperiment4, RandomForestBowSyntacticExperiment5)): # Exp3, Exp4, Exp5
                     loaded_model = model_cls.load(load_path_base) # These load using base path
                else: raise TypeError(f"Don't know how to load model type {model_to_eval_type}")

                if loaded_model is None: raise FileNotFoundError

            except FileNotFoundError: logger.error(f"Model artifacts not found for {model_filename_base} at {load_path_base}. Cannot evaluate."); return {}
            except Exception as e: logger.error(f"Error loading model {model_filename_base}: {e}", exc_info=True); return {}

            # --- Perform Evaluation ---
            # <<< MODIFIED CONDITION (Added Exp4, Exp5) >>>
            if model_to_eval_type in ['logistic_tfidf_syntactic_exp3', 'mnb_bow_syntactic_exp4', 'random_forest_bow_syntactic_exp5']:
                # These models handle their own test data loading within _load_and_prepare_data
                logger.info(f"Triggering internal evaluation for {model_to_eval_type}...")
                try:
                    # Load test data using the model's helper
                    test_prep_result = loaded_model._load_and_prepare_data(self.dataset_name, 'test', self.suffix)
                    if test_prep_result:
                        X_test_df, y_test = test_prep_result
                        X_test_transformed = loaded_model.extract_features(X_test_df)
                        _, eval_metrics = _evaluate_model_performance(loaded_model, X_test_transformed, y_test)
                        all_eval_metrics[model_to_eval_type] = eval_metrics
                        logger.info(f"Evaluation results for {model_to_eval_type}: {eval_metrics}")
                    else: logger.error(f"Failed to load/prepare test data within {model_to_eval_type} model.")
                except AttributeError: logger.error(f"{model_to_eval_type} missing _load_and_prepare_data method.")
                except Exception as e: logger.error(f"Error during {model_to_eval_type} evaluation: {e}", exc_info=True)
            else: # Evaluate other models (TFIDF, MNB, SVM Exp1/Exp2) using provided eval_data
                if eval_data is None or eval_data.empty: logger.warning(f"No evaluation data provided for {model_to_eval_type}. Skipping."); return {}
                clean_eval_result = clean_dataset(eval_data)
                if not clean_eval_result: logger.error("Evaluation data invalid after cleaning."); return {}
                eval_data_clean, y_eval = clean_eval_result
                logger.info("Extracting features for evaluation..."); X_eval = None
                try: X_eval = loaded_model.extract_features(eval_data_clean)
                except Exception as e: logger.error(f"Error during feature extraction for evaluation: {e}", exc_info=True); return {}
                if X_eval is None or X_eval.shape[0] == 0: logger.error("Feature extraction failed for evaluation data."); return {}
                _, eval_metrics = _evaluate_model_performance(loaded_model, X_eval, y_eval)
                all_eval_metrics[model_to_eval_type] = eval_metrics
                logger.info(f"Evaluation results for {model_to_eval_type}: {eval_metrics}")

            return all_eval_metrics
        else:
            logger.error(f"Unsupported model type for evaluation: {model_to_eval_type}")
            return {}