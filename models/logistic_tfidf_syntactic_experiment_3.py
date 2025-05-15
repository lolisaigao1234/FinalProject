# File: models/logistic_tfidf_syntactic_experiment_3.py
import logging
import os
from typing import Tuple, Optional, Any, Dict, List
import pandas as pd
import numpy as np
import joblib
import time
from scipy.sparse import hstack, csr_matrix

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler  # Optional: Scale syntactic features
from sklearn.base import BaseEstimator, TransformerMixin

# Import base classes and helpers from the project structure
from utils.common import NLIModel
from utils.database import DatabaseHandler
# Import helpers specifically used for baseline models and experiments
from .baseline_base import (
    clean_dataset,
    prepare_labels,  # May not be needed if clean_dataset handles it
    _evaluate_model_performance,
    SimpleParquetLoader,  # Needed for loading features in evaluate
    _handle_nan_values,  # Needed for handling NaNs during evaluation data load
    filter_syntactic_features  # Needed to identify syntactic columns
)

logger = logging.getLogger(__name__)


# Helper Transformer for selecting text columns
class TextSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Expects X to be a DataFrame
        return X[self.key]


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Selects precomputed feature columns specified during initialization.
    During fit, it identifies which of the specified columns are actually
    present in the training data.
    During transform, it ensures only those fitted columns are present,
    adding any missing ones (that were seen during fit) with a value of 0.
    """

    def __init__(self, column_names: List[str]):
        # column_names provided here are the *potential* columns expected
        self.potential_column_names = column_names
        # This will store the names of columns actually present during fitting
        self.columns_seen_during_fit_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y=None):
        """
        Identifies and stores the names of the potential columns
        that are actually present in the input DataFrame X.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame for fit.")

        # Identify which of the potential columns exist in the training data
        self.columns_seen_during_fit_ = [col for col in self.potential_column_names if col in X.columns]

        # Log which potential columns were missing during fitting (optional but helpful)
        missing_at_fit = set(self.potential_column_names) - set(self.columns_seen_during_fit_)
        if missing_at_fit:
            logger.warning(
                f"FeatureSelector fit: Potential columns {missing_at_fit} not found in training data. Selector will only use columns found: {self.columns_seen_during_fit_}")
        else:
            logger.info(
                f"FeatureSelector fit: All {len(self.columns_seen_during_fit_)} potential columns were found in the training data.")

        if not self.columns_seen_during_fit_:
            logger.error("FeatureSelector fit: No specified columns found in the training data!")
            # Or raise an error depending on desired behavior:
            # raise ValueError("No specified columns found in the training data during fit.")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input DataFrame X by selecting only the columns
        identified during the fit phase. Adds missing columns (that were
        seen during fit) with a value of 0.
        """
        # Check if fit has been called
        if self.columns_seen_during_fit_ is None:
            raise RuntimeError("FeatureSelector must be fitted before transforming data.")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame for transform.")

        # Identify which columns seen during fit are missing in the current DataFrame X
        missing_for_transform = set(self.columns_seen_during_fit_) - set(X.columns)

        if missing_for_transform:
            # This was the condition causing the original warning.
            # We handle it by adding the missing columns.
            logger.warning(
                f"FeatureSelector transform: Columns {missing_for_transform} (seen during fit) are missing in the data being transformed. Adding these columns with 0.")
            # Create a copy only if we need to add columns to avoid modifying original df
            X_processed = X.copy()
            for col in missing_for_transform:
                X_processed[col] = 0  # Add missing column filled with zeros
        else:
            # No columns missing, can work directly or with a copy if needed
            X_processed = X  # Or X.copy() if downstream steps modify it

        # Return only the columns seen during fit, ensuring consistent order and set of columns
        # Use reindex for safety, handling cases where X might have extra columns
        # Ensure columns are returned in the same order as they were seen during fit
        return X_processed.reindex(columns=self.columns_seen_during_fit_, fill_value=0)


class LogisticTFIDFSyntacticExperiment3(NLIModel):
    """
    Experiment 3: Logistic Regression combining TF-IDF (from raw text) and
                  pre-computed hand-crafted syntactic features.
    """

    def __init__(self, args=None, **kwargs):  # Accept args and kwargs like other experiments
        # Process args or use defaults for hyperparameters
        # Ensure None check for args
        self.C = getattr(args, 'C', 1.0) if args else 1.0
        self.max_iter = getattr(args, 'max_iter', 1000) if args else 1000
        self.tfidf_max_features = getattr(args, 'tfidf_max_features', 10000) if args else 10000
        self.tfidf_ngram_range = getattr(args, 'tfidf_ngram_range', (1, 2)) if args else (1, 2)

        # Initialize model with hyperparameters
        self.model = LogisticRegression(C=self.C, max_iter=self.max_iter, solver='liblinear', random_state=42)
        self.feature_pipeline: Optional[Pipeline] = None  # The combined TFIDF+Syntactic pipeline
        self.db_handler = DatabaseHandler()  # Keep if needed for data loading (primarily for training)
        self.is_trained = False
        self._syntactic_feature_columns: Optional[List[str]] = None  # Store names of syntactic features used

    # Removed _load_and_prepare_data as training logic is handled by ExperimentTrainer
    # ExperimentTrainer loads features and passes them to extract_features and train methods.

    def _build_feature_pipeline(self, sample_df_for_fitting: Optional[pd.DataFrame] = None) -> Pipeline:
        """
        Builds the scikit-learn feature transformation pipeline (TF-IDF + Syntactic Selection/Scaling).
        This is called during training to fit the pipeline and during loading to restore it.
        """
        logger.info("Building feature pipeline for Exp3...")

        # TF-IDF part for premise + hypothesis
        tfidf_premise = Pipeline([
            ('selector', TextSelector(key='premise_text')),
            ('tfidf', TfidfVectorizer(max_features=self.tfidf_max_features // 2 if self.tfidf_max_features else None,
                                      # Split features
                                      ngram_range=self.tfidf_ngram_range,
                                      stop_words='english'))
        ])
        tfidf_hypothesis = Pipeline([
            ('selector', TextSelector(key='hypothesis_text')),
            ('tfidf', TfidfVectorizer(max_features=self.tfidf_max_features // 2 if self.tfidf_max_features else None,
                                      # Split features
                                      ngram_range=self.tfidf_ngram_range,
                                      stop_words='english'))
        ])

        # Syntactic features part
        # If fitting pipeline, discover syntactic columns now
        if sample_df_for_fitting is not None and self._syntactic_feature_columns is None:
            # Use the helper function from baseline_base to identify columns
            self._syntactic_feature_columns = filter_syntactic_features(sample_df_for_fitting)
            logger.info(f"Identified {len(self._syntactic_feature_columns)} syntactic feature columns for pipeline.")
            if not self._syntactic_feature_columns:
                logger.warning(
                    "No syntactic feature columns were identified. Syntactic part of the pipeline will be empty.")
        elif self._syntactic_feature_columns is None:
            # This should only happen if loading a model that wasn't saved correctly or calling predict before train
            raise RuntimeError("Syntactic feature columns not identified. Ensure model is trained or loaded correctly.")

        # Ensure syntactic_feature_columns is a list even if empty
        syntactic_cols_to_use = self._syntactic_feature_columns if self._syntactic_feature_columns else []

        syntactic_pipe = Pipeline([
            # Use FeatureSelector to handle column selection and missing columns during transform
            ('selector', FeatureSelector(column_names=syntactic_cols_to_use)),
            # Optional: Scale syntactic features (StandardScaler handles sparse matrices if needed)
            ('scaler', StandardScaler(with_mean=False))  # Use with_mean=False for sparse data if applicable
        ])

        # Combine using FeatureUnion
        combined_features = FeatureUnion([
            ('tfidf_premise', tfidf_premise),
            ('tfidf_hypothesis', tfidf_hypothesis),
            ('syntactic', syntactic_pipe)
        ],
            # Optional: Set weights for features if desired
            transformer_weights={
                'tfidf_premise': 1.0,
                'tfidf_hypothesis': 1.0,
                'syntactic': 1.0,
            }
        )

        # Full pipeline (Feature Extraction only)
        # The model (LogisticRegression) is applied *after* this pipeline transforms the data
        pipeline = Pipeline([('features', combined_features)])
        logger.info("Feature pipeline built.")
        return pipeline

    # Implement NLIModel abstract methods

    # In models/logistic_tfidf_syntactic_experiment_3.py
    def extract_features(self, data: pd.DataFrame) -> Any:
        """
        Transforms input DataFrame using the feature pipeline.
        - If the model is not yet trained (self.is_trained is False), this indicates
          it's being called during the training phase before the .train() method.
          In this case, it FITS the pipeline on the data and then transforms it.
        - If the model IS trained (self.is_trained is True), this indicates it's
          being called during evaluation or prediction. In this case, it only
          TRANSFORMS the data using the previously fitted pipeline.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input data for feature extraction must be a pandas DataFrame.")

        # Ensure required text columns are present for TFIDF processing
        if 'premise_text' not in data.columns or 'hypothesis_text' not in data.columns:
            # We need text columns for both fitting and transforming TFIDF
            raise ValueError("Missing 'premise_text' or 'hypothesis_text' columns required for the feature pipeline.")

        if not self.is_trained:
            # --- Training Phase: Fit Pipeline and Transform ---
            logger.info("Model not trained yet. Assuming training context: Building/Fitting pipeline...")

            # Build the pipeline structure if it doesn't exist.
            # _build_feature_pipeline needs the data to identify syntactic columns.
            if not self.feature_pipeline:
                self.feature_pipeline = self._build_feature_pipeline(sample_df_for_fitting=data)
                # After building, _syntactic_feature_columns should be set internally.
                if not self._syntactic_feature_columns:
                    logger.warning("No syntactic feature columns identified during pipeline build.")

            logger.info("Fitting feature pipeline and transforming training data...")
            start_transform_time = time.time()
            try:
                # Fit the pipeline (TFIDF, Scaler, etc.) and transform the data in one step.
                # Pass 'data' as X. y is not typically needed for fit_transform here.
                X_transformed = self.feature_pipeline.fit_transform(data)  # Fit and transform
                logger.info(f"Pipeline fit during training context successful.")
            except Exception as e:
                logger.error(f"Error during pipeline fit_transform in extract_features (training context): {e}",
                             exc_info=True)
                logger.error(f"Input data columns: {data.columns.tolist()}")
                logger.error(f"Pipeline steps: {self.feature_pipeline.steps if self.feature_pipeline else 'None'}")
                raise  # Re-raise

            transform_time = time.time() - start_transform_time
            logger.info(
                f"Pipeline fit & transform completed in {transform_time:.2f}s. Output shape: {X_transformed.shape}")

            # Crucially, self.is_trained is NOT set here.
            # It will be set only after the classifier training in the .train() method succeeds.
            return X_transformed

        else:
            # --- Evaluation/Prediction Phase: Transform Only ---
            logger.info("Model is trained. Assuming evaluation/prediction context: Transforming data...")
            if not self.feature_pipeline:
                # This is an inconsistent state - model trained but no pipeline.
                # Could happen if loading failed or save was incomplete.
                raise RuntimeError("Model is marked as trained, but the feature pipeline is missing. Load/Save issue?")

            logger.debug("Transforming data using the fitted pipeline...")
            try:
                # Use transform() ONLY, as the pipeline is already fitted from the training phase call.
                extracted = self.feature_pipeline.transform(data)
                logger.debug(f"Feature extraction (transform) successful. Output shape: {extracted.shape}")
                return extracted
            except Exception as e:
                logger.error(f"Error during pipeline transform in extract_features (evaluation context): {e}",
                             exc_info=True)
                # Log details helpful for debugging transform issues (e.g., missing columns)
                logger.error(f"Syntactic columns expected by pipeline: {self._syntactic_feature_columns}")
                logger.error(f"Columns available in input data: {data.columns.tolist()}")
                if self._syntactic_feature_columns:
                    missing_syn = [col for col in self._syntactic_feature_columns if col not in data.columns]
                    if missing_syn:
                        # FeatureSelector should handle this, but log it.
                        logger.warning(
                            f"Input data missing expected syntactic columns during transform: {missing_syn}. FeatureSelector should add them as 0.")
                raise  # Re-raise the exception

    # In models/logistic_tfidf_syntactic_experiment_3.py

    def train(self, X: Any, y: np.ndarray) -> None:
        """
        Trains the Logistic Regression model on the *already transformed* features (X).
        The feature pipeline fitting now happens within the `extract_features` method
        when it's called by the ExperimentTrainer just before this `train` method.
        """
        if X is None or y is None:
            raise ValueError("Training features (X received from extract_features) or labels (y) are None.")
        if X.shape[0] == 0:
            logger.warning("Received 0 training samples (X shape is 0). Cannot train.")
            # Or raise error depending on desired behaviour
            # raise ValueError("Received 0 training samples.")
            self.is_trained = False  # Ensure not marked as trained
            return  # Exit early
        if X.shape[0] != len(y):
            raise ValueError(f"Feature/Label mismatch during training: X={X.shape}, y={len(y)}")
        if self.is_trained:
            # This case should ideally not happen with the standard ExperimentTrainer flow,
            # but handle it defensively.
            logger.warning("Model is already marked as trained. Re-training the classifier...")
            self.is_trained = False  # Reset flag before attempting to train again

        logger.info(f"Training Logistic Regression classifier on processed data with shape: {X.shape}")
        start_time = time.time()
        try:
            # Train the actual scikit-learn Logistic Regression model
            self.model.fit(X, y)
            # Set the flag ONLY after successful fitting of the classifier
            self.is_trained = True
            logger.info(f"Logistic Regression classifier training completed successfully.")
        except Exception as e:
            logger.error(f"Error during Logistic Regression model fitting: {e}", exc_info=True)
            self.is_trained = False  # Ensure flag remains False if fit fails
            raise  # Re-raise

        train_time = time.time() - start_time
        logger.info(f"Classifier training phase completed in {train_time:.2f}s.")

    def predict(self, X: Any) -> np.ndarray:
        """Predicts labels for *already transformed* feature data."""
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet.")
        if X is None:
            raise ValueError("Input features (X) for prediction are None.")

        # Note: X here should be the *already transformed* features
        # (output from self.extract_features)
        logger.debug(f"Predicting labels for {X.shape[0]} samples...")
        predictions = self.model.predict(X)
        logger.debug("Prediction finished.")
        return predictions

    # Overload predict to handle DataFrame input for convenience (extracts then predicts)
    # This might be less used with ExperimentTrainer but useful for direct calls.
    def predict_on_dataframe(self, data_df: pd.DataFrame) -> np.ndarray:
        """Extracts features from DataFrame and predicts."""
        if not isinstance(data_df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        logger.info("Extracting features from DataFrame for prediction...")
        X_transformed = self.extract_features(data_df)  # Uses the fitted pipeline
        logger.info("Predicting on extracted features...")
        return self.predict(X_transformed)

    def evaluate(self, dataset_name: str, split: str, suffix: str) -> Dict[str, Any]:
        """
        Evaluates the trained model on a given dataset split.
        Loads pre-computed combined features, transforms them, predicts, and calculates metrics.
        """
        if not self.is_trained or not self.feature_pipeline:
            logger.error(f"Cannot evaluate model {self.__class__.__name__}: Model or pipeline is not trained/loaded.")
            return {}

        logger.info(f"Evaluating {self.__class__.__name__} on {dataset_name}/{split}/{suffix}...")
        data_loader = SimpleParquetLoader()
        eval_df = None
        # Determine the expected feature file name (should match FeatureExtractor output)
        # Assuming FeatureExtractor saves stats+syntactic together
        feature_table_name = f"{dataset_name}_{split}_features_stats_syntactic_{suffix}"  # Match FeatureExtractor pattern

        try:
            # Load the pre-computed features for the evaluation split
            logger.info(f"Loading evaluation features from Parquet (table name convention: {feature_table_name})")
            eval_df = data_loader.load_data(dataset_name, split, suffix)  # SimpleParquetLoader might need suffix hint
            # ^^^ NOTE: SimpleParquetLoader might need update to find the correct file based on the pattern.
            # Alternatively, construct the exact path if known. For now, assume loader finds it.

            if eval_df is None or eval_df.empty:
                # Try loading explicitly using the expected name if SimpleParquetLoader failed
                logger.warning(
                    f"SimpleParquetLoader failed to find data. Trying DB handler with explicit table name: {feature_table_name}")
                eval_df = self.db_handler.load_dataframe(dataset_name, split, feature_table_name)

            if eval_df is None or eval_df.empty:
                raise ValueError(
                    f"Failed to load evaluation feature data for {dataset_name}/{split}/{suffix} (tried loader and DB: {feature_table_name})")

            logger.info(f"Loaded {len(eval_df)} rows for evaluation.")

            # Handle potential NaNs in loaded features (important before cleaning/extraction)
            eval_df = _handle_nan_values(eval_df, context=f"evaluate_load_{split}_{suffix}")

        except FileNotFoundError:
            logger.error(
                f"Evaluation feature file not found for {dataset_name}/{split}/{suffix}. Table name searched: {feature_table_name}. Cannot evaluate.")
            return {}
        except Exception as e:
            logger.error(f"Error loading evaluation data: {e}", exc_info=True)
            return {}

        # Clean data (primarily handles labels using 'label' column)
        cleaned_data = clean_dataset(eval_df)
        if cleaned_data is None:
            logger.error("Evaluation data became empty or invalid after cleaning.")
            return {}
        df_cleaned, y_true = cleaned_data  # y_true should be numeric labels

        if df_cleaned.empty or y_true is None or len(y_true) == 0:
            logger.warning("No valid evaluation samples left after cleaning.")
            # Return zero metrics or an empty dict
            return {'status': 'No valid samples after cleaning', 'accuracy': 0.0, 'f1': 0.0}

        # Extract features using the *loaded* and *fitted* pipeline
        X_eval = None
        try:
            logger.debug(f"Extracting evaluation features using the loaded pipeline...")
            # extract_features expects DataFrame with text and syntactic columns
            # Ensure df_cleaned has the necessary columns ('premise_text', 'hypothesis_text', and syntactic features)
            required_cols_for_extract = ['premise_text', 'hypothesis_text'] + (self._syntactic_feature_columns or [])
            missing_cols = [col for col in required_cols_for_extract if col not in df_cleaned.columns]
            if missing_cols:
                # This indicates the loaded feature file might be incomplete
                raise ValueError(
                    f"Evaluation data (df_cleaned) is missing required columns for feature extraction: {missing_cols}. "
                    f"Ensure the loaded file '{feature_table_name}' contains text and all syntactic features.")

            X_eval = self.extract_features(df_cleaned)  # Use the instance's extract_features

            # --- Alignment Check ---
            # Ensure y_true aligns with X_eval if extract_features implicitly dropped rows
            if X_eval.shape[0] != len(y_true):
                # This usually indicates an issue in data loading or feature extraction alignment
                raise ValueError(
                    f"Evaluation feature/label count mismatch after extraction: "
                    f"X_eval={X_eval.shape[0]}, y_true={len(y_true)}. Check data loading and feature extraction steps."
                )

            # Handle case where model expects 0 features (unlikely for this model but good practice)
            expected_features_dim = self.model.coef_.shape[1]  # Get expected dims from trained model
            if X_eval.shape[1] != expected_features_dim:
                raise ValueError(
                    f"Evaluation feature dimension ({X_eval.shape[1]}) != trained model dimension ({expected_features_dim})")

        except Exception as e:
            logger.error(f"Error extracting features during evaluation: {e}", exc_info=True)
            return {'status': f'Feature extraction error: {e}'}  # Return error status

        # Ensure y_true is a numpy array for metrics calculation
        y_true_np = y_true.values if isinstance(y_true, pd.Series) else np.array(y_true)

        # Perform evaluation using the helper function
        # Pass `self` which contains the `predict` method using the trained model
        logger.info("Calculating evaluation metrics...")
        eval_time, metrics = _evaluate_model_performance(self, X_eval, y_true_np)
        metrics['eval_time'] = eval_time  # Add timing info

        logger.info(f"Evaluation complete for {self.__class__.__name__} on {split}. Metrics: {metrics}")
        return metrics

    def save(self, directory: str, model_name_base: str) -> None:
        """Saves the trained Logistic Regression model and the fitted feature pipeline."""
        if not self.is_trained or not self.feature_pipeline:
            logger.warning("Attempting to save an untrained model or unfitted pipeline.")
            # Optionally raise error: raise RuntimeError("Cannot save untrained model/pipeline")
            return  # Or proceed to save potentially empty state

        # Construct full paths
        model_path = os.path.join(directory, f"{model_name_base}_model.joblib")
        pipeline_path = os.path.join(directory, f"{model_name_base}_pipeline.joblib")
        metadata_path = os.path.join(directory, f"{model_name_base}_metadata.joblib")

        logger.info(f"Saving Exp3 model to {model_path}")
        logger.info(f"Saving Exp3 pipeline to {pipeline_path}")
        logger.info(f"Saving Exp3 metadata to {metadata_path}")

        # Prepare metadata to save
        metadata = {
            'C': self.C,
            'max_iter': self.max_iter,
            'tfidf_max_features': self.tfidf_max_features,
            'tfidf_ngram_range': self.tfidf_ngram_range,
            # Crucially, save the list of syntactic feature names the pipeline was fitted with
            '_syntactic_feature_columns': self._syntactic_feature_columns
        }

        try:
            os.makedirs(directory, exist_ok=True)  # Ensure directory exists
            joblib.dump(self.model, model_path)
            joblib.dump(self.feature_pipeline, pipeline_path)
            joblib.dump(metadata, metadata_path)
            logger.info("Exp3 Model, Pipeline, and Metadata saved successfully.")
        except Exception as e:
            logger.error(f"Error saving model/pipeline/metadata: {e}", exc_info=True)
            # Optional: Clean up partially saved files
            if os.path.exists(model_path): os.remove(model_path)
            if os.path.exists(pipeline_path): os.remove(pipeline_path)
            if os.path.exists(metadata_path): os.remove(metadata_path)
            raise  # Re-raise exception

    @classmethod
    def load(cls, directory: str, model_name_base: str) -> 'LogisticTFIDFSyntacticExperiment3':
        """Loads the model, pipeline, and metadata."""
        # Construct full paths
        model_path = os.path.join(directory, f"{model_name_base}_model.joblib")
        pipeline_path = os.path.join(directory, f"{model_name_base}_pipeline.joblib")
        metadata_path = os.path.join(directory, f"{model_name_base}_metadata.joblib")

        logger.info(f"Loading Exp3 model from {model_path}")
        logger.info(f"Loading Exp3 pipeline from {pipeline_path}")
        logger.info(f"Loading Exp3 metadata from {metadata_path}")

        if not all(os.path.exists(p) for p in [model_path, pipeline_path, metadata_path]):
            raise FileNotFoundError(
                f"One or more required files (model, pipeline, metadata) not found for base name '{model_name_base}' in directory: {directory}")

        try:
            loaded_model = joblib.load(model_path)
            loaded_pipeline = joblib.load(pipeline_path)
            loaded_metadata = joblib.load(metadata_path)
        except Exception as e:
            logger.error(f"Error loading model/pipeline/metadata files: {e}", exc_info=True)
            raise

        # Re-instantiate the class using loaded metadata
        instance = cls(
            # Pass None for args, let metadata override
            args=None,
            # Provide hyperparameters from metadata
            C=loaded_metadata.get('C', 1.0),
            max_iter=loaded_metadata.get('max_iter', 1000),
            tfidf_max_features=loaded_metadata.get('tfidf_max_features', 10000),
            tfidf_ngram_range=loaded_metadata.get('tfidf_ngram_range', (1, 2))
        )
        instance.model = loaded_model
        instance.feature_pipeline = loaded_pipeline
        # Restore the crucial list of syntactic feature columns
        instance._syntactic_feature_columns = loaded_metadata.get('_syntactic_feature_columns')
        instance.is_trained = True  # Assume loaded model is trained

        # --- Post-load validation (optional but recommended) ---
        if not isinstance(instance.model, LogisticRegression):
            logger.warning(f"Loaded model is not a LogisticRegression instance ({type(instance.model)}).")
        if not isinstance(instance.feature_pipeline, Pipeline):
            logger.warning(
                f"Loaded feature pipeline is not a scikit-learn Pipeline instance ({type(instance.feature_pipeline)}).")
        if instance._syntactic_feature_columns is None:
            logger.warning(
                "Loaded model metadata did not contain '_syntactic_feature_columns'. This might cause issues during feature extraction.")
        else:
            logger.info(
                f"Restored {len(instance._syntactic_feature_columns)} syntactic feature columns expected by the pipeline.")

        logger.info("Exp3 Model loaded successfully.")
        return instance
