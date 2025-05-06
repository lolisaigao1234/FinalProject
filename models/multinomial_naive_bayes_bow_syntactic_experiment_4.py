# IS567FP/models/multinomial_naive_bayes_bow_syntactic_experiment_4.py
import logging
import os
from typing import Tuple, Optional, Any, Dict, List
import pandas as pd
import numpy as np
import joblib
import time
from scipy.sparse import hstack, csr_matrix

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # Import MinMaxScaler for non-negative scaling
from sklearn.base import BaseEstimator, TransformerMixin

# Import base classes and helpers from the project structure
from utils.common import NLIModel
from utils.database import DatabaseHandler
# Import helpers specifically used for baseline models
from .baseline_base import clean_dataset, _evaluate_model_performance, _handle_nan_values, \
    filter_syntactic_features, SimpleParquetLoader  # Use existing helpers

logger = logging.getLogger(__name__)


# --- Helper Transformers (copied from Experiment 3) ---
class TextSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.key]


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Selects precomputed feature columns specified during initialization.
    During fit, it identifies which of the specified columns are actually
    present in the training data.
    During transform, it ensures only those fitted columns are present,
    adding any missing ones (that were seen during fit) with a value of 0.
    Handles potential inconsistencies between fit/transform columns.
    """

    def __init__(self, column_names: List[str]):
        self.potential_column_names = column_names
        self.columns_seen_during_fit_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame for fit.")

        self.columns_seen_during_fit_ = [col for col in self.potential_column_names if col in X.columns]

        missing_at_fit = set(self.potential_column_names) - set(self.columns_seen_during_fit_)
        if missing_at_fit:
            logger.warning(
                f"FeatureSelector fit: Potential columns {missing_at_fit} not found in training data. Selector will only use columns found: {self.columns_seen_during_fit_}")
        else:
             logger.debug( # Changed to debug to reduce noise
                f"FeatureSelector fit: All {len(self.columns_seen_during_fit_)} potential columns were found in the training data.")

        if not self.columns_seen_during_fit_:
            logger.error("FeatureSelector fit: No specified columns found in the training data!")
            # Consider raising an error if no features are found
            # raise ValueError("No specified columns found in the training data during fit.")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.columns_seen_during_fit_ is None:
            raise RuntimeError("FeatureSelector must be fitted before transforming data.")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame for transform.")

        # Use columns identified during fit
        cols_to_use = self.columns_seen_during_fit_

        # Prepare a DataFrame with only the required columns, filling missing ones with 0
        # Use reindex for robust column handling (adds missing, removes extra, orders correctly)
        # Important: Create a copy to avoid modifying original if X is used elsewhere
        X_processed = X.reindex(columns=cols_to_use, fill_value=0)

        # Log if any columns were added (missing in input X but present in fit)
        missing_for_transform = set(cols_to_use) - set(X.columns)
        if missing_for_transform:
             logger.warning(
                f"FeatureSelector transform: Columns {missing_for_transform} (seen during fit) were missing in the input data and have been added with value 0.")

        return X_processed


class SparseScaler(BaseEstimator, TransformerMixin):
    def __init__(self, scaler=MinMaxScaler()):
        # Default to MinMaxScaler as MNB requires non-negative features
        self.scaler = scaler

    def fit(self, X, y=None):
        # Check if input is sparse, handle appropriately
        if isinstance(X, csr_matrix):
            # MinMaxScaler can handle sparse data directly
            self.scaler.fit(X)
        elif isinstance(X, pd.DataFrame) or isinstance(X, np.ndarray):
            # If dense, standard fit
            self.scaler.fit(X)
        else:
            raise TypeError(f"Unsupported data type for SparseScaler fit: {type(X)}")
        return self

    def transform(self, X):
        return self.scaler.transform(X)


# --- Experiment 4 Model Class ---
class MultinomialNaiveBayesBowSyntacticExperiment4(NLIModel):
    # *** MODIFIED __init__ to accept args ***
    def __init__(self, args=None, **kwargs): # Accept args and kwargs
        # Extract hyperparameters from args, providing defaults
        # Default values match the previous explicit defaults
        self.alpha = getattr(args, 'alpha', 1.0) if args else 1.0
        self.bow_max_features = getattr(args, 'bow_max_features', 10000) if args else 10000
        # Provide default tuple if bow_ngram_range is not in args
        default_ngram_range = (1, 1)
        self.bow_ngram_range = getattr(args, 'bow_ngram_range', default_ngram_range) if args else default_ngram_range
        # Ensure ngram_range is a tuple if extracted from args
        if isinstance(self.bow_ngram_range, list):
             self.bow_ngram_range = tuple(self.bow_ngram_range)
        elif not isinstance(self.bow_ngram_range, tuple):
             logger.warning(f"bow_ngram_range from args is not a list or tuple ({type(self.bow_ngram_range)}). Using default {default_ngram_range}.")
             self.bow_ngram_range = default_ngram_range

        self.scale_syntactic = getattr(args, 'scale_syntactic', True) if args else True

        # Initialize model with extracted/default hyperparameters
        self.model = MultinomialNB(alpha=self.alpha)
        self.feature_pipeline: Optional[Pipeline] = None
        self.db_handler = DatabaseHandler()
        self.is_trained = False
        self._syntactic_feature_columns: Optional[List[str]] = None

        logger.info(f"Initialized MultinomialNaiveBayesBowSyntacticExperiment4 with: alpha={self.alpha}, bow_max_features={self.bow_max_features}, bow_ngram_range={self.bow_ngram_range}, scale_syntactic={self.scale_syntactic}")
    # *** END MODIFICATION ***


    # Data loading and preparation (similar to Experiment 3)
    def _load_and_prepare_data(self, dataset: str, split: str, suffix: str) -> Optional[
        Tuple[pd.DataFrame, np.ndarray]]:
        logger.info(f"Exp4: Loading data for {dataset}/{split} (suffix: {suffix})")

        # 1. Load raw text data (premise, hypothesis, pair_id, label)
        try:
            pairs_table = f"pairs_{suffix}"
            sentences_table = f"sentences_{suffix}"
            pairs_df = self.db_handler.load_dataframe(dataset, split, pairs_table)
            sentences_df = self.db_handler.load_dataframe(dataset, split, sentences_table)
            if pairs_df is None or sentences_df is None or pairs_df.empty or sentences_df.empty:
                 raise ValueError(f"Intermediate data missing or empty: {pairs_table} or {sentences_table}")

            # Ensure columns exist before renaming/selecting
            if not all(col in sentences_df.columns for col in ['id', 'text']):
                raise ValueError(f"Missing 'id' or 'text' in {sentences_table}")
            if not all(col in pairs_df.columns for col in ['id', 'premise_id', 'hypothesis_id', 'label']):
                 raise ValueError(f"Missing required columns in {pairs_table}")

            sentences_premise = sentences_df[['id', 'text']].rename(columns={'text': 'premise_text', 'id': 'p_id'})
            sentences_hypothesis = sentences_df[['id', 'text']].rename(
                columns={'text': 'hypothesis_text', 'id': 'h_id'})
            pairs_essential = pairs_df[['id', 'premise_id', 'hypothesis_id', 'label']].rename(columns={'id': 'pair_id'})

            # Ensure merge columns are of compatible types (e.g., string)
            pairs_essential['premise_id'] = pairs_essential['premise_id'].astype(str)
            pairs_essential['hypothesis_id'] = pairs_essential['hypothesis_id'].astype(str)
            sentences_premise['p_id'] = sentences_premise['p_id'].astype(str)
            sentences_hypothesis['h_id'] = sentences_hypothesis['h_id'].astype(str)


            text_df = pd.merge(pairs_essential, sentences_premise, left_on='premise_id', right_on='p_id',
                               how='left').drop('p_id', axis=1, errors='ignore')
            text_df = pd.merge(text_df, sentences_hypothesis, left_on='hypothesis_id', right_on='h_id',
                               how='left').drop('h_id', axis=1, errors='ignore')

            final_text_cols = ['pair_id', 'premise_text', 'hypothesis_text', 'label']
            if not all(col in text_df.columns for col in final_text_cols):
                missing = [col for col in final_text_cols if col not in text_df.columns]
                raise ValueError(f"Columns missing after text merge: {missing}.")
            text_df = text_df[final_text_cols].fillna('') # Fill NaNs in text columns *after* merge
            logger.info(f"Exp4: Successfully loaded and merged text data. Shape: {text_df.shape}")

        except Exception as e:
            logger.error(f"Exp4: Error loading/merging intermediate text data for {dataset}/{split}/{suffix}: {e}",
                         exc_info=True)
            return None

        # 2. Load precomputed features (Need stats+syntactic for this exp)
        # Assuming FeatureExtractor saves them together under this name
        precomputed_feature_table_name = f"{dataset}_{split}_features_stats_syntactic_{suffix}"
        try:
            # Use SimpleParquetLoader to load features (which might fall back to DB)
            loader = SimpleParquetLoader()
            # Pass necessary context for loader (self might not be needed if loader is static/standalone)
            features_df = loader.load_data(self, dataset, split, suffix)

            if features_df is None or features_df.empty:
                 # Try DB handler directly if loader fails (adjust table name if necessary)
                 logger.warning(f"SimpleParquetLoader failed. Trying DB handler with table: {precomputed_feature_table_name}")
                 features_df = self.db_handler.load_dataframe(dataset, split, precomputed_feature_table_name)

            if features_df is None or features_df.empty:
                 raise FileNotFoundError(f"Failed to load features from {precomputed_feature_table_name} via Loader and DB.")

            features_df = _handle_nan_values(features_df, f"{dataset}/{split}/{suffix}_features")
            if 'pair_id' not in features_df.columns: raise ValueError("'pair_id' missing in features file.")
            logger.info(f"Exp4: Successfully loaded precomputed features. Shape: {features_df.shape}")
        except Exception as e:
            logger.error(f"Exp4: Error loading precomputed features from {precomputed_feature_table_name} (or fallback): {e}",
                         exc_info=True)
            return None

        # 3. Clean and Align Labels
        # Clean text data first to get the correct set of pair_ids and labels (y)
        clean_text_result = clean_dataset(text_df)
        if not clean_text_result:
            logger.error("Text data invalid or empty after cleaning labels.")
            return None
        text_df_clean, y_labels = clean_text_result # y_labels are the aligned labels for text_df_clean
        text_ids_after_clean = text_df_clean['pair_id'].astype(str).unique() # Get valid pair IDs

        # Clean features data (handles label column if present, but we mostly need pair_id alignment)
        clean_feat_result = clean_dataset(features_df)
        if not clean_feat_result:
            logger.error("Feature data invalid or empty after cleaning.")
            return None
        features_df_clean, _ = clean_feat_result # Don't need labels from here

        # 4. Filter ONLY syntactic features needed for this experiment
        # Ensure features_df_clean has columns to filter
        if features_df_clean.empty:
            logger.error("Features DataFrame became empty after cleaning. Cannot filter syntactic features.")
            return None

        self._syntactic_feature_columns = filter_syntactic_features(features_df_clean)
        if not self._syntactic_feature_columns:
            logger.error("No syntactic feature columns identified in the precomputed features file.")
            # If only BoW is desired, could proceed, but experiment combines BoW + Syntactic
            return None
        logger.info(f"Exp4: Keeping {len(self._syntactic_feature_columns)} syntactic features.")
        syntactic_features_to_merge = features_df_clean[['pair_id'] + self._syntactic_feature_columns]

        # Ensure pair_id is string for merging
        syntactic_features_to_merge['pair_id'] = syntactic_features_to_merge['pair_id'].astype(str)

        # 5. Merge text and *filtered* syntactic features using the cleaned text data
        # Merge ON the pair_ids that survived text cleaning
        text_to_merge = text_df_clean[['pair_id', 'premise_text', 'hypothesis_text']]
        text_to_merge['pair_id'] = text_to_merge['pair_id'].astype(str)

        combined_df = pd.merge(text_to_merge, syntactic_features_to_merge, on='pair_id', how='inner')
        num_merged = len(combined_df)
        logger.info(f"Exp4: Merged text and filtered syntactic features: {num_merged} rows.")

        if num_merged == 0:
            logger.error("No rows remained after merging cleaned text and syntactic features.")
            return None

        # Re-align labels 'y' using the pair_ids present *after* the merge
        # Create a map from the originally cleaned text data labels
        label_map = dict(zip(text_df_clean['pair_id'].astype(str), y_labels))
        final_pair_ids = combined_df['pair_id'].tolist() # Already string type from merge
        # Ensure labels exist in the map before assignment
        final_y = np.array([label_map[pid] for pid in final_pair_ids if pid in label_map])

        if len(final_y) != num_merged:
            logger.error(f"Label alignment failed after final merge. Expected {num_merged}, got {len(final_y)}.")
            # This indicates a potential issue in pair_id matching or the label map creation
            return None

        # Final DataFrame X should contain text and syntactic columns
        X_df = combined_df # Contains pair_id, premise_text, hypothesis_text, and syntactic features
        return X_df, final_y


    # Feature pipeline building
    def _build_feature_pipeline(self, sample_df_for_fitting: Optional[pd.DataFrame] = None) -> Pipeline:
        logger.info("Exp4: Building feature pipeline...")

        # 1. Bag-of-Words (BoW) pipeline for premise and hypothesis separately
        # Split max_features if specified
        max_feat_per_part = (self.bow_max_features // 2) if self.bow_max_features else None
        logger.info(f"Using max_features per text part (premise/hypothesis): {max_feat_per_part}")

        bow_premise_pipe = Pipeline([
            ('selector', TextSelector(key='premise_text')),
            ('bow', CountVectorizer(max_features=max_feat_per_part,
                                    ngram_range=self.bow_ngram_range, stop_words='english', binary=False))
        ])
        bow_hypothesis_pipe = Pipeline([
            ('selector', TextSelector(key='hypothesis_text')),
            ('bow', CountVectorizer(max_features=max_feat_per_part,
                                    ngram_range=self.bow_ngram_range, stop_words='english', binary=False))
        ])

        # 2. Syntactic features pipeline
        # Ensure syntactic columns are known (should be set during _load_and_prepare_data)
        if self._syntactic_feature_columns is None:
            if sample_df_for_fitting is None:
                raise RuntimeError("Syntactic features missing and no sample data provided to infer them.")
            logger.warning("Re-inferring syntactic columns during pipeline build.")
            # Ensure sample_df_for_fitting is not empty
            if sample_df_for_fitting.empty:
                 raise ValueError("Cannot infer syntactic columns from empty sample_df_for_fitting.")
            self._syntactic_feature_columns = filter_syntactic_features(sample_df_for_fitting)
            if not self._syntactic_feature_columns:
                # If still no columns, the pipeline part will be empty but won't crash
                 logger.warning("No syntactic columns found even after re-inferring.")

        # Ensure it's a list even if empty
        syntactic_cols_to_use = self._syntactic_feature_columns if self._syntactic_feature_columns else []

        syntactic_steps = [('selector', FeatureSelector(column_names=syntactic_cols_to_use))]
        if self.scale_syntactic:
            # Use MinMaxScaler to ensure non-negativity for MNB
            syntactic_steps.append(('scaler', SparseScaler(scaler=MinMaxScaler()))) # SparseScaler handles dense/sparse
            logger.info("Adding MinMaxScaler for syntactic features.")
        else:
            # MNB needs non-negative features. If raw syntactic features can be negative,
            # scaling or binning is essential. We assume for now they are non-negative or
            # scale_syntactic=True is used. Add a warning.
            logger.warning("Not scaling syntactic features. Ensure they are non-negative for MultinomialNB.")
            # Consider adding a check or ClipNegative transformer if negative values are possible and scaling is off.

        syntactic_pipe = Pipeline(syntactic_steps)

        # 3. Combine using FeatureUnion
        # Handles mixed sparse/dense inputs. MNB works with sparse.
        feature_union_list = []
        # Only add BoW if max_features allows it (or is None)
        if max_feat_per_part is None or max_feat_per_part > 0:
             feature_union_list.append(('bow_premise', bow_premise_pipe))
             feature_union_list.append(('bow_hypothesis', bow_hypothesis_pipe))
             logger.info("Including BoW features in FeatureUnion.")
        else:
             logger.warning("BoW max_features per part is 0. Excluding BoW features from pipeline.")

        # Only add syntactic if columns were found
        if syntactic_cols_to_use:
            feature_union_list.append(('syntactic', syntactic_pipe))
            logger.info(f"Including {len(syntactic_cols_to_use)} syntactic features in FeatureUnion.")
        else:
            logger.warning("No syntactic feature columns found. Excluding syntactic features from pipeline.")

        if not feature_union_list:
             raise RuntimeError("Pipeline cannot be built: No features (BoW or Syntactic) selected.")

        combined_features = FeatureUnion(feature_union_list)


        # 4. Full Pipeline (only feature extraction)
        pipeline = Pipeline([('features', combined_features)])
        logger.info("Exp4: Feature pipeline built.")
        return pipeline

    # Training method (Modified to match structure of Exp3)
    def train(self, X: Any, y: np.ndarray) -> None:
        """
        Trains the Multinomial Naive Bayes model on the *already transformed* features (X).
        Feature pipeline fitting should happen within `extract_features` called by trainer.
        """
        if X is None or y is None:
             raise ValueError("Training features (X) or labels (y) are None.")
        if X.shape[0] == 0:
             logger.warning("Received 0 training samples (X shape is 0). Cannot train.")
             self.is_trained = False
             return
        if X.shape[0] != len(y):
             raise ValueError(f"Feature/Label mismatch during training: X={X.shape}, y={len(y)}")
        if self.is_trained:
             logger.warning("Model is already marked as trained. Re-training the classifier...")
             self.is_trained = False # Reset flag

        # Check for negative values just before fitting MNB
        # Note: Checking sparse matrices for negativity can be slow.
        # Consider checking only if scale_syntactic was False.
        min_val = X.min() if hasattr(X, 'min') else 0 # Basic check
        if min_val < 0:
            logger.warning(f"WARNING: Negative values (min={min_val}) detected in features passed to MultinomialNB! This WILL cause errors. Ensure scaling or clipping is applied.")
            # Optionally, raise an error here:
            # raise ValueError("Negative values detected in features for MultinomialNB.")

        logger.info(f"Training MultinomialNB classifier on processed data with shape: {X.shape}")
        start_time = time.time()
        try:
            self.model.fit(X, y)
            self.is_trained = True # Set flag only after successful fit
            logger.info(f"MultinomialNB classifier training completed successfully.")
        except Exception as e:
             logger.error(f"Error during MultinomialNB model fitting: {e}", exc_info=True)
             self.is_trained = False # Ensure flag is False on failure
             raise # Re-raise

        train_time = time.time() - start_time
        logger.info(f"Classifier training phase completed in {train_time:.2f}s.")

    # NLIModel abstract method implementations
    # extract_features: Handles fitting pipeline if not trained, else transforms
    def extract_features(self, data: pd.DataFrame) -> Any:
        """
        Transforms input DataFrame using the feature pipeline.
        Fits the pipeline on the first call (training context) and then only transforms.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input data for feature extraction must be a pandas DataFrame.")

        # Ensure required text columns for BoW are present
        if 'premise_text' not in data.columns or 'hypothesis_text' not in data.columns:
            raise ValueError("Missing 'premise_text' or 'hypothesis_text' columns required for BoW.")

        if not self.is_trained:
            # --- Training Phase: Fit Pipeline and Transform ---
            logger.info("Model not trained yet. Assuming training context: Building/Fitting pipeline...")

            if not self.feature_pipeline:
                # Pass data to infer syntactic columns if needed
                self.feature_pipeline = self._build_feature_pipeline(sample_df_for_fitting=data)

            logger.info("Fitting feature pipeline and transforming training data...")
            start_transform_time = time.time()
            try:
                X_transformed = self.feature_pipeline.fit_transform(data, y=None) # y not needed for unsupervised parts
            except Exception as e:
                logger.error(f"Error during pipeline fit_transform in extract_features (training context): {e}", exc_info=True)
                logger.error(f"Input data columns: {data.columns.tolist()}")
                logger.error(f"Pipeline steps: {self.feature_pipeline.steps if self.feature_pipeline else 'None'}")
                raise

            transform_time = time.time() - start_transform_time
            logger.info(
                f"Pipeline fit & transform completed in {transform_time:.2f}s. Output shape: {X_transformed.shape}")

            # self.is_trained is set by the train() method after this runs
            return X_transformed

        else:
            # --- Evaluation/Prediction Phase: Transform Only ---
            logger.info("Model is trained. Assuming evaluation/prediction context: Transforming data...")
            if not self.feature_pipeline:
                raise RuntimeError("Model is marked as trained, but the feature pipeline is missing.")

            logger.debug("Transforming data using the fitted pipeline...")
            try:
                # Use transform() only
                extracted = self.feature_pipeline.transform(data)
                logger.debug(f"Feature extraction (transform) successful. Output shape: {extracted.shape}")
                return extracted
            except Exception as e:
                logger.error(f"Error during pipeline transform in extract_features (evaluation context): {e}", exc_info=True)
                logger.error(f"Syntactic columns expected by pipeline: {self._syntactic_feature_columns}")
                logger.error(f"Columns available in input data: {data.columns.tolist()}")
                raise

    def predict(self, X: Any) -> np.ndarray:
        """Predicts labels for *already transformed* feature data."""
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet.")
        if X is None:
            raise ValueError("Input features (X) for prediction are None.")
        # Check feature dimension consistency
        if hasattr(self.model, 'feature_count_') and X.shape[1] != self.model.feature_count_.shape[1]:
            raise ValueError(f"Prediction feature dimension mismatch: Input={X.shape[1]}, Expected={self.model.feature_count_.shape[1]}")

        logger.debug(f"Predicting labels for {X.shape[0]} samples...")
        predictions = self.model.predict(X)
        logger.debug("Prediction finished.")
        return predictions

    # Overload predict to handle DataFrame input
    def predict_on_dataframe(self, data_df: pd.DataFrame) -> np.ndarray:
        if not isinstance(data_df, pd.DataFrame): raise ValueError("Input must be a pandas DataFrame")
        X_transformed = self.extract_features(data_df)
        return self.predict(X_transformed)

    def save(self, directory: str, model_name_base: str) -> None:
        """Saves the trained MultinomialNB model and the fitted feature pipeline."""
        if not self.is_trained or not self.feature_pipeline:
            logger.warning("Attempting to save an untrained model or unfitted pipeline.")
            return

        model_path = os.path.join(directory, f"{model_name_base}_model.joblib")
        pipeline_path = os.path.join(directory, f"{model_name_base}_pipeline.joblib")
        metadata_path = os.path.join(directory, f"{model_name_base}_metadata.joblib")

        logger.info(f"Saving Exp4 model to {model_path}")
        logger.info(f"Saving Exp4 pipeline to {pipeline_path}")
        logger.info(f"Saving Exp4 metadata to {metadata_path}")

        # Prepare metadata including hyperparameters and feature columns
        metadata = {
            'alpha': self.alpha,
            'bow_max_features': self.bow_max_features,
            'bow_ngram_range': self.bow_ngram_range,
            'scale_syntactic': self.scale_syntactic,
            '_syntactic_feature_columns': self._syntactic_feature_columns
        }

        try:
            os.makedirs(directory, exist_ok=True)
            joblib.dump(self.model, model_path)
            joblib.dump(self.feature_pipeline, pipeline_path)
            joblib.dump(metadata, metadata_path)
            logger.info("Exp4 Model, Pipeline, and Metadata saved successfully.")
        except Exception as e:
            logger.error(f"Error saving Exp4 artifacts: {e}", exc_info=True)
            # Optional: Cleanup partial files
            if os.path.exists(model_path): os.remove(model_path)
            if os.path.exists(pipeline_path): os.remove(pipeline_path)
            if os.path.exists(metadata_path): os.remove(metadata_path)
            raise

    @classmethod
    def load(cls, directory: str, model_name_base: str) -> 'MultinomialNaiveBayesBowSyntacticExperiment4':
        """Loads the model, pipeline, and metadata."""
        model_path = os.path.join(directory, f"{model_name_base}_model.joblib")
        pipeline_path = os.path.join(directory, f"{model_name_base}_pipeline.joblib")
        metadata_path = os.path.join(directory, f"{model_name_base}_metadata.joblib")

        logger.info(f"Loading Exp4 model from {model_path}, pipeline from {pipeline_path}, metadata from {metadata_path}")
        if not all(os.path.exists(p) for p in [model_path, pipeline_path, metadata_path]):
            raise FileNotFoundError(f"Model artifacts missing for base path: {model_name_base} in {directory}")

        try:
            loaded_model = joblib.load(model_path)
            loaded_pipeline = joblib.load(pipeline_path)
            loaded_metadata = joblib.load(metadata_path)
        except Exception as e:
            logger.error(f"Error loading Exp4 artifacts: {e}", exc_info=True); raise

        # Instantiate using loaded metadata - pass None for args
        instance = cls(
            args=None, # Set args to None, use metadata
            alpha=loaded_metadata.get('alpha', 1.0),
            bow_max_features=loaded_metadata.get('bow_max_features', 10000),
            bow_ngram_range=loaded_metadata.get('bow_ngram_range', (1, 1)),
            scale_syntactic=loaded_metadata.get('scale_syntactic', True)
        )
        instance.model = loaded_model
        instance.feature_pipeline = loaded_pipeline
        instance._syntactic_feature_columns = loaded_metadata.get('_syntactic_feature_columns')
        instance.is_trained = True

        # Post-load checks
        if not isinstance(instance.model, MultinomialNB):
             logger.warning(f"Loaded model type mismatch: Expected MultinomialNB, got {type(instance.model)}")
        if not isinstance(instance.feature_pipeline, Pipeline):
             logger.warning(f"Loaded pipeline type mismatch: Expected Pipeline, got {type(instance.feature_pipeline)}")
        if instance._syntactic_feature_columns is None:
            logger.warning("Loaded Exp4 model metadata missing syntactic feature columns.")
        else:
            logger.info(f"Restored {len(instance._syntactic_feature_columns)} syntactic columns for pipeline.")

        logger.info("Exp4 Model loaded successfully.")
        return instance


    # Evaluation method (Adapted from Exp3)
    def evaluate(self, dataset_name: str, split: str, suffix: str) -> Dict[str, Any]:
        """
        Evaluates the trained MultinomialNB model on a given dataset split.
        Loads pre-computed combined features (stats+syntactic), transforms them using
        the fitted pipeline, predicts, and calculates metrics.
        """
        if not self.is_trained or not self.feature_pipeline:
            logger.error(f"Cannot evaluate model {self.__class__.__name__}: Model or pipeline is not trained/loaded.")
            return {'status': 'Evaluation failed: Model or pipeline not ready.'}

        logger.info(f"Evaluating {self.__class__.__name__} on {dataset_name}/{split}/{suffix}...")
        data_loader = SimpleParquetLoader()
        eval_df = None
        # --- Feature Loading ---
        # Assuming FeatureExtractor saves stats+syntactic together
        feature_table_name_convention = f"{dataset_name}_{split}_features_stats_syntactic_{suffix}"
        logger.info(
            f"Attempting to load evaluation features (expecting convention like: {feature_table_name_convention}.parquet)")

        try:
            # Use SimpleParquetLoader (which might fall back to DB)
            eval_df = data_loader.load_data(self, dataset_name, split, suffix) # Pass suffix

            if eval_df is None or eval_df.empty:
                # Explicitly try loading from DB if loader failed
                logger.warning(
                    f"SimpleParquetLoader failed. Trying DB handler with table: {feature_table_name_convention}")
                eval_df = self.db_handler.load_dataframe(dataset_name, split, feature_table_name_convention)

            if eval_df is None or eval_df.empty:
                raise FileNotFoundError(
                    f"Failed to load evaluation feature data for {dataset_name}/{split}/{suffix} using loader and DB ({feature_table_name_convention})")

            logger.info(f"Loaded {len(eval_df)} rows for evaluation.")
            # Handle potential NaNs in loaded features BEFORE cleaning/extraction
            eval_df = _handle_nan_values(eval_df, context=f"evaluate_load_{split}_{suffix}")

        except FileNotFoundError:
            logger.error(
                f"Evaluation feature data not found for {dataset_name}/{split}/{suffix} (tried convention: {feature_table_name_convention}). Cannot evaluate.")
            return {'status': 'Evaluation failed: Feature file not found.'}
        except Exception as e:
            logger.error(f"Error loading evaluation data: {e}", exc_info=True)
            return {'status': f'Evaluation failed: Error loading data - {e}'}

        # --- Data Cleaning (Labels) ---
        cleaned_data = clean_dataset(eval_df)
        if cleaned_data is None:
            logger.error("Evaluation data became empty or invalid after cleaning labels.")
            return {'status': 'Evaluation failed: Data invalid after cleaning.'}
        df_cleaned, y_true = cleaned_data # y_true should be numpy array of integer labels

        if df_cleaned.empty or y_true is None or len(y_true) == 0:
            logger.warning("No valid evaluation samples left after cleaning labels.")
            return {'status': 'No valid samples after cleaning', 'accuracy': 0.0, 'f1': 0.0}

        # --- Feature Extraction (Transform ONLY) ---
        X_eval = None
        try:
            logger.info(f"Transforming evaluation features using the loaded pipeline...")
            # Requires DataFrame with text and syntactic columns matching pipeline expectation
            required_cols_for_extract = ['premise_text', 'hypothesis_text'] + (self._syntactic_feature_columns or [])
            missing_cols = [col for col in required_cols_for_extract if col not in df_cleaned.columns]

            if missing_cols:
                raise ValueError(
                    f"Evaluation data (df_cleaned) is missing columns required by the feature pipeline: {missing_cols}. "
                    f"Ensure the loaded file contains text and all expected syntactic features: {self._syntactic_feature_columns}")

            X_eval = self.extract_features(df_cleaned) # Should call self.feature_pipeline.transform()

            # --- Alignment & Dimension Check ---
            if X_eval.shape[0] != len(y_true):
                raise ValueError(
                    f"Evaluation feature/label count mismatch after transform: X_eval={X_eval.shape[0]}, y_true={len(y_true)}")

            # Check feature dimensions against trained model (MultinomialNB specific check)
            if hasattr(self.model, 'feature_count_'):
                expected_features_dim = self.model.feature_count_.shape[1]
                if X_eval.shape[1] != expected_features_dim:
                    raise ValueError(
                        f"Evaluation feature dimension ({X_eval.shape[1]}) != trained model dimension ({expected_features_dim})")
            else:
                logger.warning("Could not verify evaluation feature dimensions against trained MultinomialNB model.")

        except Exception as e:
            logger.error(f"Error transforming features during evaluation: {e}", exc_info=True)
            return {'status': f'Evaluation failed: Feature transformation error - {e}'}

        # --- Calculate Metrics ---
        y_true_np = y_true if isinstance(y_true, np.ndarray) else np.array(y_true)

        # Use helper function _evaluate_model_performance
        logger.info("Calculating evaluation metrics...")
        eval_time, metrics = _evaluate_model_performance(self, X_eval, y_true_np)
        metrics['eval_time'] = eval_time

        logger.info(f"Evaluation complete for {self.__class__.__name__} on {split}. Metrics: {metrics}")
        return metrics