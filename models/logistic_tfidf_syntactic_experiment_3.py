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
# Import helpers specifically used for baseline models
from .baseline_base import clean_dataset, prepare_labels, _evaluate_model_performance  # Use existing helpers

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
        return X_processed.reindex(columns=self.columns_seen_during_fit_, fill_value=0)


class LogisticTFIDFSyntacticExperiment3(NLIModel):
    """
    Experiment 3: Logistic Regression combining TF-IDF (from raw text) and
                  pre-computed hand-crafted syntactic features.
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000,
                 tfidf_max_features: Optional[int] = 10000, tfidf_ngram_range: Tuple[int, int] = (1, 2)):
        self.C = C
        self.max_iter = max_iter
        self.tfidf_max_features = tfidf_max_features
        self.tfidf_ngram_range = tfidf_ngram_range
        self.model = LogisticRegression(C=self.C, max_iter=self.max_iter, solver='liblinear', random_state=42)
        self.feature_pipeline: Optional[Pipeline] = None
        self.db_handler = DatabaseHandler()
        self.is_trained = False
        self._syntactic_feature_columns: Optional[List[str]] = None  # Store names of syntactic features used

    def _load_and_prepare_data(self, dataset: str, split: str, suffix: str) -> Optional[
        Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]]:
        """Loads raw text and precomputed feature data, cleans, and aligns them."""
        logger.info(f"Exp3: Loading data for {dataset}/{split} (suffix: {suffix})")

        # 1. Load raw text data (using the helper method adapted from TextBaselineModel)
        # We need premise_text, hypothesis_text, pair_id, label
        try:
            # Construct the expected intermediate table names
            pairs_table = f"pairs_{suffix}"
            sentences_table = f"sentences_{suffix}"

            pairs_df = self.db_handler.load_dataframe(dataset, split, pairs_table)
            sentences_df = self.db_handler.load_dataframe(dataset, split, sentences_table)

            if pairs_df.empty or sentences_df.empty:
                raise ValueError(f"Intermediate data missing for {dataset}/{split}/{suffix}")

            logger.debug("Merging intermediate pairs and sentences data...")
            sentences_premise = sentences_df[['id', 'text']].rename(columns={'text': 'premise_text', 'id': 'p_id'})
            sentences_hypothesis = sentences_df[['id', 'text']].rename(
                columns={'text': 'hypothesis_text', 'id': 'h_id'})
            pairs_essential = pairs_df[['id', 'premise_id', 'hypothesis_id', 'label']].rename(columns={'id': 'pair_id'})

            text_df = pd.merge(pairs_essential, sentences_premise, left_on='premise_id', right_on='p_id', how='left')
            text_df = pd.merge(text_df, sentences_hypothesis, left_on='hypothesis_id', right_on='h_id', how='left')

            final_cols = ['pair_id', 'premise_text', 'hypothesis_text', 'label']
            if not all(col in text_df.columns for col in
                       ['pair_id', 'premise_id', 'hypothesis_id', 'premise_text', 'hypothesis_text', 'label']):
                missing = [col for col in ['pair_id', 'premise_text', 'hypothesis_text', 'label'] if
                           col not in text_df.columns]
                logger.error(f"Columns missing after text merge: {missing}.")
                return None
            text_df = text_df[final_cols].fillna('')  # Fill NaNs in text cols

            logger.info(f"Successfully loaded and merged text data. Shape: {text_df.shape}")

        except Exception as e:
            logger.error(f"Error loading/merging intermediate text data for {dataset}/{split}/{suffix}: {e}",
                         exc_info=True)
            return None

        # 2. Load precomputed features (expecting the file with both lexical and syntactic)
        # The filename convention should match what feature_extractor saves and db_handler loads
        precomputed_feature_table_name = f"{dataset}_{split}_features_lexical_syntactic_{suffix}"
        try:
            features_df = self.db_handler.load_dataframe(dataset, split, precomputed_feature_table_name)
            features_df = _handle_nan_values(features_df, f"{dataset}/{split}/{suffix}_features")  # Handle NaNs
            if 'pair_id' not in features_df.columns:
                raise ValueError("'pair_id' missing from precomputed features file.")
            logger.info(f"Successfully loaded precomputed features. Shape: {features_df.shape}")
        except Exception as e:
            logger.error(f"Error loading precomputed features from {precomputed_feature_table_name}: {e}",
                         exc_info=True)
            return None

        # 3. Clean and Align
        # Clean the text dataframe (handles labels)
        clean_text_result = clean_dataset(text_df)
        if not clean_text_result:
            logger.error("Text data invalid after cleaning.")
            return None
        text_df_clean, y = clean_text_result

        # Clean the features dataframe (handles labels, aligns labels if different names used)
        clean_feat_result = clean_dataset(features_df)
        if not clean_feat_result:
            logger.error("Feature data invalid after cleaning.")
            return None
        features_df_clean, _ = clean_feat_result  # We use labels from text_df

        # 4. Merge text and features based on pair_id
        # Keep only necessary columns before merge to avoid conflicts
        text_to_merge = text_df_clean[['pair_id', 'premise_text', 'hypothesis_text']]
        # We need pair_id and all feature columns from features_df_clean
        features_to_merge = features_df_clean.drop(columns=['label', 'gold_label'], errors='ignore')

        # Perform the merge, ensuring alignment
        combined_df = pd.merge(text_to_merge, features_to_merge, on='pair_id', how='inner')
        num_merged = len(combined_df)
        logger.info(f"Merged text and features: {num_merged} rows.")

        if num_merged == 0:
            logger.error("No rows remained after merging text and feature data. Check 'pair_id' alignment.")
            return None

        # Re-align labels 'y' based on the final 'pair_id's in combined_df
        final_pair_ids = combined_df['pair_id'].tolist()
        # Create a mapping from pair_id to label from the cleaned text data
        label_map = dict(zip(text_df_clean['pair_id'], y))
        final_y = np.array([label_map.get(pid) for pid in final_pair_ids])

        if len(final_y) != num_merged:
            logger.error("Label alignment failed after merge.")
            return None

        # Separate data for pipeline input: DataFrame and labels
        X_df = combined_df  # DataFrame contains text and features

        return X_df, y  # Return DataFrame and aligned labels

    def _build_feature_pipeline(self, sample_df_for_fitting: Optional[pd.DataFrame] = None) -> Pipeline:
        """Builds the scikit-learn pipeline with FeatureUnion."""
        logger.info("Building feature pipeline...")

        # TF-IDF part for premise + hypothesis
        tfidf_premise = Pipeline([
            ('selector', TextSelector(key='premise_text')),
            ('tfidf', TfidfVectorizer(max_features=self.tfidf_max_features // 2,  # Split features
                                      ngram_range=self.tfidf_ngram_range,
                                      stop_words='english'))
        ])
        tfidf_hypothesis = Pipeline([
            ('selector', TextSelector(key='hypothesis_text')),
            ('tfidf', TfidfVectorizer(max_features=self.tfidf_max_features // 2,  # Split features
                                      ngram_range=self.tfidf_ngram_range,
                                      stop_words='english'))
        ])

        # Syntactic features part
        # If fitting pipeline, discover syntactic columns now
        if sample_df_for_fitting is not None and self._syntactic_feature_columns is None:
            self._syntactic_feature_columns = filter_syntactic_features(sample_df_for_fitting)
            logger.info(f"Identified {len(self._syntactic_feature_columns)} syntactic feature columns for pipeline.")
        elif self._syntactic_feature_columns is None:
            # This should not happen if called correctly during training, but handle defensively
            raise RuntimeError("Syntactic feature columns not identified before building prediction pipeline.")

        syntactic_pipe = Pipeline([
            ('selector', FeatureSelector(column_names=self._syntactic_feature_columns)),
            # Optional: Scale syntactic features
            ('scaler', StandardScaler(with_mean=False))  # StandardScaler handles sparse matrices if needed
        ])

        # Combine using FeatureUnion
        combined_features = FeatureUnion([
            ('tfidf_premise', tfidf_premise),
            ('tfidf_hypothesis', tfidf_hypothesis),
            ('syntactic', syntactic_pipe)
        ])

        # Full pipeline (Feature Extraction -> Model)
        # The model is applied *after* the pipeline transforms the data
        pipeline = Pipeline([('features', combined_features)])
        # Removed model from pipeline ('clf', self.model)])
        logger.info("Feature pipeline built.")
        return pipeline

    def train(self, train_dataset: str, train_split: str, train_suffix: str,
              val_dataset: Optional[str] = None, val_split: Optional[str] = None, val_suffix: Optional[str] = None) -> \
    Dict[str, Any]:
        """Trains the model using combined features."""
        logger.info(f"Starting Exp3 training for {train_dataset}/{train_split}/{train_suffix}")
        start_time = time.time()

        # 1. Load and prepare training data
        train_prep_result = self._load_and_prepare_data(train_dataset, train_split, train_suffix)
        if not train_prep_result:
            logger.error("Failed to load/prepare training data.")
            return {'error': 'Training data loading failed.'}
        X_train_df, y_train = train_prep_result

        # 2. Build and fit the feature pipeline
        self.feature_pipeline = self._build_feature_pipeline(sample_df_for_fitting=X_train_df)
        logger.info("Fitting feature pipeline on training data...")
        try:
            X_train_transformed = self.feature_pipeline.fit_transform(X_train_df, y_train)
            logger.info(f"Training data transformed. Shape: {X_train_transformed.shape}")
        except Exception as e:
            logger.error(f"Error fitting/transforming training data with pipeline: {e}", exc_info=True)
            return {'error': 'Feature pipeline fitting failed.'}

        # 3. Train the Logistic Regression model
        logger.info("Training Logistic Regression model...")
        self.model.fit(X_train_transformed, y_train)
        self.is_trained = True
        train_time = time.time() - start_time
        logger.info(f"Model training completed in {train_time:.2f}s.")

        # 4. Evaluate on validation data (if provided)
        eval_metrics = {}
        eval_time = 0.0
        if val_dataset and val_split and val_suffix:
            logger.info(f"Loading and evaluating on validation data: {val_dataset}/{val_split}/{val_suffix}")
            val_prep_result = self._load_and_prepare_data(val_dataset, val_split, val_suffix)
            if val_prep_result:
                X_val_df, y_val = val_prep_result
                try:
                    logger.info("Transforming validation data...")
                    X_val_transformed = self.feature_pipeline.transform(X_val_df)  # Use transform, not fit_transform
                    logger.info("Evaluating model on validation data...")
                    eval_time, metrics = _evaluate_model_performance(self, X_val_transformed,
                                                                     y_val)  # Pass self (as NLIModel)
                    eval_metrics = metrics
                    eval_metrics['eval_time'] = eval_time
                except Exception as e:
                    logger.error(f"Error during validation evaluation: {e}", exc_info=True)
                    eval_metrics = {'error': 'Validation evaluation failed.'}
            else:
                logger.warning("Failed to load/prepare validation data. Skipping validation.")
                eval_metrics = {'warning': 'Validation data failed to load.'}
        else:
            logger.info("No validation data provided. Skipping validation.")

        results = {
            'train_time': train_time,
            **eval_metrics
        }
        return results

    # Implement NLIModel abstract methods
    def extract_features(self, data: pd.DataFrame) -> Any:
        """Transforms data using the *fitted* feature pipeline."""
        if not self.feature_pipeline or not self.is_trained:  # Check if pipeline is fitted (implicitly via is_trained)
            raise RuntimeError("Feature pipeline is not fitted. Train the model first.")
        logger.debug("Extracting features using fitted pipeline...")
        # Ensure input is DataFrame as expected by pipeline
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input data for feature extraction must be a pandas DataFrame.")
        return self.feature_pipeline.transform(data)

    def predict(self, X: Any) -> np.ndarray:
        """Predicts labels for transformed feature data."""
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet.")
        # Note: X here should be the *already transformed* features
        logger.debug(f"Predicting labels for {X.shape[0]} samples...")
        return self.model.predict(X)

    # Overload predict to handle DataFrame input for convenience (extracts then predicts)
    def predict_on_dataframe(self, data_df: pd.DataFrame) -> np.ndarray:
        """Loads data, extracts features, and predicts."""
        if not isinstance(data_df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        X_transformed = self.extract_features(data_df)  # Uses the fitted pipeline
        return self.predict(X_transformed)

    def save(self, filepath: str, model_name) -> None:
        """Saves the trained Logistic Regression model and the fitted feature pipeline."""
        if not self.is_trained or not self.feature_pipeline:
            logger.warning("Attempting to save an untrained model or unfitted pipeline.")
            return
        # Filepath should be the base path, e.g., /path/to/model_exp3
        model_path = f"{filepath}_model.joblib"
        pipeline_path = f"{filepath}_pipeline.joblib"
        metadata_path = f"{filepath}_metadata.joblib"  # Store hyperparameters and feature names

        logger.info(f"Saving Exp3 model to {model_path}")
        logger.info(f"Saving Exp3 pipeline to {pipeline_path}")
        logger.info(f"Saving Exp3 metadata to {metadata_path}")

        metadata = {
            'C': self.C,
            'max_iter': self.max_iter,
            'tfidf_max_features': self.tfidf_max_features,
            'tfidf_ngram_range': self.tfidf_ngram_range,
            '_syntactic_feature_columns': self._syntactic_feature_columns
        }

        try:
            joblib.dump(self.model, model_path)
            joblib.dump(self.feature_pipeline, pipeline_path)
            joblib.dump(metadata, metadata_path)
        except Exception as e:
            logger.error(f"Error saving model/pipeline/metadata: {e}", exc_info=True)

    @classmethod
    def load(cls, filepath: str, feature_extractor: Optional[Any] = None) -> 'LogisticTFIDFSyntacticExperiment3':
        """Loads the model, pipeline, and metadata."""
        # Filepath is the base path
        model_path = f"{filepath}_model.joblib"
        pipeline_path = f"{filepath}_pipeline.joblib"
        metadata_path = f"{filepath}_metadata.joblib"

        logger.info(f"Loading Exp3 model from {model_path}")
        logger.info(f"Loading Exp3 pipeline from {pipeline_path}")
        logger.info(f"Loading Exp3 metadata from {metadata_path}")

        if not all(os.path.exists(p) for p in [model_path, pipeline_path, metadata_path]):
            raise FileNotFoundError(f"Model, pipeline, or metadata file missing for base path: {filepath}")

        try:
            loaded_model = joblib.load(model_path)
            loaded_pipeline = joblib.load(pipeline_path)
            loaded_metadata = joblib.load(metadata_path)
        except Exception as e:
            logger.error(f"Error loading model/pipeline/metadata files: {e}", exc_info=True)
            raise

        # Re-instantiate the class with loaded parameters
        instance = cls(
            C=loaded_metadata.get('C', 1.0),
            max_iter=loaded_metadata.get('max_iter', 1000),
            tfidf_max_features=loaded_metadata.get('tfidf_max_features', 10000),
            tfidf_ngram_range=loaded_metadata.get('tfidf_ngram_range', (1, 2))
        )
        instance.model = loaded_model
        instance.feature_pipeline = loaded_pipeline
        instance._syntactic_feature_columns = loaded_metadata.get('_syntactic_feature_columns')
        instance.is_trained = True  # Assume loaded model is trained

        if instance._syntactic_feature_columns is None:
            logger.warning("Loaded model metadata did not contain syntactic feature column names.")

        logger.info("Exp3 Model loaded successfully.")
        return instance
