# Create file: IS567FP/models/gradient_boosting_tfidf_syntactic_experiment_6.py
import logging
import os
from typing import Tuple, Optional, Any, Dict, List
import pandas as pd
import numpy as np
import joblib
import time
from scipy.sparse import hstack, csr_matrix

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler # To scale syntactic features if needed
from sklearn.base import BaseEstimator, TransformerMixin

# Import base classes and helpers from the project structure
from utils.common import NLIModel
from utils.database import DatabaseHandler
# Import helpers used for baseline models
from .baseline_base import clean_dataset, _evaluate_model_performance # Use existing helpers
# Use filtering logic from SVM baseline to identify syntactic features

logger = logging.getLogger(__name__)

# --- Helper Transformers (copied from Experiment 3/4) ---
class TextSelector(BaseEstimator, TransformerMixin):
    """Selects text columns from a DataFrame."""
    def __init__(self, key):
        self.key = key
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.key]

class FeatureSelector(BaseEstimator, TransformerMixin):
    """Selects precomputed feature columns, handling potential missing ones."""
    def __init__(self, column_names):
        self.column_names = column_names
        self.selected_columns_ = [] # Initialize selected columns list

    def fit(self, X, y=None):
        # Identify available columns during fit
        self.selected_columns_ = [col for col in self.column_names if col in X.columns]
        missing = set(self.column_names) - set(self.selected_columns_)
        if missing:
            logger.warning(f"FeatureSelector: Columns {missing} not found during fit. Using only existing: {self.selected_columns_}")
        return self

    def transform(self, X):
        # Use columns identified during fit, add missing ones with 0 during transform if needed
        missing = set(self.selected_columns_) - set(X.columns)
        if missing:
            logger.warning(f"FeatureSelector: Columns {missing} not found during transform. Adding columns with 0.")
            X_copy = X.copy()
            for col in missing:
                X_copy[col] = 0
            # Return only the columns identified during fit, in the correct order
            return X_copy[self.selected_columns_]
        return X[self.selected_columns_]

# --- Experiment 6 Model Class ---
class GradientBoostingTFIDFSyntacticExperiment6(NLIModel):
    """
    Experiment 6: Gradient Boosting combining TF-IDF (from raw text) and
                  pre-computed hand-crafted syntactic features.
    """
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 3, random_state: int = 42,
                 tfidf_max_features: Optional[int] = 10000, tfidf_ngram_range: Tuple[int, int] = (1, 2),
                 scale_syntactic: bool = True): # Option to scale syntactic features
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.tfidf_max_features = tfidf_max_features
        self.tfidf_ngram_range = tfidf_ngram_range
        self.scale_syntactic = scale_syntactic

        # Initialize the Gradient Boosting model
        self.model = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        self.feature_pipeline: Optional[Pipeline] = None
        self.db_handler = DatabaseHandler()
        self.is_trained = False
        self._syntactic_feature_columns: Optional[List[str]] = None # Store names

    def _load_and_prepare_data(self, dataset: str, split: str, suffix: str) -> Optional[Tuple[pd.DataFrame, np.ndarray]]:
        """Loads raw text and precomputed syntactic features, cleans, and aligns them."""
        logger.info(f"Exp6: Loading data for {dataset}/{split} (suffix: {suffix})")

        # 1. Load raw text data (premise, hypothesis, pair_id, label)
        try:
             pairs_table = f"pairs_{suffix}"
             sentences_table = f"sentences_{suffix}"
             pairs_df = self.db_handler.load_dataframe(dataset, split, pairs_table)
             sentences_df = self.db_handler.load_dataframe(dataset, split, sentences_table)
             if pairs_df.empty or sentences_df.empty: raise ValueError("Intermediate data missing")

             sentences_premise = sentences_df[['id', 'text']].rename(columns={'text': 'premise_text', 'id': 'p_id'})
             sentences_hypothesis = sentences_df[['id', 'text']].rename(columns={'text': 'hypothesis_text', 'id': 'h_id'})
             pairs_essential = pairs_df[['id', 'premise_id', 'hypothesis_id', 'label']].rename(columns={'id': 'pair_id'})

             text_df = pd.merge(pairs_essential, sentences_premise, left_on='premise_id', right_on='p_id', how='left').drop('p_id', axis=1, errors='ignore')
             text_df = pd.merge(text_df, sentences_hypothesis, left_on='hypothesis_id', right_on='h_id', how='left').drop('h_id', axis=1, errors='ignore')

             final_text_cols = ['pair_id', 'premise_text', 'hypothesis_text', 'label']
             if not all(col in text_df.columns for col in final_text_cols):
                 missing = [col for col in final_text_cols if col not in text_df.columns]
                 raise ValueError(f"Columns missing after text merge: {missing}.")
             text_df = text_df[final_text_cols].fillna('')
             logger.info(f"Exp6: Successfully loaded and merged text data. Shape: {text_df.shape}")

        except Exception as e:
            logger.error(f"Exp6: Error loading/merging intermediate text data: {e}", exc_info=True)
            return None

        # 2. Load precomputed features (need lexical+syntactic file to filter syntactic)
        precomputed_feature_table_name = f"{dataset}_{split}_features_lexical_syntactic_{suffix}"
        try:
            features_df = self.db_handler.load_dataframe(dataset, split, precomputed_feature_table_name)
            features_df = _handle_nan_values(features_df, f"{dataset}/{split}/{suffix}_features")
            if 'pair_id' not in features_df.columns: raise ValueError("'pair_id' missing.")
            logger.info(f"Exp6: Successfully loaded precomputed features. Shape: {features_df.shape}")
        except Exception as e:
            logger.error(f"Exp6: Error loading precomputed features from {precomputed_feature_table_name}: {e}", exc_info=True)
            return None

        # 3. Clean and Align
        clean_text_result = clean_dataset(text_df)
        if not clean_text_result: logger.error("Text data invalid after cleaning."); return None
        text_df_clean, y = clean_text_result

        clean_feat_result = clean_dataset(features_df)
        if not clean_feat_result: logger.error("Feature data invalid after cleaning."); return None
        features_df_clean, _ = clean_feat_result

        # 4. Filter ONLY syntactic features needed
        self._syntactic_feature_columns = filter_syntactic_features(features_df_clean)
        if not self._syntactic_feature_columns:
            logger.error("No syntactic feature columns identified in precomputed features.")
            return None
        logger.info(f"Exp6: Keeping {len(self._syntactic_feature_columns)} syntactic features.")
        syntactic_features_to_merge = features_df_clean[['pair_id'] + self._syntactic_feature_columns]

        # 5. Merge text and filtered syntactic features
        text_to_merge = text_df_clean[['pair_id', 'premise_text', 'hypothesis_text']]
        combined_df = pd.merge(text_to_merge, syntactic_features_to_merge, on='pair_id', how='inner')
        num_merged = len(combined_df)
        logger.info(f"Exp6: Merged text and syntactic features: {num_merged} rows.")

        if num_merged == 0: logger.error("No rows remained after merging."); return None

        # Re-align labels 'y'
        final_pair_ids = combined_df['pair_id'].tolist()
        label_map = dict(zip(text_df_clean['pair_id'], y))
        final_y = np.array([label_map.get(pid) for pid in final_pair_ids if pid in label_map])

        if len(final_y) != num_merged: logger.error("Label alignment failed."); return None

        X_df = combined_df # Contains text columns and syntactic feature columns
        return X_df, final_y

    def _build_feature_pipeline(self, sample_df_for_fitting: Optional[pd.DataFrame] = None) -> Pipeline:
        """Builds the feature extraction pipeline."""
        logger.info("Exp6: Building feature pipeline...")

        # TF-IDF for premise and hypothesis
        tfidf_premise = Pipeline([
            ('selector', TextSelector(key='premise_text')),
            ('tfidf', TfidfVectorizer(max_features=self.tfidf_max_features // 2,
                                        ngram_range=self.tfidf_ngram_range,
                                        stop_words='english'))
        ])
        tfidf_hypothesis = Pipeline([
            ('selector', TextSelector(key='hypothesis_text')),
            ('tfidf', TfidfVectorizer(max_features=self.tfidf_max_features // 2,
                                        ngram_range=self.tfidf_ngram_range,
                                        stop_words='english'))
        ])

        # Syntactic features pipeline part
        if self._syntactic_feature_columns is None:
            if sample_df_for_fitting is None:
                raise RuntimeError("Syntactic features needed but no sample data provided.")
            logger.warning("Re-inferring syntactic columns during pipeline build.")
            self._syntactic_feature_columns = filter_syntactic_features(sample_df_for_fitting)

        syntactic_steps = [('selector', FeatureSelector(column_names=self._syntactic_feature_columns))]
        if self.scale_syntactic:
            # StandardScaler works fine with GradientBoosting, handles sparse if TFIDF outputs sparse
            syntactic_steps.append(('scaler', StandardScaler(with_mean=False)))
            logger.info("Adding StandardScaler for syntactic features.")

        syntactic_pipe = Pipeline(syntactic_steps)

        # Combine using FeatureUnion
        combined_features = FeatureUnion([
            ('tfidf_premise', tfidf_premise),
            ('tfidf_hypothesis', tfidf_hypothesis),
            ('syntactic', syntactic_pipe)
        ])

        # Full pipeline (just feature extraction)
        pipeline = Pipeline([('features', combined_features)])
        logger.info("Exp6: Feature pipeline built.")
        return pipeline

    def train(self, train_dataset: str, train_split: str, train_suffix: str,
              val_dataset: Optional[str] = None, val_split: Optional[str] = None, val_suffix: Optional[str] = None) -> Dict[str, Any]:
        """Trains the Gradient Boosting model."""
        logger.info(f"Starting Exp6 training for {train_dataset}/{train_split}/{train_suffix}")
        start_time = time.time()

        train_prep_result = self._load_and_prepare_data(train_dataset, train_split, train_suffix)
        if not train_prep_result: return {'error': 'Training data loading/preparation failed.'}
        X_train_df, y_train = train_prep_result

        self.feature_pipeline = self._build_feature_pipeline(sample_df_for_fitting=X_train_df)
        logger.info("Fitting feature pipeline on training data...")
        try:
            X_train_transformed = self.feature_pipeline.fit_transform(X_train_df, y_train)
            # Gradient Boosting usually prefers dense arrays
            if isinstance(X_train_transformed, csr_matrix):
                 X_train_transformed = X_train_transformed.toarray()
            logger.info(f"Training data transformed. Shape: {X_train_transformed.shape}")
        except Exception as e:
            logger.error(f"Error fitting/transforming training data: {e}", exc_info=True)
            return {'error': 'Feature pipeline fitting failed.'}

        # Check for NaNs before training
        if np.isnan(X_train_transformed).any():
            logger.warning("NaNs detected in transformed training data! Filling with 0.")
            X_train_transformed = np.nan_to_num(X_train_transformed, nan=0.0)


        logger.info("Training Gradient Boosting model...")
        self.model.fit(X_train_transformed, y_train)
        self.is_trained = True
        train_time = time.time() - start_time
        logger.info(f"Model training completed in {train_time:.2f}s.")

        # Validation Evaluation
        eval_metrics = {}
        if val_dataset and val_split and val_suffix:
            logger.info(f"Loading and evaluating on validation data: {val_dataset}/{val_split}/{val_suffix}")
            val_prep_result = self._load_and_prepare_data(val_dataset, val_split, val_suffix)
            if val_prep_result:
                X_val_df, y_val = val_prep_result
                try:
                    logger.info("Transforming validation data...")
                    X_val_transformed = self.feature_pipeline.transform(X_val_df)
                    if isinstance(X_val_transformed, csr_matrix):
                         X_val_transformed = X_val_transformed.toarray()
                    # Check/fill NaNs
                    if np.isnan(X_val_transformed).any():
                        logger.warning("NaNs detected in transformed validation data! Filling with 0.")
                        X_val_transformed = np.nan_to_num(X_val_transformed, nan=0.0)

                    logger.info("Evaluating model on validation data...")
                    eval_time, metrics = _evaluate_model_performance(self, X_val_transformed, y_val)
                    eval_metrics = metrics
                    eval_metrics['eval_time'] = eval_time
                except Exception as e:
                    logger.error(f"Error during validation: {e}", exc_info=True)
                    eval_metrics = {'error': 'Validation evaluation failed.'}
            else: logger.warning("Failed to load/prepare validation data.")
        else: logger.info("No validation data provided.")

        results = {'train_time': train_time, **eval_metrics}
        return results

    def extract_features(self, data: pd.DataFrame) -> np.ndarray:
         """Transforms data using the fitted feature pipeline."""
         if not self.feature_pipeline or not self.is_trained:
             raise RuntimeError("Feature pipeline is not fitted.")
         if not isinstance(data, pd.DataFrame):
             raise TypeError("Input must be a pandas DataFrame.")
         logger.debug("Exp6: Extracting features using fitted pipeline...")
         X_transformed = self.feature_pipeline.transform(data)
         # Convert to dense for Gradient Boosting
         if isinstance(X_transformed, csr_matrix):
             X_transformed = X_transformed.toarray()
         # Handle potential NaNs after transformation
         if np.isnan(X_transformed).any():
             logger.warning("NaNs detected after transformation in extract_features! Filling with 0.")
             X_transformed = np.nan_to_num(X_transformed, nan=0.0)
         return X_transformed

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts labels for pre-transformed feature data."""
        if not self.is_trained: raise RuntimeError("Model not trained.")
        if isinstance(X, csr_matrix): # Ensure dense input
            logger.warning("Received sparse matrix for prediction, converting to dense for Gradient Boosting.")
            X = X.toarray()
        if np.isnan(X).any():
            logger.warning("NaNs detected in prediction data! Filling with 0 before predicting.")
            X = np.nan_to_num(X, nan=0.0)
        return self.model.predict(X)

    # Overload predict to handle DataFrame input
    def predict_on_dataframe(self, data_df: pd.DataFrame) -> np.ndarray:
         if not isinstance(data_df, pd.DataFrame): raise ValueError("Input must be a pandas DataFrame")
         X_transformed = self.extract_features(data_df)
         return self.predict(X_transformed)

    def save(self, filepath: str, model_name) -> None:
        """Saves the model, pipeline, and metadata."""
        if not self.is_trained or not self.feature_pipeline:
            logger.warning("Untrained model/pipeline, skipping save.")
            return
        model_path = f"{filepath}_model.joblib"
        pipeline_path = f"{filepath}_pipeline.joblib"
        metadata_path = f"{filepath}_metadata.joblib"
        logger.info(f"Saving Exp6 artifacts to base: {filepath}")
        metadata = {
             'n_estimators': self.n_estimators,
             'learning_rate': self.learning_rate,
             'max_depth': self.max_depth,
             'random_state': self.random_state,
             'tfidf_max_features': self.tfidf_max_features,
             'tfidf_ngram_range': self.tfidf_ngram_range,
             'scale_syntactic': self.scale_syntactic,
             '_syntactic_feature_columns': self._syntactic_feature_columns
        }
        try:
            joblib.dump(self.model, model_path)
            joblib.dump(self.feature_pipeline, pipeline_path)
            joblib.dump(metadata, metadata_path)
        except Exception as e: logger.error(f"Error saving Exp6 artifacts: {e}", exc_info=True)

    @classmethod
    def load(cls, filepath: str, feature_extractor: Optional[Any] = None) -> 'GradientBoostingTFIDFSyntacticExperiment6':
        """Loads the model, pipeline, and metadata."""
        model_path = f"{filepath}_model.joblib"
        pipeline_path = f"{filepath}_pipeline.joblib"
        metadata_path = f"{filepath}_metadata.joblib"
        logger.info(f"Loading Exp6 artifacts from base: {filepath}")
        if not all(os.path.exists(p) for p in [model_path, pipeline_path, metadata_path]):
            raise FileNotFoundError(f"Artifacts missing for base path: {filepath}")
        try:
            loaded_model = joblib.load(model_path)
            loaded_pipeline = joblib.load(pipeline_path)
            loaded_metadata = joblib.load(metadata_path)
        except Exception as e: logger.error(f"Error loading Exp6 artifacts: {e}", exc_info=True); raise

        instance = cls(
            n_estimators=loaded_metadata.get('n_estimators', 100),
            learning_rate=loaded_metadata.get('learning_rate', 0.1),
            max_depth=loaded_metadata.get('max_depth', 3),
            random_state=loaded_metadata.get('random_state', 42),
            tfidf_max_features=loaded_metadata.get('tfidf_max_features', 10000),
            tfidf_ngram_range=loaded_metadata.get('tfidf_ngram_range', (1, 2)),
            scale_syntactic=loaded_metadata.get('scale_syntactic', True)
        )
        instance.model = loaded_model
        instance.feature_pipeline = loaded_pipeline
        instance._syntactic_feature_columns = loaded_metadata.get('_syntactic_feature_columns')
        instance.is_trained = True
        if instance._syntactic_feature_columns is None:
             logger.warning("Loaded Exp6 metadata missing syntactic feature columns.")
        logger.info("Exp6 Model loaded successfully.")
        return instance