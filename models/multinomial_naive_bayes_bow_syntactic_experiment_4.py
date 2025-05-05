# Create file: IS567FP/models/multinomial_naive_bayes_bow_syntactic_experiment_4.py
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler # Import MinMaxScaler for non-negative scaling
from sklearn.base import BaseEstimator, TransformerMixin

# Import base classes and helpers from the project structure
from utils.common import NLIModel
from utils.database import DatabaseHandler
# Import helpers specifically used for baseline models
from .baseline_base import clean_dataset, _evaluate_model_performance # Use existing helpers

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
    def __init__(self, column_names):
        self.column_names = column_names
        self.selected_columns_ = [] # Initialize here

    def fit(self, X, y=None):
        missing = [col for col in self.column_names if col not in X.columns]
        self.selected_columns_ = [col for col in self.column_names if col in X.columns]
        if missing:
            logger.warning(f"FeatureSelector: Columns {missing} not found during fit. Using only existing: {self.selected_columns_}")
        return self

    def transform(self, X):
        # Ensure only columns present during fit are used
        missing = [col for col in self.selected_columns_ if col not in X.columns]
        if missing:
            logger.warning(f"FeatureSelector: Columns {missing} not found during transform. Adding columns with 0.")
            X_copy = X.copy()
            for col in missing:
                X_copy[col] = 0
            # Use selected_columns_ determined during fit
            return X_copy[self.selected_columns_]
        return X[self.selected_columns_]

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
    def __init__(self, alpha: float = 1.0,
                 bow_max_features: Optional[int] = 10000, bow_ngram_range: Tuple[int, int] = (1, 1),
                 scale_syntactic: bool = True): # Option to scale syntactic features
        self.alpha = alpha
        self.bow_max_features = bow_max_features
        self.bow_ngram_range = bow_ngram_range
        self.scale_syntactic = scale_syntactic
        self.model = MultinomialNB(alpha=self.alpha)
        self.feature_pipeline: Optional[Pipeline] = None
        self.db_handler = DatabaseHandler()
        self.is_trained = False
        self._syntactic_feature_columns: Optional[List[str]] = None

    # Data loading and preparation (similar to Experiment 3)
    def _load_and_prepare_data(self, dataset: str, split: str, suffix: str) -> Optional[Tuple[pd.DataFrame, np.ndarray]]:
        logger.info(f"Exp4: Loading data for {dataset}/{split} (suffix: {suffix})")

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
             logger.info(f"Exp4: Successfully loaded and merged text data. Shape: {text_df.shape}")

        except Exception as e:
            logger.error(f"Exp4: Error loading/merging intermediate text data for {dataset}/{split}/{suffix}: {e}", exc_info=True)
            return None

        # 2. Load precomputed features (lexical+syntactic needed to filter syntactic)
        precomputed_feature_table_name = f"{dataset}_{split}_features_lexical_syntactic_{suffix}"
        try:
            features_df = self.db_handler.load_dataframe(dataset, split, precomputed_feature_table_name)
            features_df = _handle_nan_values(features_df, f"{dataset}/{split}/{suffix}_features")
            if 'pair_id' not in features_df.columns: raise ValueError("'pair_id' missing.")
            logger.info(f"Exp4: Successfully loaded precomputed features. Shape: {features_df.shape}")
        except Exception as e:
            logger.error(f"Exp4: Error loading precomputed features from {precomputed_feature_table_name}: {e}", exc_info=True)
            return None

        # 3. Clean and Align
        clean_text_result = clean_dataset(text_df)
        if not clean_text_result: logger.error("Text data invalid after cleaning."); return None
        text_df_clean, y = clean_text_result

        clean_feat_result = clean_dataset(features_df) # Clean features df too
        if not clean_feat_result: logger.error("Feature data invalid after cleaning."); return None
        features_df_clean, _ = clean_feat_result

        # 4. Filter ONLY syntactic features needed for this experiment
        self._syntactic_feature_columns = filter_syntactic_features(features_df_clean)
        if not self._syntactic_feature_columns:
            logger.error("No syntactic feature columns identified in the precomputed features file.")
            return None
        logger.info(f"Exp4: Keeping {len(self._syntactic_feature_columns)} syntactic features.")
        syntactic_features_to_merge = features_df_clean[['pair_id'] + self._syntactic_feature_columns]

        # 5. Merge text and *filtered* syntactic features
        text_to_merge = text_df_clean[['pair_id', 'premise_text', 'hypothesis_text']]
        combined_df = pd.merge(text_to_merge, syntactic_features_to_merge, on='pair_id', how='inner')
        num_merged = len(combined_df)
        logger.info(f"Exp4: Merged text and syntactic features: {num_merged} rows.")

        if num_merged == 0: logger.error("No rows remained after merging text and syntactic features."); return None

        # Re-align labels 'y'
        final_pair_ids = combined_df['pair_id'].tolist()
        label_map = dict(zip(text_df_clean['pair_id'], y))
        final_y = np.array([label_map.get(pid) for pid in final_pair_ids if pid in label_map]) # Ensure pid exists

        if len(final_y) != num_merged: logger.error("Label alignment failed after merge."); return None

        X_df = combined_df
        return X_df, final_y

    # Feature pipeline building
    def _build_feature_pipeline(self, sample_df_for_fitting: Optional[pd.DataFrame] = None) -> Pipeline:
        logger.info("Exp4: Building feature pipeline...")

        # 1. Bag-of-Words (BoW) pipeline for combined premise + hypothesis
        # MNB handles text better if combined sometimes
        # Option A: Combine text before vectorizer (simpler pipeline)
        # class TextCombiner(BaseEstimator, TransformerMixin):
        #     def fit(self, X, y=None): return self
        #     def transform(self, X): return X['premise_text'] + " " + X['hypothesis_text']
        # text_bow_pipe = Pipeline([
        #     ('combiner', TextCombiner()),
        #     ('bow', CountVectorizer(max_features=self.bow_max_features,
        #                             ngram_range=self.bow_ngram_range,
        #                             stop_words='english',
        #                             binary=False)) # Typically False for MNB
        # ])

        # Option B: Separate vectorizers and combine later (more complex but flexible)
        bow_premise_pipe = Pipeline([
            ('selector', TextSelector(key='premise_text')),
            ('bow', CountVectorizer(max_features=self.bow_max_features // 2,
                                    ngram_range=self.bow_ngram_range, stop_words='english', binary=False))
        ])
        bow_hypothesis_pipe = Pipeline([
            ('selector', TextSelector(key='hypothesis_text')),
            ('bow', CountVectorizer(max_features=self.bow_max_features // 2,
                                    ngram_range=self.bow_ngram_range, stop_words='english', binary=False))
        ])


        # 2. Syntactic features pipeline
        # Ensure syntactic columns are known (should be set during _load_and_prepare_data)
        if self._syntactic_feature_columns is None:
            if sample_df_for_fitting is None:
                 raise RuntimeError("Syntactic features missing and no sample data provided to infer them.")
            logger.warning("Re-inferring syntactic columns during pipeline build.")
            self._syntactic_feature_columns = filter_syntactic_features(sample_df_for_fitting)

        syntactic_steps = [('selector', FeatureSelector(column_names=self._syntactic_feature_columns))]
        if self.scale_syntactic:
             # Use MinMaxScaler to ensure non-negativity for MNB
             syntactic_steps.append(('scaler', SparseScaler(scaler=MinMaxScaler())))
             logger.info("Adding MinMaxScaler for syntactic features.")
        else:
             # MNB needs non-negative features. If raw syntactic features can be negative,
             # scaling or binning is essential. We assume for now they are non-negative or
             # scale_syntactic=True is used. If not scaling, need to ensure non-negativity.
             # We could add a check or a transformer to clip negative values to 0.
             logger.warning("Not scaling syntactic features. Ensure they are non-negative for MNB.")
             # Example clipping (add this step if needed and not scaling):
             # class ClipNegative(BaseEstimator, TransformerMixin):
             #     def fit(self, X, y=None): return self
             #     def transform(self, X):
             #         if isinstance(X, csr_matrix): X.data[X.data < 0] = 0
             #         else: X[X < 0] = 0
             #         return X
             # syntactic_steps.append(('clipper', ClipNegative()))


        syntactic_pipe = Pipeline(syntactic_steps)

        # 3. Combine using FeatureUnion
        # Note: TF-IDF/BoW outputs sparse, Scaler might output dense depending on input/scaler.
        # FeatureUnion handles mixed sparse/dense inputs. MNB works with sparse.
        combined_features = FeatureUnion([
            ('bow_premise', bow_premise_pipe),
            ('bow_hypothesis', bow_hypothesis_pipe),
            ('syntactic', syntactic_pipe)
            # Add transformer_weights if needed, e.g., {'bow_...': 1.0, 'syntactic': 0.5}
        ])

        # 4. Full Pipeline (only feature extraction)
        pipeline = Pipeline([('features', combined_features)])
        logger.info("Exp4: Feature pipeline built.")
        return pipeline

    # Training method
    def train(self, train_dataset: str, train_split: str, train_suffix: str,
              val_dataset: Optional[str] = None, val_split: Optional[str] = None, val_suffix: Optional[str] = None) -> Dict[str, Any]:
        logger.info(f"Starting Exp4 training for {train_dataset}/{train_split}/{train_suffix}")
        start_time = time.time()

        train_prep_result = self._load_and_prepare_data(train_dataset, train_split, train_suffix)
        if not train_prep_result: return {'error': 'Training data loading/preparation failed.'}
        X_train_df, y_train = train_prep_result

        self.feature_pipeline = self._build_feature_pipeline(sample_df_for_fitting=X_train_df)
        logger.info("Fitting feature pipeline on training data...")
        try:
            # MNB requires non-negative features. TF-IDF/CountVectorizer produce non-negative.
            # Ensure syntactic features are handled appropriately (scaled to non-neg or already non-neg).
            X_train_transformed = self.feature_pipeline.fit_transform(X_train_df, y_train)
            # Check for negative values if not scaling syntactic features
            if not self.scale_syntactic and X_train_transformed.min() < 0:
                 logger.warning("Negative values detected in features passed to MNB after transformation! This might cause errors. Consider scaling syntactic features.")
            logger.info(f"Training data transformed. Shape: {X_train_transformed.shape}")
        except Exception as e:
            logger.error(f"Error fitting/transforming training data with pipeline: {e}", exc_info=True)
            return {'error': 'Feature pipeline fitting failed.'}

        logger.info("Training Multinomial Naive Bayes model...")
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
                    logger.info("Evaluating model on validation data...")
                    eval_time, metrics = _evaluate_model_performance(self, X_val_transformed, y_val)
                    eval_metrics = metrics
                    eval_metrics['eval_time'] = eval_time
                except Exception as e:
                    logger.error(f"Error during validation evaluation: {e}", exc_info=True)
                    eval_metrics = {'error': 'Validation evaluation failed.'}
            else:
                logger.warning("Failed to load/prepare validation data. Skipping validation.")
        else:
            logger.info("No validation data provided.")

        results = {'train_time': train_time, **eval_metrics}
        return results

    # NLIModel abstract method implementations
    def extract_features(self, data: pd.DataFrame) -> Any:
        if not self.feature_pipeline or not self.is_trained:
            raise RuntimeError("Feature pipeline is not fitted. Train the model first.")
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame.")
        return self.feature_pipeline.transform(data)

    def predict(self, X: Any) -> np.ndarray:
        if not self.is_trained: raise RuntimeError("Model has not been trained yet.")
        return self.model.predict(X)

    # Overload predict to handle DataFrame input
    def predict_on_dataframe(self, data_df: pd.DataFrame) -> np.ndarray:
         if not isinstance(data_df, pd.DataFrame): raise ValueError("Input must be a pandas DataFrame")
         X_transformed = self.extract_features(data_df)
         return self.predict(X_transformed)

    def save(self, filepath: str, model_name) -> None:
        if not self.is_trained or not self.feature_pipeline:
            logger.warning("Attempting to save an untrained model or unfitted pipeline.")
            return
        model_path = f"{filepath}_model.joblib"
        pipeline_path = f"{filepath}_pipeline.joblib"
        metadata_path = f"{filepath}_metadata.joblib"
        logger.info(f"Saving Exp4 model to {model_path}, pipeline to {pipeline_path}, metadata to {metadata_path}")
        metadata = {
             'alpha': self.alpha,
             'bow_max_features': self.bow_max_features,
             'bow_ngram_range': self.bow_ngram_range,
             'scale_syntactic': self.scale_syntactic,
             '_syntactic_feature_columns': self._syntactic_feature_columns
        }
        try:
            joblib.dump(self.model, model_path)
            joblib.dump(self.feature_pipeline, pipeline_path)
            joblib.dump(metadata, metadata_path)
        except Exception as e: logger.error(f"Error saving Exp4 artifacts: {e}", exc_info=True)

    @classmethod
    def load(cls, filepath: str, feature_extractor: Optional[Any] = None) -> 'MultinomialNaiveBayesBowSyntacticExperiment4':
        model_path = f"{filepath}_model.joblib"
        pipeline_path = f"{filepath}_pipeline.joblib"
        metadata_path = f"{filepath}_metadata.joblib"
        logger.info(f"Loading Exp4 model from {model_path}, pipeline from {pipeline_path}, metadata from {metadata_path}")
        if not all(os.path.exists(p) for p in [model_path, pipeline_path, metadata_path]):
            raise FileNotFoundError(f"Model artifacts missing for base path: {filepath}")
        try:
            loaded_model = joblib.load(model_path)
            loaded_pipeline = joblib.load(pipeline_path)
            loaded_metadata = joblib.load(metadata_path)
        except Exception as e: logger.error(f"Error loading Exp4 artifacts: {e}", exc_info=True); raise

        instance = cls(
            alpha=loaded_metadata.get('alpha', 1.0),
            bow_max_features=loaded_metadata.get('bow_max_features', 10000),
            bow_ngram_range=loaded_metadata.get('bow_ngram_range', (1, 1)),
            scale_syntactic=loaded_metadata.get('scale_syntactic', True)
        )
        instance.model = loaded_model
        instance.feature_pipeline = loaded_pipeline
        instance._syntactic_feature_columns = loaded_metadata.get('_syntactic_feature_columns')
        instance.is_trained = True
        if instance._syntactic_feature_columns is None:
             logger.warning("Loaded Exp4 model metadata missing syntactic feature columns.")
        logger.info("Exp4 Model loaded successfully.")
        return instance

