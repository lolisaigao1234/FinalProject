# models/random_forest_bow_syntactic_experiment_5.py
import logging
import os
import joblib
import numpy as np
import pandas as pd
import time
from typing import List, Optional, Any, Dict, Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline, FeatureUnion # Keep FeatureUnion
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix, hstack # Keep hstack

# --- Base Classes and Helpers ---
from utils.common import NLIModel
from utils.database import DatabaseHandler
from .baseline_base import (
    clean_dataset,
    _handle_nan_values,
    filter_syntactic_features,
    SimpleParquetLoader,
    _evaluate_model_performance # Use this helper directly
)
# Import helper transformers
from .multinomial_naive_bayes_bow_syntactic_experiment_4 import TextSelector, FeatureSelector, SparseScaler

logger = logging.getLogger(__name__)

# --- Random Forest Experiment 5 Class (Re-refactored for Trainer compatibility) ---
class RandomForestBowSyntacticExperiment5(NLIModel):
    MODEL_NAME = "RandomForest_BoW_Syntactic_Exp5_PipelineCompatible" # New name

    def __init__(self,
                 args=None,
                 bow_max_features: Optional[int] = 5000,
                 bow_ngram_range: tuple = (1, 1),
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 random_state: int = 42,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 scale_syntactic: bool = False,
                 ):

        # --- Store Hyperparameters ---
        self.bow_max_features = getattr(args, 'bow_max_features', bow_max_features) if args else bow_max_features
        self.bow_ngram_range = getattr(args, 'bow_ngram_range', bow_ngram_range) if args else bow_ngram_range
        self.n_estimators = getattr(args, 'n_estimators', n_estimators) if args else n_estimators
        self.max_depth = getattr(args, 'max_depth', max_depth) if args else max_depth
        self.random_state = getattr(args, 'random_state', random_state) if args else random_state
        self.min_samples_split = getattr(args, 'min_samples_split', min_samples_split) if args else min_samples_split
        self.min_samples_leaf = getattr(args, 'min_samples_leaf', min_samples_leaf) if args else min_samples_leaf
        self.scale_syntactic = getattr(args, 'scale_syntactic', scale_syntactic) if args else scale_syntactic

        self._init_params = { # For saving/loading
            'bow_max_features': self.bow_max_features,
            'bow_ngram_range': self.bow_ngram_range,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'random_state': self.random_state,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'scale_syntactic': self.scale_syntactic,
        }

        # --- Initialize Model Components (will be fitted later) ---
        self.feature_pipeline: Optional[FeatureUnion] = None # Holds the *fitted* feature extractor part
        self.classifier: Optional[RandomForestClassifier] = None # Holds the *fitted* classifier
        self.db_handler = DatabaseHandler()
        self.is_trained = False
        self._syntactic_feature_columns: Optional[List[str]] = None

        logger.info(f"Initialized {self.MODEL_NAME} with params: {self._init_params}")

    def _build_feature_pipeline_unfitted(self, sample_df_for_fitting: pd.DataFrame) -> FeatureUnion:
        """Builds the UNFITTED FeatureUnion pipeline for feature extraction."""
        logger.debug(f"{self.MODEL_NAME}: Building unfitted feature pipeline...")
        feature_union_list = []

        # 1. BoW pipeline part
        max_feat_per_part = (self.bow_max_features // 2) if self.bow_max_features else None
        if max_feat_per_part is None or max_feat_per_part > 0:
            bow_premise_pipe = Pipeline([
                ('selector', TextSelector(key='premise_text')),
                ('bow', CountVectorizer(max_features=max_feat_per_part, ngram_range=self.bow_ngram_range, stop_words='english'))
            ])
            bow_hypothesis_pipe = Pipeline([
                ('selector', TextSelector(key='hypothesis_text')),
                ('bow', CountVectorizer(max_features=max_feat_per_part, ngram_range=self.bow_ngram_range, stop_words='english'))
            ])
            feature_union_list.append(('bow_premise', bow_premise_pipe))
            feature_union_list.append(('bow_hypothesis', bow_hypothesis_pipe))
        else: logger.warning("Excluding BoW features (max_features=0).")

        # 2. Syntactic features pipeline part
        # Infer columns from sample data (crucial step)
        self._syntactic_feature_columns = filter_syntactic_features(sample_df_for_fitting)
        syntactic_cols_to_use = self._syntactic_feature_columns if self._syntactic_feature_columns else []

        if syntactic_cols_to_use:
            syntactic_steps = [('selector', FeatureSelector(column_names=syntactic_cols_to_use))]
            if self.scale_syntactic:
                syntactic_steps.append(('scaler', SparseScaler(scaler=StandardScaler(with_mean=False))))
            syntactic_pipe = Pipeline(syntactic_steps)
            feature_union_list.append(('syntactic', syntactic_pipe))
        else: logger.warning("No syntactic features found/used.")

        if not feature_union_list:
             raise RuntimeError(f"{self.MODEL_NAME}: Feature pipeline cannot be built - no features configured.")

        return FeatureUnion(feature_union_list)

    def _load_and_prepare_data(self, dataset: str, split: str, suffix: str) -> Optional[Tuple[pd.DataFrame, np.ndarray]]:
        """Loads and prepares data (text + features) required for feature extraction."""
        # Reuse the implementation from the previous version
        # (Loads text from DB, loads precomputed features from Parquet, merges, cleans)
        logger.info(f"{self.MODEL_NAME}: Loading data for {dataset}/{split} (suffix: {suffix})")
        text_df = None
        try: # Load text
            pairs_table = f"pairs_{suffix}"
            sentences_table = f"sentences_{suffix}"
            pairs_df = self.db_handler.load_dataframe(dataset, split, pairs_table)
            sentences_df = self.db_handler.load_dataframe(dataset, split, sentences_table)
            if pairs_df is None or sentences_df is None or pairs_df.empty or sentences_df.empty: raise ValueError(f"Intermediate text missing.")
            if not all(c in sentences_df.columns for c in ['id', 'text']): raise ValueError("Sentences columns missing.")
            if not all(c in pairs_df.columns for c in ['id', 'premise_id', 'hypothesis_id', 'label']): raise ValueError("Pairs columns missing.")
            # Merge logic (same as before)
            sentences_premise = sentences_df[['id', 'text']].rename(columns={'text': 'premise_text', 'id': 'p_id'}).astype(str)
            sentences_hypothesis = sentences_df[['id', 'text']].rename(columns={'text': 'hypothesis_text', 'id': 'h_id'}).astype(str)
            pairs_essential = pairs_df[['id', 'premise_id', 'hypothesis_id', 'label']].rename(columns={'id': 'pair_id'})
            pairs_essential[['premise_id', 'hypothesis_id']] = pairs_essential[['premise_id', 'hypothesis_id']].astype(str)
            text_df = pd.merge(pairs_essential, sentences_premise, left_on='premise_id', right_on='p_id', how='left').drop('p_id', axis=1, errors='ignore')
            text_df = pd.merge(text_df, sentences_hypothesis, left_on='hypothesis_id', right_on='h_id', how='left').drop('h_id', axis=1, errors='ignore')
            final_text_cols = ['pair_id', 'premise_text', 'hypothesis_text', 'label']
            if not all(col in text_df.columns for col in final_text_cols): raise ValueError("Text merge failed.")
            text_df = text_df[final_text_cols].fillna({'premise_text': '', 'hypothesis_text': ''})
        except Exception as e: logger.error(f"Error loading text data: {e}", exc_info=True); return None

        features_df = None
        try: # Load precomputed features
            loader = SimpleParquetLoader()
            features_df = loader.load_data(self, dataset, split, suffix) # Pass model instance 'self'
            if features_df is None or features_df.empty: raise FileNotFoundError("Features loader returned empty.")
            features_df = _handle_nan_values(features_df, f"{dataset}/{split}/{suffix}_features")
            if 'pair_id' not in features_df.columns: raise ValueError("'pair_id' missing in features file.")
        except Exception as e: logger.error(f"Error loading features data: {e}", exc_info=True); return None

        # Clean text data first (gets valid labels)
        clean_text_result = clean_dataset(text_df)
        if not clean_text_result: logger.error("Text data invalid after cleaning."); return None
        text_df_clean, y_labels = clean_text_result

        # Filter syntactic features (use column names determined during build/load, or filter now)
        if self._syntactic_feature_columns is None:
             self._syntactic_feature_columns = filter_syntactic_features(features_df)
        syntactic_cols_to_keep = self._syntactic_feature_columns if self._syntactic_feature_columns else []
        features_df['pair_id'] = features_df['pair_id'].astype(str)
        syntactic_features_to_merge = features_df[['pair_id'] + syntactic_cols_to_keep].copy()

        # Merge cleaned text and filtered syntactic features
        text_to_merge = text_df_clean[['pair_id', 'premise_text', 'hypothesis_text']].copy()
        text_to_merge['pair_id'] = text_to_merge['pair_id'].astype(str)
        combined_df = pd.merge(text_to_merge, syntactic_features_to_merge, on='pair_id', how='inner')
        if combined_df.empty: logger.error("No rows after merging text and features."); return None

        # Re-align labels
        label_map = dict(zip(text_df_clean['pair_id'].astype(str), y_labels))
        final_y = np.array([label_map.get(pid) for pid in combined_df['pair_id']]) # Use .get for safety
        valid_label_mask = final_y != None # Ensure alignment worked
        if not np.all(valid_label_mask): logger.error("Label alignment failed after merge."); return None
        final_y = final_y.astype(int) # Convert to int after check

        if len(final_y) != len(combined_df): logger.error("Label count mismatch after merge."); return None

        logger.info(f"{self.MODEL_NAME}: Data loaded/prepared for {split}. Shape: {combined_df.shape}")
        return combined_df, final_y # Return combined DF and aligned labels

    # --- NLIModel Methods ---

    def extract_features(self, data: pd.DataFrame) -> Any:
        """
        Builds and fits the feature pipeline during the first call (training).
        Transforms data using the fitted pipeline on subsequent calls (evaluation).
        """
        if not isinstance(data, pd.DataFrame):
             raise TypeError("extract_features expects a pandas DataFrame.")
        if data.empty: return csr_matrix((0, 0))

        if not self.is_trained: # Training context: fit feature pipeline
            logger.info(f"{self.MODEL_NAME}: Fitting feature pipeline and transforming training data...")
            start_time = time.time()
            unfitted_pipeline = self._build_feature_pipeline_unfitted(sample_df_for_fitting=data)
            try:
                # Fit the feature pipeline AND transform the data
                X_transformed = unfitted_pipeline.fit_transform(data)
                # Store the *fitted* feature pipeline
                self.feature_pipeline = unfitted_pipeline
                fit_time = time.time() - start_time
                logger.info(f"Feature pipeline fit and transform complete in {fit_time:.2f}s. Output shape: {X_transformed.shape}")
                # Also store the final syntactic columns used
                try:
                    fitted_selector = self.feature_pipeline.transformer_list[-1][1].named_steps['selector']
                    self._syntactic_feature_columns = fitted_selector.columns_seen_during_fit_
                    logger.debug(f"Stored {len(self._syntactic_feature_columns)} syntactic cols from fitted pipeline.")
                except Exception as e: logger.warning(f"Could not get syntactic cols from feature pipeline: {e}")

                return X_transformed
            except Exception as e:
                logger.error(f"Error during feature pipeline fit_transform: {e}", exc_info=True)
                raise
        else: # Evaluation context: use stored fitted feature pipeline
            logger.debug(f"{self.MODEL_NAME}: Transforming data using stored feature pipeline...")
            if self.feature_pipeline is None:
                raise RuntimeError(f"{self.MODEL_NAME} is marked trained, but feature_pipeline is missing.")
            try:
                X_transformed = self.feature_pipeline.transform(data)
                logger.debug(f"Feature transformation complete. Output shape: {X_transformed.shape}")
                return X_transformed
            except Exception as e:
                logger.error(f"Error during feature pipeline transform: {e}", exc_info=True)
                # Log expected columns vs available columns
                expected_cols = getattr(self.feature_pipeline.transformer_list[-1][1].named_steps.get('selector'), 'columns_seen_during_fit_', [])
                logger.error(f"Feature pipeline expected syntactic cols: {expected_cols}")
                logger.error(f"Input data columns: {data.columns.tolist()}")
                raise

    def train(self, X: Any, y: np.ndarray) -> None:
        """
        Fits the RandomForestClassifier on the *already transformed* features X.
        """
        if X is None or y is None:
            logger.error("Training requires non-null features (X) and labels (y).")
            return
        # Basic check on X type (should be sparse matrix from extract_features)
        if not isinstance(X, csr_matrix):
             logger.warning(f"Input X to train is type {type(X)}, expected sparse matrix. Classifier might handle it.")
        if X.shape[0] != len(y):
            raise ValueError(f"Feature/Label mismatch in train: X={X.shape}, y={len(y)}")
        if X.shape[0] == 0:
            logger.warning("Cannot train classifier with 0 samples.")
            return

        logger.info(f"Training {self.MODEL_NAME} classifier on {X.shape[0]} samples, {X.shape[1]} features...")
        start_time = time.time()
        # Initialize the classifier (ensure hyperparameters are from __init__)
        self.classifier = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1
        )
        try:
            self.classifier.fit(X, y)
            self.is_trained = True # Mark as trained only AFTER successful fit
            train_time = time.time() - start_time
            logger.info(f"Classifier training complete in {train_time:.2f}s.")
        except Exception as e:
            logger.error(f"Error during classifier fitting: {e}", exc_info=True)
            self.is_trained = False # Ensure flag is false on error
            raise

    def predict(self, X: Any) -> np.ndarray:
        """
        Makes predictions. Handles DataFrame input by transforming first,
        or predicts directly on pre-transformed features.
        """
        if not self.is_trained or self.classifier is None:
            raise RuntimeError(f"{self.MODEL_NAME} has not been trained or classifier is missing.")

        X_transformed = None
        if isinstance(X, pd.DataFrame):
            # Transform DataFrame using the feature pipeline
            logger.debug(f"{self.MODEL_NAME} predict received DataFrame, transforming features...")
            try:
                 X_transformed = self.extract_features(X) # Use the extract_features method
            except Exception as e:
                 logger.error(f"Error transforming features during predict: {e}", exc_info=True)
                 raise
        else:
            # Assume X is already transformed features
            logger.debug(f"{self.MODEL_NAME} predict received pre-transformed features.")
            X_transformed = X # Use input directly

        if X_transformed is None:
             raise ValueError("Feature transformation for prediction failed.")

        # Check feature dimensions before predicting
        if hasattr(self.classifier, 'n_features_in_') and X_transformed.shape[1] != self.classifier.n_features_in_:
             raise ValueError(f"Prediction feature dimension mismatch: Input={X_transformed.shape[1]}, Expected={self.classifier.n_features_in_}")

        logger.debug(f"Predicting with classifier on shape {X_transformed.shape}...")
        try:
             predictions = self.classifier.predict(X_transformed)
             logger.debug("Prediction complete.")
             return predictions
        except Exception as e:
            logger.error(f"Error during classifier prediction: {e}", exc_info=True)
            raise

    def save(self, directory: str, model_name_base: str) -> None:
        """Saves the fitted feature pipeline, fitted classifier, and metadata."""
        if not self.is_trained or self.feature_pipeline is None or self.classifier is None:
            logger.warning(f"Attempting to save {self.MODEL_NAME} before it's fully trained/fitted.")
            # Decide whether to proceed or return
            # return

        pipeline_path = os.path.join(directory, f"{model_name_base}_feature_pipeline.joblib")
        classifier_path = os.path.join(directory, f"{model_name_base}_classifier.joblib")
        metadata_path = os.path.join(directory, f"{model_name_base}_metadata.joblib")

        metadata = {
            'init_params': self._init_params,
            'is_trained': self.is_trained,
            '_syntactic_feature_columns': self._syntactic_feature_columns
        }

        logger.info(f"Saving {self.MODEL_NAME} feature pipeline to {pipeline_path}")
        logger.info(f"Saving {self.MODEL_NAME} classifier to {classifier_path}")
        logger.info(f"Saving {self.MODEL_NAME} metadata to {metadata_path}")
        try:
            os.makedirs(directory, exist_ok=True)
            if self.feature_pipeline: joblib.dump(self.feature_pipeline, pipeline_path)
            if self.classifier: joblib.dump(self.classifier, classifier_path)
            joblib.dump(metadata, metadata_path)
            logger.info(f"{self.MODEL_NAME} components saved successfully.")
        except Exception as e:
            logger.error(f"Error saving {self.MODEL_NAME} artifacts: {e}", exc_info=True)
            # Clean up partial files
            if os.path.exists(pipeline_path): os.remove(pipeline_path)
            if os.path.exists(classifier_path): os.remove(classifier_path)
            if os.path.exists(metadata_path): os.remove(metadata_path)
            raise

    @classmethod
    def load(cls, directory: str, model_name_base: str) -> 'RandomForestBowSyntacticExperiment5':
        """Loads the feature pipeline, classifier, and metadata."""
        pipeline_path = os.path.join(directory, f"{model_name_base}_feature_pipeline.joblib")
        classifier_path = os.path.join(directory, f"{model_name_base}_classifier.joblib")
        metadata_path = os.path.join(directory, f"{model_name_base}_metadata.joblib")

        logger.info(f"Loading {cls.MODEL_NAME} from {directory}...")
        if not all(os.path.exists(p) for p in [pipeline_path, classifier_path, metadata_path]):
            missing = [p for p in [pipeline_path, classifier_path, metadata_path] if not os.path.exists(p)]
            raise FileNotFoundError(f"Required model files missing: {missing}")

        try:
            loaded_pipeline = joblib.load(pipeline_path)
            loaded_classifier = joblib.load(classifier_path)
            loaded_metadata = joblib.load(metadata_path)
        except Exception as e: logger.error(f"Error loading artifacts: {e}", exc_info=True); raise

        # Instantiate using loaded metadata init_params
        init_params = loaded_metadata.get('init_params', {})
        instance = cls(args=None, **init_params) # Pass loaded params

        instance.feature_pipeline = loaded_pipeline
        instance.classifier = loaded_classifier
        instance.is_trained = loaded_metadata.get('is_trained', False)
        instance._syntactic_feature_columns = loaded_metadata.get('_syntactic_feature_columns')

        # Post-load checks
        if not isinstance(instance.feature_pipeline, FeatureUnion): logger.warning("Loaded feature_pipeline is not FeatureUnion.")
        if not isinstance(instance.classifier, RandomForestClassifier): logger.warning("Loaded classifier is not RandomForestClassifier.")
        if instance.is_trained and instance._syntactic_feature_columns is None: logger.warning("Trained model loaded but syntactic columns missing.")

        logger.info(f"{cls.MODEL_NAME} loaded successfully. Trained: {instance.is_trained}")
        return instance

    def evaluate(self, dataset_name: str, split: str, suffix: str) -> Dict[str, Any]:
        """Evaluates the trained model pipeline on a given dataset split."""
        if not self.is_trained or self.feature_pipeline is None or self.classifier is None:
            logger.error(f"Cannot evaluate: {self.MODEL_NAME} is not trained or components missing.")
            return {'status': 'Evaluation failed: Model not ready.'}

        logger.info(f"--- Starting evaluation for {self.MODEL_NAME} on {dataset_name}/{split}/{suffix} ---")
        metrics = {}
        eval_start_time = time.time()

        # 1. Load and prepare data (expects DataFrame with text and features)
        prepared_data = self._load_and_prepare_data(dataset_name, split, suffix)
        if prepared_data is None:
            return {'status': f'Evaluation failed: Cannot load/prepare data for split {split}.'}
        df_eval, y_true = prepared_data

        if df_eval.empty or y_true is None or len(y_true) == 0:
            return {'status': f'No valid samples for split {split}', 'accuracy': 0.0, 'f1': 0.0}

        # 2. Make predictions (using the predict method which handles transformation)
        try:
            y_pred = self.predict(df_eval)

            # 3. Calculate metrics using the helper from baseline_base
            logger.info("Calculating evaluation metrics...")
            # _evaluate_model_performance needs the model instance (self), X, y
            # We have y_pred, so calculate directly or adapt helper
            # Direct calculation:
            accuracy = accuracy_score(y_true, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
            eval_time = time.time() - eval_start_time

            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'eval_time': eval_time
            }
            logger.info(f"Eval Metrics - Acc: {accuracy:.4f}, P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}")

        except Exception as e:
            logger.error(f"Error during evaluation predict/metrics for {split}: {e}", exc_info=True)
            return {'status': f'Evaluation failed during predict/metrics for split {split}: {e}'}

        logger.info(f"Evaluation complete for {self.MODEL_NAME} on {split}. Total time: {metrics.get('eval_time', 0):.2f}s")
        return metrics