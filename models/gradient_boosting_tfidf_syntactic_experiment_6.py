# models/gradient_boosting_tfidf_syntactic_experiment_6.py
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
from .baseline_base import clean_dataset, _evaluate_model_performance, _handle_nan_values, filter_syntactic_features, SimpleParquetLoader # Use existing helpers

logger = logging.getLogger(__name__)

# --- Helper Transformers (copied from Experiment 3/4) ---
class TextSelector(BaseEstimator, TransformerMixin):
    """Selects text columns from a DataFrame."""
    def __init__(self, key):
        self.key = key
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        # Handle potential KeyError if column is missing during transform
        return X.get(self.key, pd.Series([''] * len(X))) # Return empty strings if key missing


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Selects precomputed feature columns specified during initialization.
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
            logger.warning(f"FeatureSelector fit: Potential columns {missing_at_fit} not found. Using only found columns: {self.columns_seen_during_fit_}")
        if not self.columns_seen_during_fit_:
            logger.error("FeatureSelector fit: No specified columns found in the training data!")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.columns_seen_during_fit_ is None:
            raise RuntimeError("FeatureSelector must be fitted before transforming.")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame for transform.")
        cols_to_use = self.columns_seen_during_fit_
        # Use reindex for robust handling: adds missing, removes extra, orders correctly
        X_processed = X.reindex(columns=cols_to_use, fill_value=0)
        missing_for_transform = set(cols_to_use) - set(X.columns)
        if missing_for_transform:
             logger.warning(f"FeatureSelector transform: Columns {missing_for_transform} (seen during fit) were missing and added with value 0.")
        return X_processed


class SparseScaler(BaseEstimator, TransformerMixin):
    """Applies a scaler (like StandardScaler) compatibly with sparse matrices."""
    def __init__(self, scaler=StandardScaler(with_mean=False)): # Default for RF/GBM
        self.scaler = scaler

    def fit(self, X, y=None):
        # Only fit the scaler
        self.scaler.fit(X)
        return self

    def transform(self, X):
        # Transform and return
        return self.scaler.transform(X)

# --- Experiment 6 Model Class ---
class GradientBoostingTFIDFSyntacticExperiment6(NLIModel):
    """
    Experiment 6: Gradient Boosting combining TF-IDF (from raw text) and
                  pre-computed hand-crafted syntactic features.
    """
    MODEL_NAME = "GradientBoosting_TFIDF_Syntactic_Exp6"

    def __init__(self, args=None, **kwargs): # Modified to accept args
        # Extract hyperparameters from args or use defaults
        self.n_estimators = getattr(args, 'n_estimators', 100) if args else 100
        self.learning_rate = getattr(args, 'learning_rate', 0.1) if args else 0.1
        self.max_depth = getattr(args, 'max_depth', 3) if args else 3
        self.random_state = getattr(args, 'random_state', 42) if args else 42
        self.tfidf_max_features = getattr(args, 'tfidf_max_features', 10000) if args else 10000
        self.tfidf_ngram_range = getattr(args, 'tfidf_ngram_range', (1, 2)) if args else (1, 2)
        self.scale_syntactic = getattr(args, 'scale_syntactic', True) if args else True

        # Store init params for saving/loading
        self._init_params = {
             'n_estimators': self.n_estimators,
             'learning_rate': self.learning_rate,
             'max_depth': self.max_depth,
             'random_state': self.random_state,
             'tfidf_max_features': self.tfidf_max_features,
             'tfidf_ngram_range': self.tfidf_ngram_range,
             'scale_syntactic': self.scale_syntactic
        }

        # Initialize the Gradient Boosting model (will be fitted in train)
        self.model: Optional[GradientBoostingClassifier] = None
        self.feature_pipeline: Optional[Pipeline] = None # Holds fitted feature pipeline
        self.db_handler = DatabaseHandler()
        self.is_trained = False
        self._syntactic_feature_columns: Optional[List[str]] = None # Store names used

        logger.info(f"Initialized {self.MODEL_NAME} with params: {self._init_params}")


    def _load_and_prepare_data(self, dataset: str, split: str, suffix: str) -> Optional[Tuple[pd.DataFrame, np.ndarray]]:
        """Loads raw text and precomputed features, cleans, merges, and aligns them."""
        logger.info(f"{self.MODEL_NAME}: Loading data for {dataset}/{split} (suffix: {suffix})")

        # 1. Load raw text data (premise, hypothesis, pair_id, label) - Robust loading
        text_df = None
        try:
             pairs_table = f"pairs_{suffix}"
             sentences_table = f"sentences_{suffix}"
             pairs_df = self.db_handler.load_dataframe(dataset, split, pairs_table)
             sentences_df = self.db_handler.load_dataframe(dataset, split, sentences_table)
             if pairs_df is None or sentences_df is None or pairs_df.empty or sentences_df.empty:
                  raise ValueError(f"Intermediate data missing or empty: {pairs_table} or {sentences_table}")
             # Ensure required columns exist
             if not all(c in sentences_df.columns for c in ['id', 'text']): raise ValueError("Sentences table missing columns.")
             if not all(c in pairs_df.columns for c in ['id', 'premise_id', 'hypothesis_id', 'label']): raise ValueError("Pairs table missing columns.")

             sentences_premise = sentences_df[['id', 'text']].rename(columns={'text': 'premise_text', 'id': 'p_id'}).astype(str)
             sentences_hypothesis = sentences_df[['id', 'text']].rename(columns={'text': 'hypothesis_text', 'id': 'h_id'}).astype(str)
             pairs_essential = pairs_df[['id', 'premise_id', 'hypothesis_id', 'label']].rename(columns={'id': 'pair_id'})
             pairs_essential[['premise_id', 'hypothesis_id']] = pairs_essential[['premise_id', 'hypothesis_id']].astype(str)

             text_df = pd.merge(pairs_essential, sentences_premise, left_on='premise_id', right_on='p_id', how='left').drop('p_id', axis=1, errors='ignore')
             text_df = pd.merge(text_df, sentences_hypothesis, left_on='hypothesis_id', right_on='h_id', how='left').drop('h_id', axis=1, errors='ignore')

             final_text_cols = ['pair_id', 'premise_text', 'hypothesis_text', 'label']
             if not all(col in text_df.columns for col in final_text_cols): raise ValueError(f"Columns missing after text merge: {final_text_cols}")
             text_df = text_df[final_text_cols].fillna({'premise_text': '', 'hypothesis_text': ''}) # Fill NaNs in text
             text_df['pair_id'] = text_df['pair_id'].astype(str) # Ensure consistent pair_id type
             logger.info(f"{self.MODEL_NAME}: Successfully loaded and merged text data. Shape: {text_df.shape}")

        except Exception as e:
            logger.error(f"{self.MODEL_NAME}: Error loading/merging text data: {e}", exc_info=True)
            return None

        # 2. Load precomputed features (expect combined stats+syntactic, will filter later)
        features_df = None
        try:
            loader = SimpleParquetLoader()
            # Pass necessary context for loader (self might not be needed if loader is static/standalone)
            features_df = loader.load_data(self, dataset, split, suffix)
            if features_df is None or features_df.empty:
                 # Fallback to DB handler if loader fails
                 precomputed_feature_table_name = f"{dataset}_{split}_features_stats_syntactic_{suffix}" # Adapt if name differs
                 logger.warning(f"SimpleParquetLoader failed. Trying DB handler with table: {precomputed_feature_table_name}")
                 features_df = self.db_handler.load_dataframe(dataset, split, precomputed_feature_table_name)

            if features_df is None or features_df.empty:
                 raise FileNotFoundError(f"Failed to load features via Loader and DB.")

            features_df = _handle_nan_values(features_df, f"{dataset}/{split}/{suffix}_features")
            if 'pair_id' not in features_df.columns: raise ValueError("'pair_id' missing in features file.")
            features_df['pair_id'] = features_df['pair_id'].astype(str) # Ensure consistent pair_id type
            logger.info(f"{self.MODEL_NAME}: Successfully loaded precomputed features. Shape: {features_df.shape}")
        except Exception as e:
            logger.error(f"{self.MODEL_NAME}: Error loading precomputed features: {e}", exc_info=True)
            return None

        # 3. Clean text data first to get the reference pair_ids and labels
        clean_text_result = clean_dataset(text_df)
        if not clean_text_result: logger.error("Text data invalid or empty after cleaning."); return None
        text_df_clean, y_labels = clean_text_result # y_labels aligned with text_df_clean

        # 4. Filter ONLY syntactic features needed from features_df
        # Use stored columns if available (from training/load), otherwise filter now
        if self._syntactic_feature_columns is None:
            self._syntactic_feature_columns = filter_syntactic_features(features_df)
        syntactic_cols_to_keep = self._syntactic_feature_columns if self._syntactic_feature_columns else []
        if not syntactic_cols_to_keep:
            logger.warning("No syntactic feature columns identified or specified. Proceeding without them.")
        else:
             logger.info(f"{self.MODEL_NAME}: Filtering for {len(syntactic_cols_to_keep)} syntactic features.")
        # Select pair_id and the syntactic columns
        syntactic_features_to_merge = features_df[['pair_id'] + syntactic_cols_to_keep].copy()

        # 5. Merge cleaned text data with filtered syntactic features
        # Use 'inner' merge to keep only rows present in both cleaned text and features
        text_to_merge = text_df_clean[['pair_id', 'premise_text', 'hypothesis_text']] # Already cleaned
        combined_df = pd.merge(text_to_merge, syntactic_features_to_merge, on='pair_id', how='inner')
        num_merged = len(combined_df)
        logger.info(f"{self.MODEL_NAME}: Merged text and syntactic features: {num_merged} rows.")
        if num_merged == 0: logger.error("No rows remained after merging."); return None

        # 6. Re-align labels 'y' based on the final merged pair_ids
        label_map = dict(zip(text_df_clean['pair_id'], y_labels))
        final_y = np.array([label_map.get(pid) for pid in combined_df['pair_id']])
        # Check for None labels (indicating merge/alignment issue)
        valid_mask = (final_y != None)
        if not np.all(valid_mask):
            missing_count = len(final_y) - np.sum(valid_mask)
            logger.error(f"Label alignment failed for {missing_count} rows after final merge.")
            # Filter out rows with missing labels
            combined_df = combined_df[valid_mask]
            final_y = final_y[valid_mask].astype(int)
            if combined_df.empty: logger.error("DataFrame empty after removing misaligned labels."); return None
        else:
            final_y = final_y.astype(int)

        if len(final_y) != len(combined_df):
            logger.error(f"Final label count ({len(final_y)}) mismatch with DataFrame rows ({len(combined_df)}).")
            return None

        X_df = combined_df # Contains pair_id, text columns, and filtered syntactic feature columns
        return X_df, final_y


    def _build_feature_pipeline_unfitted(self, sample_df_for_fitting: Optional[pd.DataFrame] = None) -> Pipeline:
        """Builds the UNFITTED feature extraction pipeline (FeatureUnion)."""
        logger.info(f"{self.MODEL_NAME}: Building unfitted feature pipeline...")
        feature_union_list = []

        # TF-IDF part
        max_feat_per_part = (self.tfidf_max_features // 2) if self.tfidf_max_features else None
        if max_feat_per_part is None or max_feat_per_part > 0:
            tfidf_premise = Pipeline([
                ('selector', TextSelector(key='premise_text')),
                ('tfidf', TfidfVectorizer(max_features=max_feat_per_part, ngram_range=self.tfidf_ngram_range, stop_words='english'))
            ])
            tfidf_hypothesis = Pipeline([
                ('selector', TextSelector(key='hypothesis_text')),
                ('tfidf', TfidfVectorizer(max_features=max_feat_per_part, ngram_range=self.tfidf_ngram_range, stop_words='english'))
            ])
            feature_union_list.extend([('tfidf_premise', tfidf_premise), ('tfidf_hypothesis', tfidf_hypothesis)])
        else: logger.warning("TF-IDF features excluded (max_features=0).")

        # Syntactic part
        # Infer columns from sample data if not already known (from loading)
        if self._syntactic_feature_columns is None:
            if sample_df_for_fitting is None:
                 raise RuntimeError("Syntactic columns needed but no sample data provided for inference.")
            logger.debug("Inferring syntactic columns from sample data for pipeline build.")
            self._syntactic_feature_columns = filter_syntactic_features(sample_df_for_fitting)

        syntactic_cols_to_use = self._syntactic_feature_columns if self._syntactic_feature_columns else []
        if syntactic_cols_to_use:
            syntactic_steps = [('selector', FeatureSelector(column_names=syntactic_cols_to_use))]
            if self.scale_syntactic:
                # Use StandardScaler (with_mean=False for sparse compatibility) via SparseScaler
                syntactic_steps.append(('scaler', SparseScaler(scaler=StandardScaler(with_mean=False))))
                logger.info("Adding StandardScaler (via SparseScaler) for syntactic features.")
            syntactic_pipe = Pipeline(syntactic_steps)
            feature_union_list.append(('syntactic', syntactic_pipe))
            logger.info(f"Including {len(syntactic_cols_to_use)} syntactic features in pipeline.")
        else:
            logger.warning("No syntactic features included in pipeline.")

        if not feature_union_list:
            raise RuntimeError(f"{self.MODEL_NAME}: Feature pipeline cannot be built - no features configured.")

        # Combine using FeatureUnion
        combined_features = FeatureUnion(feature_union_list)
        # Wrap in a final pipeline step for clarity if needed, though FeatureUnion is often the final step
        pipeline = Pipeline([('features', combined_features)])
        logger.info(f"{self.MODEL_NAME}: Unfitted feature pipeline built.")
        return pipeline


    # --- NLIModel Methods ---

    def extract_features(self, data: pd.DataFrame) -> Any:
        """
        Extracts features using the pipeline. Fits if called in training context.
        """
        if not isinstance(data, pd.DataFrame): raise TypeError("Input must be a pandas DataFrame.")
        if data.empty: return csr_matrix((0,0)) # Handle empty input

        if not self.is_trained:
            # --- Training Phase: Fit Pipeline and Transform ---
            logger.info(f"{self.MODEL_NAME}: Fitting feature pipeline and transforming training data...")
            start_time = time.time()
            # Build the unfitted pipeline structure first
            unfitted_pipeline = self._build_feature_pipeline_unfitted(sample_df_for_fitting=data)
            try:
                # Fit the pipeline on the data and transform it
                X_transformed = unfitted_pipeline.fit_transform(data)
                # Store the *fitted* pipeline
                self.feature_pipeline = unfitted_pipeline
                # Store the actual syntactic columns used (extracted from fitted selector)
                try:
                     fitted_selector = self.feature_pipeline.named_steps['features'].transformer_list[-1][1].named_steps['selector']
                     self._syntactic_feature_columns = fitted_selector.columns_seen_during_fit_
                except (KeyError, IndexError, AttributeError) as e:
                     logger.warning(f"Could not reliably get syntactic columns from fitted pipeline: {e}")

                fit_time = time.time() - start_time
                logger.info(f"Pipeline fit & transform complete in {fit_time:.2f}s. Shape: {X_transformed.shape}")
                return X_transformed
            except Exception as e:
                logger.error(f"Error during pipeline fit_transform: {e}", exc_info=True)
                raise
        else:
            # --- Evaluation/Prediction Phase: Transform Only ---
            logger.debug(f"{self.MODEL_NAME}: Transforming data using fitted pipeline...")
            if not self.feature_pipeline:
                raise RuntimeError("Model is trained, but feature pipeline is missing.")
            try:
                X_transformed = self.feature_pipeline.transform(data)
                logger.debug(f"Transform complete. Shape: {X_transformed.shape}")
                # Convert to dense for Gradient Boosting if needed (GBoost prefers dense)
                if isinstance(X_transformed, csr_matrix):
                    logger.debug("Converting sparse matrix to dense for Gradient Boosting.")
                    X_transformed = X_transformed.toarray()
                # Handle potential NaNs introduced by scaling/transformations BEFORE prediction
                if np.isnan(X_transformed).any():
                     logger.warning("NaNs detected in transformed data! Filling with 0.")
                     X_transformed = np.nan_to_num(X_transformed, nan=0.0)
                return X_transformed
            except Exception as e:
                 logger.error(f"Error during pipeline transform: {e}", exc_info=True)
                 expected_cols = self._syntactic_feature_columns
                 logger.error(f"Pipeline expected syntactic cols: {expected_cols}")
                 logger.error(f"Input data columns: {data.columns.tolist()}")
                 raise


    def train(self, X: Any, y: np.ndarray) -> None:
        """Trains the Gradient Boosting classifier on transformed features."""
        if X is None or y is None: raise ValueError("Training needs non-null X and y.")
        if X.shape[0] != len(y): raise ValueError(f"Feature/Label mismatch: X={X.shape}, y={len(y)}")
        if X.shape[0] == 0: logger.warning("Cannot train classifier with 0 samples."); return

        # Convert X to dense if it's sparse (GBoost usually prefers dense)
        if isinstance(X, csr_matrix):
            logger.info("Converting sparse training data to dense for Gradient Boosting.")
            X = X.toarray()

        # Handle potential NaNs before training
        if np.isnan(X).any():
            logger.warning("NaNs detected in training data before fitting classifier! Filling with 0.")
            X = np.nan_to_num(X, nan=0.0)


        logger.info(f"Training {self.MODEL_NAME} classifier on {X.shape[0]} samples, {X.shape[1]} features...")
        start_time = time.time()
        # Initialize the classifier here
        self.model = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state
            # Add other relevant GB params if needed (e.g., min_samples_split)
        )
        try:
            self.model.fit(X, y)
            self.is_trained = True # Mark trained only after successful fit
            train_time = time.time() - start_time
            logger.info(f"Classifier training complete in {train_time:.2f}s.")
        except Exception as e:
            logger.error(f"Error during classifier fitting: {e}", exc_info=True)
            self.is_trained = False # Ensure not marked trained on failure
            raise


    def predict(self, X: Any) -> np.ndarray:
        """Makes predictions. Expects *already transformed* features."""
        if not self.is_trained or self.model is None:
            raise RuntimeError(f"{self.MODEL_NAME} is not trained or model is missing.")
        if X is None: raise ValueError("Input X for prediction is None.")

        # Ensure X is dense for Gradient Boosting prediction
        if isinstance(X, csr_matrix):
             logger.debug("Converting sparse input to dense for prediction.")
             X = X.toarray()

        # Handle NaNs in prediction input
        if np.isnan(X).any():
            logger.warning("NaNs detected in prediction input data! Filling with 0.")
            X = np.nan_to_num(X, nan=0.0)

        # Check feature dimensions
        if hasattr(self.model, 'n_features_in_') and X.shape[1] != self.model.n_features_in_:
            raise ValueError(f"Prediction feature dim mismatch: Input={X.shape[1]}, Expected={self.model.n_features_in_}")

        logger.debug(f"Predicting with classifier on shape {X.shape}...")
        try:
            predictions = self.model.predict(X)
            logger.debug("Prediction complete.")
            return predictions
        except Exception as e:
            logger.error(f"Error during classifier prediction: {e}", exc_info=True)
            raise

    # NEW: evaluate method implementation
    def evaluate(self, dataset_name: str, split: str, suffix: str) -> Dict[str, Any]:
        """
        Evaluates the trained model on a given dataset split.
        Loads data, transforms features, predicts, and calculates metrics.
        """
        if not self.is_trained or self.feature_pipeline is None or self.model is None:
            logger.error(f"Cannot evaluate: {self.MODEL_NAME} is not trained or components missing.")
            return {'status': 'Evaluation failed: Model not ready.'}

        logger.info(f"--- Starting evaluation for {self.MODEL_NAME} on {dataset_name}/{split}/{suffix} ---")
        eval_start_time = time.time()

        # 1. Load and prepare data (expects DataFrame with text and features)
        prepared_data = self._load_and_prepare_data(dataset_name, split, suffix)
        if prepared_data is None:
            return {'status': f'Evaluation failed: Cannot load/prepare data for split {split}.'}
        df_eval, y_true = prepared_data

        if df_eval.empty or y_true is None or len(y_true) == 0:
            logger.warning(f"No valid samples for evaluation split {split} after loading/preparation.")
            return {'status': f'No valid samples for split {split}', 'accuracy': 0.0, 'f1': 0.0}

        # 2. Extract features (using the fitted pipeline, transform only)
        try:
            X_eval = self.extract_features(df_eval) # Should return dense array after NaN handling
        except Exception as e:
            logger.error(f"Error transforming features during evaluation for {split}: {e}", exc_info=True)
            return {'status': f'Evaluation failed: Feature transformation error - {e}'}

        # 3. Make predictions using the trained classifier model
        try:
            y_pred = self.predict(X_eval) # Pass the transformed features
        except Exception as e:
            logger.error(f"Error during prediction in evaluation for {split}: {e}", exc_info=True)
            return {'status': f'Evaluation failed: Prediction error - {e}'}

        # 4. Calculate metrics using the helper
        logger.info("Calculating evaluation metrics...")
        # _evaluate_model_performance needs the model instance, X, y_true
        # We pass X_eval (transformed features) and y_true
        # Note: The helper calls model.predict(X) internally, so we pass the instance 'self'
        eval_metrics_time, metrics = _evaluate_model_performance(self, X_eval, y_true)
        metrics['eval_time'] = time.time() - eval_start_time # Total eval time
        metrics['eval_metrics_time'] = eval_metrics_time # Time for predict + metric calculation

        logger.info(f"Eval Metrics - Acc: {metrics.get('accuracy', 0):.4f}, F1: {metrics.get('f1', 0):.4f}")
        logger.info(f"Evaluation complete for {self.MODEL_NAME} on {split}. Total time: {metrics.get('eval_time', 0):.2f}s")
        return metrics


    def save(self, directory: str, model_name_base: str) -> None:
        """Saves the fitted pipeline, classifier, and metadata."""
        if not self.is_trained or self.feature_pipeline is None or self.model is None:
            logger.warning(f"Attempting to save {self.MODEL_NAME} before full training/fitting.")
            # return # Optional: prevent saving if not ready

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
            if self.model: joblib.dump(self.model, classifier_path)
            joblib.dump(metadata, metadata_path)
            logger.info(f"{self.MODEL_NAME} components saved successfully.")
        except Exception as e:
            logger.error(f"Error saving {self.MODEL_NAME} artifacts: {e}", exc_info=True)
            # Clean up partial files if error occurs
            for p in [pipeline_path, classifier_path, metadata_path]:
                if os.path.exists(p): os.remove(p)
            raise

    @classmethod
    def load(cls, directory: str, model_name_base: str) -> 'GradientBoostingTFIDFSyntacticExperiment6':
        """Loads the pipeline, classifier, and metadata."""
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
        instance.model = loaded_classifier # Assign loaded classifier
        instance.is_trained = loaded_metadata.get('is_trained', False)
        instance._syntactic_feature_columns = loaded_metadata.get('_syntactic_feature_columns')

        # Post-load checks
        if not isinstance(instance.feature_pipeline, Pipeline): logger.warning("Loaded feature_pipeline is not Pipeline.")
        if not isinstance(instance.model, GradientBoostingClassifier): logger.warning("Loaded classifier is not GradientBoostingClassifier.")
        if instance.is_trained and instance._syntactic_feature_columns is None: logger.warning("Trained model loaded but syntactic columns missing in metadata.")

        logger.info(f"{cls.MODEL_NAME} loaded successfully. Trained: {instance.is_trained}")
        return instance