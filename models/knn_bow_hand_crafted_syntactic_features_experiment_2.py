# File: IS567FP/models/knn_bow_hand_crafted_syntactic_features_experiment_2.py
from abc import ABC

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer

# Assuming config is importable and defines necessary constants
try:
    import config
except ModuleNotFoundError:
    print("Warning: config.py not found. Using placeholder values.")
    class ConfigPlaceholder:
        RANDOM_SEED = 42 # KNN doesn't use random_state, but other parts might
        KNN_N_NEIGHBORS = 5
        KNN_WEIGHTS = 'uniform'
        KNN_METRIC = 'minkowski'
        BOW_MAX_FEATURES = 5000
        BOW_NGRAM_RANGE = (1, 1)
        BOW_STOP_WORDS = 'english'
        BOW_LOWERCASE = True
        BOW_BINARY = False
        SYNTACTIC_FEATURE_COLUMNS = ['syntactic_feat1', 'syntactic_feat2'] # Example
    config = ConfigPlaceholder()

# Make sure BaselineBase can be imported
try:
    from models.baseline_base import TextBaselineModel
except ModuleNotFoundError:
    print("Warning: baseline_base.py not found. Defining a dummy BaselineBase.")
    class BaselineBase:
        def __init__(self, **kwargs):
            self.model_config = kwargs
            self.name = "Dummy Base Model"
            self.description = "Base class placeholder"
            self.feature_type = 'text'
            self.pipeline = None
            self.params = {}
        # Add dummy methods like fit, predict, etc. as needed

# Helper function to select columns - useful for ColumnTransformer
def select_column(df, column_name):
    return df[column_name]

# Helper function to select multiple columns - useful for ColumnTransformer
def select_columns(df, column_names):
    # Ensure columns exist
    missing_cols = [col for col in column_names if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns in DataFrame: {missing_cols}")
    return df[column_names]

class KnnBowSyntacticExperiment2(TextBaselineModel, ABC):
    """
    Experiment 2: k-Nearest Neighbors + BoW + Hand-crafted Syntactic Features
    Combines lexical features (BoW) and pre-computed hand-crafted syntactic
    features using feature union before feeding into a KNN classifier.
    Requires input data to be a DataFrame with 'text' and syntactic feature columns.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Experiment 2: KNN + BoW + Syntactic Features"
        self.description = "KNN classifier using combined BoW and hand-crafted syntactic features."
        self.model_config_fields = ['n_neighbors', 'weights', 'metric']
        # This model type needs both text and precomputed syntactic features
        self.feature_type = 'bow+syntactic'

        # Define which columns contain the syntactic features
        self.syntactic_feature_columns = config.SYNTACTIC_FEATURE_COLUMNS if hasattr(config, 'SYNTACTIC_FEATURE_COLUMNS') else []
        if not self.syntactic_feature_columns:
             print(f"Warning: Syntactic feature columns not defined in config. {self.name} might not work correctly.")

        # Get hyperparameters from config or use defaults
        self.params = {
            'n_neighbors': config.KNN_N_NEIGHBORS,
            'weights': config.KNN_WEIGHTS,
            'metric': config.KNN_METRIC
            # Add other KNN params like 'p' for minkowski if needed
        }

        # Update params with any kwargs passed during instantiation
        relevant_kwargs = {k: v for k, v in kwargs.items() if k in self.model_config_fields}
        self.params.update(relevant_kwargs)
        self.model_config = self.params.copy() # Store the actual used config

        # Define the feature processing pipeline using ColumnTransformer
        # Assumes input X to fit/predict is a pandas DataFrame
        # with a 'text' column and the syntactic feature columns

        # Transformer for Bag-of-Words on the 'text' column
        bow_transformer = Pipeline([
            ('select', FunctionTransformer(select_column, kw_args={'column_name': 'text'}, validate=False)),
            ('vectorizer', CountVectorizer(
                max_features=config.BOW_MAX_FEATURES,
                ngram_range=config.BOW_NGRAM_RANGE,
                stop_words=config.BOW_STOP_WORDS if config.BOW_STOP_WORDS else None,
                lowercase=config.BOW_LOWERCASE,
                binary=config.BOW_BINARY
            ))
        ])

        # Transformer for Syntactic features (select and scale)
        syntactic_transformer = Pipeline([
             ('select', FunctionTransformer(select_columns, kw_args={'column_names': self.syntactic_feature_columns}, validate=False)),
             ('scaler', StandardScaler()) # Scaling is important for KNN
        ])

        # Combine features using FeatureUnion within ColumnTransformer or directly
        # Using ColumnTransformer is generally more robust for DataFrame inputs
        preprocessor = ColumnTransformer(
            transformers=[
                ('bow', bow_transformer, ['text']), # Apply BoW pipeline to 'text' column
                ('syntactic', syntactic_transformer, self.syntactic_feature_columns) # Apply syntactic pipeline to feature columns
            ],
            remainder='drop' # Drop any other columns not specified
        )

        # Define the full pipeline including the classifier
        self.pipeline = Pipeline([
            ('features', preprocessor),
            ('classifier', KNeighborsClassifier(
                n_neighbors=self.params['n_neighbors'],
                weights=self.params['weights'],
                metric=self.params['metric']
                # Add other KNN parameters if needed, e.g., p=config.KNN_P
            ))
        ])

    def get_pipeline(self):
        """Returns the scikit-learn pipeline object."""
        return self.pipeline

    def get_params(self):
        """Returns the parameters for the model."""
        pipeline_params = self.pipeline.get_params()
        # Combine classifier params
        model_params = {f'classifier__{k}': v for k, v in self.params.items()}

        # Add vectorizer params from config (prefixed correctly for ColumnTransformer structure)
        vectorizer_params = {
             'features__bow__vectorizer__max_features': config.BOW_MAX_FEATURES,
             'features__bow__vectorizer__ngram_range': config.BOW_NGRAM_RANGE,
             'features__bow__vectorizer__stop_words': config.BOW_STOP_WORDS,
             'features__bow__vectorizer__lowercase': config.BOW_LOWERCASE,
             'features__bow__vectorizer__binary': config.BOW_BINARY
        }
        # Filter vectorizer params based on actual CountVectorizer params
        cv_params_available = CountVectorizer().get_params().keys()
        vectorizer_params_filtered = {k: v for k, v in vectorizer_params.items() if k.split('__')[-1] in cv_params_available}

        # Add scaler params (e.g., features__syntactic__scaler__with_mean)
        scaler_params = {k: v for k, v in pipeline_params.items() if k.startswith('features__syntactic__scaler__')}

        # Combine all parameters
        all_params = {**model_params, **vectorizer_params_filtered, **scaler_params}
        return all_params
