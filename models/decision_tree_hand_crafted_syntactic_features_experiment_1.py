# File: IS567FP/models/decision_tree_hand_crafted_syntactic_features_experiment_1.py
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler # Assuming syntactic features are numeric

# Assuming config is importable and defines necessary constants
try:
    import config
except ModuleNotFoundError:
    print("Warning: config.py not found. Using placeholder values.")
    class ConfigPlaceholder:
        RANDOM_SEED = 42
        DECISION_TREE_MAX_DEPTH = None
        DECISION_TREE_MIN_SAMPLES_SPLIT = 2
        DECISION_TREE_MIN_SAMPLES_LEAF = 1
    config = ConfigPlaceholder()

# Make sure BaselineBase can be imported
try:
    from models.baseline_base import BaselineBase
except ModuleNotFoundError:
    print("Warning: baseline_base.py not found. Defining a dummy BaselineBase.")
    class BaselineBase:
        def __init__(self, **kwargs):
            self.model_config = kwargs
            self.name = "Dummy Base Model"
            self.description = "Base class placeholder"
            self.feature_type = 'text' # Default or example
            self.pipeline = None
            self.params = {}
        # Add dummy methods like fit, predict, etc. as in the previous file if needed

# Placeholder for extracting syntactic features (assuming they are precomputed)
# In the actual pipeline, data passed to fit/predict should already contain these features.
class SyntacticFeatureSelector:
    """
    Placeholder transformer assuming input X is a DataFrame
    and we need to select columns corresponding to syntactic features.
    Alternatively, if X passed is already just the numeric features, this isn't needed.
    """
    def __init__(self, feature_columns):
        self.feature_columns = feature_columns

    def fit(self, X, y=None):
        return self # No fitting needed

    def transform(self, X):
        # Assuming X is a pandas DataFrame
        if hasattr(X, 'columns'):
            # Ensure all expected columns exist
            missing_cols = [col for col in self.feature_columns if col not in X.columns]
            if missing_cols:
                raise ValueError(f"Missing syntactic feature columns in input data: {missing_cols}")
            return X[self.feature_columns].values # Return as numpy array
        else:
            # If X is already a numpy array, assume it's the correct features
            # Potentially add shape validation based on expected number of features
            print("Warning: Input X is not a DataFrame. Assuming it contains the correct syntactic features.")
            return X

class DecisionTreeSyntacticExperiment1(BaselineBase):
    """
    Experiment 1: Decision Tree + Hand-crafted Syntactic Features
    Decision Tree using only hand-crafted features derived from parse trees.
    Assumes features are pre-computed and passed in the input data.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Experiment 1: Decision Tree + Syntactic Features"
        self.description = "Decision Tree classifier using only hand-crafted syntactic features."
        self.model_config_fields = ['max_depth', 'min_samples_split', 'min_samples_leaf', 'random_state']
        self.feature_type = 'syntactic' # Indicates uses precomputed syntactic features

        # Define which columns contain the syntactic features (example)
        # This should ideally come from config or be determined dynamically
        # If the trainer handles feature selection, this might not be needed here.
        self.syntactic_feature_columns = config.SYNTACTIC_FEATURE_COLUMNS if hasattr(config, 'SYNTACTIC_FEATURE_COLUMNS') else []
        if not self.syntactic_feature_columns:
             print(f"Warning: Syntactic feature columns not defined in config. {self.name} might not work correctly.")


        # Get hyperparameters from config or use defaults
        self.params = {
            'max_depth': config.DECISION_TREE_MAX_DEPTH,
            'min_samples_split': config.DECISION_TREE_MIN_SAMPLES_SPLIT,
            'min_samples_leaf': config.DECISION_TREE_MIN_SAMPLES_LEAF,
            'random_state': config.RANDOM_SEED
        }

        # Update params with any kwargs passed during instantiation
        relevant_kwargs = {k: v for k, v in kwargs.items() if k in self.model_config_fields}
        self.params.update(relevant_kwargs)
        self.model_config = self.params.copy() # Store the actual used config

        # Define the pipeline - Assuming input 'X' to fit/predict will be the data
        # containing the syntactic features.
        # The BaselineTrainer should pass the correct feature set based on self.feature_type
        self.pipeline = Pipeline([
            # Optional: Step to select/verify the correct columns if X is a DataFrame
            # ('selector', SyntacticFeatureSelector(self.syntactic_feature_columns)), # Uncomment if needed
            # Optional: Scaling for numeric features (though less critical for Trees)
            ('scaler', StandardScaler()),
            ('classifier', DecisionTreeClassifier(
                max_depth=self.params['max_depth'],
                min_samples_split=self.params['min_samples_split'],
                min_samples_leaf=self.params['min_samples_leaf'],
                random_state=self.params['random_state']
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
        # Add scaler params if needed for grid search (e.g., scaler__with_mean)
        # scaler_params = {k: v for k, v in pipeline_params.items() if k.startswith('scaler__')}
        # all_params = {**model_params, **scaler_params}
        all_params = model_params # Keep it simple if only tuning the classifier
        return all_params
