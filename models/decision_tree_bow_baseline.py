# File: IS567FP/models/decision_tree_bow_baseline.py
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

# Assuming config is importable and defines necessary constants
try:
    import config
except ModuleNotFoundError:
    print("Warning: config.py not found. Using placeholder values.")
    # Define placeholder config values if config.py is not available
    # In a real scenario, ensure config.py is in the Python path
    class ConfigPlaceholder:
        RANDOM_SEED = 42
        DECISION_TREE_MAX_DEPTH = None
        DECISION_TREE_MIN_SAMPLES_SPLIT = 2
        DECISION_TREE_MIN_SAMPLES_LEAF = 1
        BOW_MAX_FEATURES = 5000
        BOW_NGRAM_RANGE = (1, 1)
        BOW_STOP_WORDS = 'english'
        BOW_LOWERCASE = True
        BOW_BINARY = False
    config = ConfigPlaceholder()

# Make sure BaselineBase can be imported
try:
    from models.baseline_base import BaselineBase
except ModuleNotFoundError:
    print("Warning: baseline_base.py not found. Defining a dummy BaselineBase.")
    # Define a dummy base class if not found
    class BaselineBase:
        def __init__(self, **kwargs):
            self.model_config = kwargs
            self.name = "Dummy Base Model"
            self.description = "Base class placeholder"
            self.feature_type = 'text' # Default or example
            self.pipeline = None
            self.params = {}

        def fit(self, X, y):
             if self.pipeline:
                 print(f"Fitting {self.name}")
                 self.pipeline.fit(X, y)
             else:
                 print("Pipeline not defined.")

        def predict(self, X):
             if self.pipeline:
                 return self.pipeline.predict(X)
             else:
                 print("Pipeline not defined.")
                 return None

        def predict_proba(self, X):
             if self.pipeline:
                try:
                    return self.pipeline.predict_proba(X)
                except AttributeError:
                    print(f"{self.name} classifier does not support predict_proba.")
                    # Return dummy probabilities based on predict for compatibility
                    predictions = self.predict(X)
                    # Assuming binary classification [class 0, class 1]
                    # This is a placeholder and might not be suitable for all metrics
                    n_samples = len(predictions) if hasattr(predictions, '__len__') else 0
                    proba = [[1.0, 0.0] if p == 0 else [0.0, 1.0] for p in predictions] # Simplified
                    # Try to get classes_ if possible, otherwise assume 0 and 1
                    try:
                       classes = self.pipeline.classes_
                       n_classes = len(classes)
                       proba = np.zeros((n_samples, n_classes))
                       for i, p in enumerate(predictions):
                           class_idx = np.where(classes == p)[0][0]
                           proba[i, class_idx] = 1.0
                    except Exception:
                        # Fallback to simple binary assumption if classes_ fails
                         proba = [[1.0, 0.0] if p == 0 else [0.0, 1.0] for p in predictions]

                    return np.array(proba)


             else:
                 print("Pipeline not defined.")
                 return None

        def get_pipeline(self):
            return self.pipeline

        def get_params(self):
            return self.params

        def get_config(self):
            # Return model-specific configuration used
            config_dict = {'name': self.name}
            if hasattr(self, 'model_config_fields'):
                 config_dict.update({k: self.params.get(k) for k in self.model_config_fields})
            return config_dict


class DecisionTreeBowBaseline(BaselineBase):
    """
    Baseline 1: Decision Tree + Bag-of-Words (BoW)
    Simple and fast decision tree classifier using word counts as features.
    Feature extraction via CountVectorizer.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Baseline 1: Decision Tree + BoW"
        self.description = "Decision Tree classifier using Bag-of-Words features."
        self.model_config_fields = ['max_depth', 'min_samples_split', 'min_samples_leaf', 'random_state']
        self.feature_type = 'bow' # Indicates uses 'text' column for BOW

        # Get hyperparameters from config or use defaults
        self.params = {
            'max_depth': config.DECISION_TREE_MAX_DEPTH,
            'min_samples_split': config.DECISION_TREE_MIN_SAMPLES_SPLIT,
            'min_samples_leaf': config.DECISION_TREE_MIN_SAMPLES_LEAF,
            'random_state': config.RANDOM_SEED
        }

        # Update params with any kwargs passed during instantiation
        # Filter kwargs to only include expected hyperparameters
        relevant_kwargs = {k: v for k, v in kwargs.items() if k in self.model_config_fields}
        self.params.update(relevant_kwargs)
        self.model_config = self.params.copy() # Store the actual used config

        # Define the pipeline
        self.pipeline = Pipeline([
            ('vectorizer', CountVectorizer(
                max_features=config.BOW_MAX_FEATURES,
                ngram_range=config.BOW_NGRAM_RANGE,
                stop_words=config.BOW_STOP_WORDS if config.BOW_STOP_WORDS else None, # Handle None case
                lowercase=config.BOW_LOWERCASE,
                binary=config.BOW_BINARY
                # Add other CountVectorizer parameters from config if needed
            )),
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
        # Return relevant pipeline parameters, especially for grid search or logging
        pipeline_params = self.pipeline.get_params() # Gets params of steps
        # Combine classifier params and vectorizer params
        model_params = {f'classifier__{k}': v for k, v in self.params.items()}

        # Manually add vectorizer params from config as they are set directly
        vectorizer_params = {
             'vectorizer__max_features': config.BOW_MAX_FEATURES,
             'vectorizer__ngram_range': config.BOW_NGRAM_RANGE,
             'vectorizer__stop_words': config.BOW_STOP_WORDS,
             'vectorizer__lowercase': config.BOW_LOWERCASE,
             'vectorizer__binary': config.BOW_BINARY
        }
        # Filter vectorizer params to only include those settable/gettable in CountVectorizer
        cv_params_available = CountVectorizer().get_params().keys()
        vectorizer_params_filtered = {k: v for k, v in vectorizer_params.items() if k.split('__')[1] in cv_params_available}


        # Combine all parameters
        all_params = {**model_params, **vectorizer_params_filtered}
        return all_params
