# File: IS567FP/models/__init__.py


# --- Baselines ---
# Import new Decision Tree Baseline 1 (replaces SVM)
# Assuming the file 'decision_tree_bow_baseline.py' defines 'DecisionTreeBowBaseline'
from .decision_tree_bow_baseline import DecisionTreeBowBaseline
# Verified existing baselines
from .logistic_tf_idf_baseline import LogisticTFIDFBaseline
from .multinomial_naive_bayes_bow_baseline import MultinomialNaiveBayesBaseline # VERIFIED NAME

# --- Experiments ---
# Import new Decision Tree Experiment 1 (replaces SVM)
# Assuming the file 'decision_tree_hand_crafted_syntactic_features_experiment_1.py' defines 'DecisionTreeSyntacticExperiment1'
from .decision_tree_hand_crafted_syntactic_features_experiment_1 import DecisionTreeSyntacticExperiment1
# Import new KNN Experiment 2 (replaces SVM)
# Assuming the file 'knn_bow_hand_crafted_syntactic_features_experiment_2.py' defines 'KnnBowSyntacticExperiment2'
from .knn_bow_hand_crafted_syntactic_features_experiment_2 import KnnBowSyntacticExperiment2
# Verified existing experiments
from .logistic_tfidf_syntactic_experiment_3 import LogisticTFIDFSyntacticExperiment3
from .multinomial_naive_bayes_bow_syntactic_experiment_4 import MultinomialNaiveBayesBowSyntacticExperiment4 # VERIFIED NAME
from .random_forest_bow_syntactic_experiment_5 import RandomForestBowSyntacticExperiment5
from .gradient_boosting_tfidf_syntactic_experiment_6 import GradientBoostingTFIDFSyntacticExperiment6
# Verified Experiments 7 and 8 classes
from .cross_eval_syntactic_experiment_7 import CrossEvalSyntacticExperiment7
from .cross_validate_syntactic_experiment_8 import CrossValidateSyntacticExperiment8


# --- Model Registry ---
# Maps experiment names (used in main.py --experiments argument) to model classes
MODEL_REGISTRY = {
    # Baselines
    "baseline-1": DecisionTreeBowBaseline,     # NEW
    "baseline-2": LogisticTFIDFBaseline,
    "baseline-3": MultinomialNaiveBayesBaseline,    # CORRECTED NAME

    # Experiments with Hand-crafted Features
    "experiment-1": DecisionTreeSyntacticExperiment1, # NEW
    "experiment-2": KnnBowSyntacticExperiment2,      # NEW
    "experiment-3": LogisticTFIDFSyntacticExperiment3,
    "experiment-4": MultinomialNaiveBayesBowSyntacticExperiment4, # CORRECTED NAME
    "experiment-5": RandomForestBowSyntacticExperiment5,
    "experiment-6": GradientBoostingTFIDFSyntacticExperiment6,

    # Special Experiments
    "experiment-7": CrossEvalSyntacticExperiment7,
    "experiment-8": CrossValidateSyntacticExperiment8,
}

# --- Exported names ---
# Controls what 'from models import *' imports
__all__ = [
    # Baselines
    'DecisionTreeBowBaseline', # ADDED
    'LogisticTFIDFBaseline',
    'MultinomialNaiveBayesBaseline', # CORRECTED NAME

    # Experiments
    'DecisionTreeSyntacticExperiment1', # ADDED
    'KnnBowSyntacticExperiment2',      # ADDED
    'LogisticTFIDFSyntacticExperiment3',
    'MultinomialNaiveBayesBowSyntacticExperiment4', # CORRECTED NAME
    'RandomForestBowSyntacticExperiment5',
    'GradientBoostingTFIDFSyntacticExperiment6',
    'CrossEvalSyntacticExperiment7',
    'CrossValidateSyntacticExperiment8',

    # Registry
    'MODEL_REGISTRY',

    # NOTE: Ensure the old SVM class names are removed if they were previously in __all__
]