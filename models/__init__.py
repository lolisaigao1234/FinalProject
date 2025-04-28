# Modify file: IS567FP/models/__init__.py
# Add import and __all__ entry for Experiment 8
from .svm_bow_baseline import SVMWithBagOfWords, SVMWithSyntax, SVMWithBothFeatures
from .logistic_tf_idf_baseline import LogisticTFIDFBaseline
from .multinomial_naive_bayes_bow_baseline import MultinomialNaiveBayesBaseline
from .svm_hand_crafted_syntactic_features_experiment_1 import SVMHandcraftedSyntacticExperiment1
from .svm_bow_hand_crafted_syntactic_features_experiment_2 import SVMBowHandCraftedSyntacticExperiment2
from .logistic_tfidf_syntactic_experiment_3 import LogisticTFIDFSyntacticExperiment3
from .multinomial_naive_bayes_bow_syntactic_experiment_4 import MultinomialNaiveBayesBowSyntacticExperiment4
from .random_forest_bow_syntactic_experiment_5 import RandomForestBowSyntacticExperiment5
# Import the new Experiment 6 model
from .gradient_boosting_tfidf_syntactic_experiment_6 import GradientBoostingTFIDFSyntacticExperiment6
# Import Experiment 7
from .cross_eval_syntactic_experiment_7 import CrossEvalSyntacticExperiment7
# <<< Import Experiment 8 >>>
from .cross_validate_syntactic_experiment_8 import CrossValidateSyntacticExperiment8


__all__ = [
    'SVMWithBagOfWords',
    'SVMWithSyntax',
    'SVMWithBothFeatures',
    'SVMHandcraftedSyntacticExperiment1',
    'SVMBowHandCraftedSyntacticExperiment2',
    'LogisticTFIDFBaseline',
    'MultinomialNaiveBayesBaseline',
    'MultinomialNaiveBayesBowSyntacticExperiment4',
    'LogisticTFIDFSyntacticExperiment3',
    'RandomForestBowSyntacticExperiment5',
    'GradientBoostingTFIDFSyntacticExperiment6',
    'CrossEvalSyntacticExperiment7',
    'CrossValidateSyntacticExperiment8' # <<< Added Experiment 8 >>>
]