# Modify: IS567FP/models/__init__.py

# models/__init__.py
# Add MultinomialNaiveBayesBaseline and its trainer
from .svm_bow_baseline import SVMWithBagOfWords, SVMWithSyntax, SVMWithBothFeatures # Keep SVM models if needed
from .logistic_tf_idf_baseline import LogisticTFIDFBaseline
from .multinomial_naive_bayes_bow_baseline import MultinomialNaiveBayesBaseline
# Import the new Experiment 1 model
from .svm_hand_crafted_syntactic_features_experiment_1 import SVMHandcraftedSyntacticExperiment1

__all__ = [
    # 'SVMTrainer', # SVMTrainer class removed, handled by BaselineTrainer
    'SVMWithBagOfWords',
    'SVMWithSyntax',
    'SVMWithBothFeatures',
    'SVMHandcraftedSyntacticExperiment1', # <-- Add new model
    'LogisticTFIDFBaseline',
    # 'LogisticRegressionTrainer', # Trainer class removed
    'MultinomialNaiveBayesBaseline',
    # 'MultinomialNaiveBayesTrainer' # Trainer class removed
]