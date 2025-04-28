# models/__init__.py
# Add MultinomialNaiveBayesBaseline and its trainer
from .SVMTrainer import SVMTrainer, SVMWithBagOfWords, SVMWithSyntax, SVMWithBothFeatures # Keep SVM models if needed
from .NeuroTrainer import ModelTrainer, NLIDataset # Make NLIDataset available if used elsewhere
from .baseline_transformer import BaselineTransformerNLI
from .logistic_tf_idf_baseline import LogisticTFIDFBaseline, LogisticRegressionTrainer
from .multinomial_naive_bayes_baseline import MultinomialNaiveBayesBaseline, MultinomialNaiveBayesTrainer # <-- Add this line

__all__ = [
    'SVMTrainer',
    'SVMWithBagOfWords',
    'SVMWithSyntax',
    'SVMWithBothFeatures',
    'ModelTrainer',
    'NLIDataset',
    'BaselineTransformerNLI',
    'LogisticTFIDFBaseline',
    'LogisticRegressionTrainer',
    'MultinomialNaiveBayesBaseline', # <-- Add this line
    'MultinomialNaiveBayesTrainer' # <-- Add this line
]