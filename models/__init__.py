# models/__init__.py
# Add MultinomialNaiveBayesBaseline and its trainer
from .svm_bow_baseline import SVMTrainer, SVMWithBagOfWords, SVMWithSyntax, SVMWithBothFeatures # Keep SVM models if needed
from .logistic_tf_idf_baseline import LogisticTFIDFBaseline, LogisticRegressionTrainer
from .multinomial_naive_bayes_bow_baseline import MultinomialNaiveBayesBaseline, MultinomialNaiveBayesTrainer # <-- Add this line

__all__ = [
    'SVMTrainer',
    'SVMWithBagOfWords',
    'SVMWithSyntax',
    'SVMWithBothFeatures',
    'LogisticTFIDFBaseline',
    'LogisticRegressionTrainer',
    'MultinomialNaiveBayesBaseline', # <-- Add this line
    'MultinomialNaiveBayesTrainer' # <-- Add this line
]