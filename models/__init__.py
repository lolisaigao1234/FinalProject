# models/__init__.py
from .transformer_model import BERTWithSyntacticAttention
from .SVMTrainer import SVMTrainer, SVMWithBagOfWords, SVMWithSyntax, SVMWithBothFeatures # Keep SVM models if needed
from .NeuroTrainer import ModelTrainer, NLIDataset # Make NLIDataset available if used elsewhere
from .baseline_transformer import BaselineTransformerNLI
from .gcn_transformer import TransformerWithGCN
from .logistic_tf_idf_baseline import LogisticTFIDFBaseline, LogisticRegressionTrainer # <-- Add this line

__all__ = [
    'BERTWithSyntacticAttention',
    'SVMTrainer',
    'SVMWithBagOfWords',
    'SVMWithSyntax',
    'SVMWithBothFeatures',
    'ModelTrainer',
    'NLIDataset',
    'BaselineTransformerNLI',
    'TransformerWithGCN',
    'LogisticTFIDFBaseline', # <-- Add this line
    'LogisticRegressionTrainer' # <-- Add this line
]