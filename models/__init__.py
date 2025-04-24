# models/__init__.py
from .transformer_model import BERTWithSyntacticAttention
from .SVMTrainer import SVMTrainer
from .NeuroTrainer import ModelTrainer
from .baseline_transformer import BaselineTransformerNLI
from .gcn_transformer import TransformerWithGCN # <-- Add this line (if file exists)

__all__ = [
    'BERTWithSyntacticAttention',
    'SVMTrainer',
    'ModelTrainer',
    'BaselineTransformerNLI',
    'TransformerWithGCN' # <-- Add this line
]