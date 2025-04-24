# models/__init__.py
from .transformer_model import BERTWithSyntacticAttention
from .SVMTrainer import SVMTrainer
from .NeuroTrainer import ModelTrainer
from .baseline_transformer import BaselineTransformerNLI # <-- Add this line

__all__ = [
    'BERTWithSyntacticAttention',
    'SVMTrainer',
    'ModelTrainer',
    'BaselineTransformerNLI' # <-- Add this line
]