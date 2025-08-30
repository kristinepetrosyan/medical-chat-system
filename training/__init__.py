"""
Training module for medical chat system
"""

from .train import main as train_model
from .evaluate import evaluate_model

__all__ = ['train_model', 'evaluate_model']
