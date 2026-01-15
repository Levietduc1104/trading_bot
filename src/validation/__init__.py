"""
Validation Module - Walk-Forward Testing & Out-of-Sample Validation

This module provides tools for validating trading strategies without overfitting bias.
"""

from .walk_forward import WalkForwardValidator

__all__ = ['WalkForwardValidator']
