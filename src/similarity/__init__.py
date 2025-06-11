"""
Similarity Analysis Module for Financial Agent

This module contains components for:
- 7-day window creation and feature extraction
- Pattern similarity calculation
- Historical pattern matching
"""

from .window_creator import WindowCreator
from .similarity_calculator import SimilarityCalculator
from .pattern_searcher import PatternSearcher

__all__ = ['WindowCreator', 'SimilarityCalculator', 'PatternSearcher'] 