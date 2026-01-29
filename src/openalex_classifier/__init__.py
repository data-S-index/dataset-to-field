"""
OpenAlex Topic Classifier
=========================

Lightweight CPU-based topic classification for scientific datasets
using the OpenAlex taxonomy (4,516 topics).

Usage:
    from openalex_classifier import TopicClassifier
    
    classifier = TopicClassifier()
    classifier.initialize()
    result = classifier.classify(record)
"""

from .classifier import TopicClassifier
from .config import Config

__version__ = "1.0.0"
__all__ = ["TopicClassifier", "Config"]

