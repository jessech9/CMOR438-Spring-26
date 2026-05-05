"""Backward-compatible ensemble imports.

Older notebooks imported ensemble estimators from this module. The
implementations now live in ``random_forest`` and ``gradient_boosting``.
"""

from .gradient_boosting import GradientBoostingClassifier
from .random_forest import RandomForestClassifier, RandomForestRegressor

__all__ = [
    "GradientBoostingClassifier",
    "RandomForestClassifier",
    "RandomForestRegressor",
]
