"""Supervised learning algorithms.

Modules
-------
linear         : LinearRegression, LogisticRegression
knn            : KNeighborsClassifier, KNeighborsRegressor
decision_tree  : DecisionTreeClassifier, DecisionTreeRegressor
random_forest  : RandomForestClassifier, RandomForestRegressor
gradient_boosting : GradientBoostingClassifier
"""

from .linear import LinearRegression, LogisticRegression
from .knn import KNeighborsClassifier, KNeighborsRegressor
from .decision_tree import DecisionTreeClassifier, DecisionTreeRegressor
from .random_forest import RandomForestClassifier, RandomForestRegressor
from .gradient_boosting import GradientBoostingClassifier

__all__ = [
    "LinearRegression",
    "LogisticRegression",
    "KNeighborsClassifier",
    "KNeighborsRegressor",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "GradientBoostingClassifier",
]
