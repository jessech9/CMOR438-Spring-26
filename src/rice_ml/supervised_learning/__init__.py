"""Supervised learning algorithms.

Modules
-------
linear         : LinearRegression, LogisticRegression
knn            : KNeighborsClassifier, KNeighborsRegressor
decision_tree  : DecisionTreeClassifier, DecisionTreeRegressor
random_forest  : RandomForestClassifier, RandomForestRegressor
"""

from .linear import LinearRegression, LogisticRegression
from .knn import KNeighborsClassifier, KNeighborsRegressor
from .decision_tree import DecisionTreeClassifier, DecisionTreeRegressor
from .random_forest import RandomForestClassifier, RandomForestRegressor

__all__ = [
    "LinearRegression",
    "LogisticRegression",
    "KNeighborsClassifier",
    "KNeighborsRegressor",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "RandomForestClassifier",
    "RandomForestRegressor",
]
