"""Supervised learning algorithms.

Submodules
----------
linear      : Linear Regression, Logistic Regression
neighbors   : k-Nearest Neighbors classifier and regressor
trees       : CART decision tree classifier and regression tree
ensembles   : Random Forest and Gradient Boosting
neural      : Perceptron and Multilayer Perceptron
"""

from .linear import LinearRegression, LogisticRegression
from .neighbors import KNeighborsClassifier, KNeighborsRegressor
from .trees import DecisionTreeClassifier, DecisionTreeRegressor
from .ensembles import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from .neural import Perceptron, MLPClassifier

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
    "GradientBoostingRegressor",
    "Perceptron",
    "MLPClassifier",
]