"""
Model definitions and hyperparameter grid configuration.
Includes classifiers: Logistic Regression, KNN, SVM (Linear, RBF, Poly),
Random Forest, XGBoost, AdaBoost, Gradient Boosting, LDA, QDA, Naive Bayes, Decision Tree.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

import config

def get_model_zoo(random_state: int = config.RANDOM_SEED) -> dict:
    """
    Returns dictionary of classifier instances optimized for training speed and stability.
    """
    models = {
        "LogisticRegression": LogisticRegression(max_iter=200, tol=1e-3, random_state=random_state),
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "SVM_Linear": SVC(kernel='linear', max_iter=3000, random_state=random_state),
        "SVM_RBF": SVC(kernel='rbf', max_iter=3000, random_state=random_state),
        "SVM_Poly": SVC(kernel='poly', degree=3, max_iter=3000, random_state=random_state),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=random_state, eval_metric='mlogloss', n_jobs=-1),
        "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=random_state),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=50, random_state=random_state),
        "LDA": LinearDiscriminantAnalysis(),
        "QDA": QuadraticDiscriminantAnalysis(reg_param=0.1),
        "GaussianNB": GaussianNB(),
        "DecisionTree": DecisionTreeClassifier(max_depth=10, random_state=random_state)
    }
    return models

def get_hyperparameter_grids() -> dict:
    """
    Returns search grids for hyperparameter tuning.
    """
    grids = {
        "LogisticRegression": {
            "classifier__C": [0.1, 1.0, 10.0]
        },
        "KNN": {
            "classifier__n_neighbors": [3, 5, 7],
            "classifier__weights": ["uniform", "distance"]
        },
        "SVM_Linear": {
            "classifier__C": [0.1, 1.0, 10.0]
        },
        "SVM_RBF": {
            "classifier__C": [0.1, 1.0, 10.0],
            "classifier__gamma": ["scale", 0.01]
        },
        "RandomForest": {
            "classifier__n_estimators": [50, 100],
            "classifier__max_depth": [None, 10]
        },
        "XGBoost": {
            "classifier__n_estimators": [50, 100],
            "classifier__learning_rate": [0.1, 0.2]
        }
    }
    return grids
