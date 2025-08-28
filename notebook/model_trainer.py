import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from pathlib import Path
import json

from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier,
    ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.utils import DataFrameUtils

# Artifacts dir
ARTIFACTS = Path("artifacts"); ARTIFACTS.mkdir(exist_ok=True)

# Load and preprocess data
df = pd.read_csv("notebook/data/Telco-Customer-Churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
df["SeniorCitizen"] = df["SeniorCitizen"].astype('object')
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

X = df.drop("Churn", axis=1)
y = df["Churn"]

cat_cols, num_cols = DataFrameUtils(X).grab_col_names(cat_th=10, car_th=20)

# Pipelines for preprocessing
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

def create_model(model, X_train, y_train):
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    # ROC-AUC with proba or decision scores
    roc = None
    if hasattr(model, 'predict_proba'):
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            roc = roc_auc_score(y_test, y_proba)
        except Exception:
            roc = None
    elif hasattr(model, 'decision_function'):
        try:
            y_score = model.decision_function(X_test)
            roc = roc_auc_score(y_test, y_score)
        except Exception:
            roc = None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print({"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": roc})

    with open(ARTIFACTS / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump({"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": roc}, f, ensure_ascii=False, indent=2)

def model_selection(X_train, y_train, X_test, y_test):
    grids = {
        "RandomForest": (RandomForestClassifier(random_state=42), {
            'classifier__n_estimators': [200, 400],
            'classifier__max_depth': [None, 10, 20]
        }),
        "LogReg": (LogisticRegression(max_iter=1000), {
            'classifier__C': [0.5, 1.0, 2.0],
            'classifier__penalty': ['l2']
        }),
        "XGB": (XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=300, learning_rate=0.1), {
            'classifier__max_depth': [3, 5],
            'classifier__subsample': [0.8, 1.0]
        }),
        "CatBoost": (CatBoostClassifier(verbose=0, random_state=42), {
            'classifier__depth': [4, 6],
            'classifier__iterations': [300, 600]
        })
    }

    best_model = None
    best_name = None
    best_cv = -np.inf

    for name, (clf, param_grid) in grids.items():
        print(f"GridSearch {name}...")
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])
        gs = GridSearchCV(pipe, param_grid=param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=0)
        gs.fit(X_train, y_train)
        print(f"{name} best CV ROC_AUC: {gs.best_score_:.4f}, params: {gs.best_params_}")
        if gs.best_score_ > best_cv:
            best_cv = gs.best_score_
            best_model = gs.best_estimator_
            best_name = name

    print(f"Best Model: {best_name} | CV ROC_AUC: {best_cv:.4f}")
    # Evaluate on test and persist metrics
    evaluate_model(best_model, X_test, y_test)
    return best_model

def feature_importance(model):
    clf = model.named_steps['classifier']
    if hasattr(clf, 'feature_importances_'):
        feature_names = model.named_steps['preprocessor'].get_feature_names_out()
        importances = clf.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(12, 6))
        plt.title("Feature Importances")
        plt.bar(range(len(importances[:30])), importances[indices][:30], align='center')
        plt.xticks(range(len(importances[:30])), [feature_names[i] for i in indices[:30]], rotation=90)
        plt.tight_layout()
        plt.savefig(ARTIFACTS / 'feature_importance.png', dpi=150)
        plt.close()
    else:
        print(f"{clf.__class__.__name__} does not support feature_importances_.")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and select best model
best_model = model_selection(X_train, y_train, X_test, y_test)

# Save and reload model
joblib.dump(best_model, 'best_model.pkl')
loaded_model = joblib.load('best_model.pkl')
evaluate_model(loaded_model, X_test, y_test)

# Show feature importances
feature_importance(loaded_model)
