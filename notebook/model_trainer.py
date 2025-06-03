import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score
)

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
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

def model_selection(X_train, y_train, X_test, y_test):
    models = {
        "RandomForest": RandomForestClassifier(),
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "DecisionTree": DecisionTreeClassifier(),
        "CatBoost": CatBoostClassifier(verbose=0, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "LGBM": LGBMClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "ExtraTrees": ExtraTreesClassifier(),
        "SVC": SVC(),
        "KNeighbors": KNeighborsClassifier(),
        "GaussianNB": GaussianNB()
    }

    best_model = None
    best_score = 0

    for name, model in models.items():
        print(f"Training {name}...")
        pipeline = create_model(model, X_train, y_train)
        evaluate_model(pipeline, X_test, y_test)

        score = cross_val_score(pipeline, X_train, y_train, cv=5).mean()
        print(f"{name} Cross-Validation Score: {score:.4f}\n")

        if score > best_score:
            best_score = score
            best_model = pipeline

    print(f"Best Model: {best_model.named_steps['classifier'].__class__.__name__} with score: {best_score:.4f}")
    return best_model

def feature_importance(model):
    clf = model.named_steps['classifier']
    if hasattr(clf, 'feature_importances_'):
        feature_names = model.named_steps['preprocessor'].get_feature_names_out()
        importances = clf.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(12, 6))
        plt.title("Feature Importances")
        plt.bar(range(len(importances)), importances[indices], align='center')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.show()
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