# train.py
# Train leak-free pipeline model and save to artifacts/model.joblib
# Dataset: UCI Wine Quality (red + white)
# Label: binary (0 if quality<=5, 1 if quality>=6)
from iqr_clipper import IQRClipper
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

from lightgbm import LGBMClassifier

# ======================
# CONFIG
# ======================
RANDOM_STATE = 42
TEST_SIZE = 0.2

LABEL_THRESHOLD = 6        # 1 if quality>=6 else 0
SCALER_MODE = "standard"   # "standard" or "minmax"
IQR_FACTOR = 1.5

ART_DIR = Path("artifacts")
ART_DIR.mkdir(exist_ok=True)

# ======================
# DATA LOADING (auto-download)
# ======================
def load_data_uci():
    red_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    white_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

    df_red = pd.read_csv(red_url, sep=";")
    df_white = pd.read_csv(white_url, sep=";")
    df_red["wine_type"] = "red"
    df_white["wine_type"] = "white"
    df = pd.concat([df_red, df_white], ignore_index=True)
    return df

# ======================
# CUSTOM TRANSFORMER: IQR clipping
# ======================
class IQRClipper(BaseEstimator, TransformerMixin):
    """
    Fit on train only (inside pipeline) -> leak-free.
    Clips numeric features to [Q1 - factor*IQR, Q3 + factor*IQR] per feature.
    """
    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.q1_ = np.nanpercentile(X, 25, axis=0)
        self.q3_ = np.nanpercentile(X, 75, axis=0)
        self.iqr_ = self.q3_ - self.q1_
        self.lower_ = self.q1_ - self.factor * self.iqr_
        self.upper_ = self.q3_ + self.factor * self.iqr_
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.lower_, self.upper_)

# ======================
# PIPELINE
# ======================
def make_preprocess(X: pd.DataFrame):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    scaler = StandardScaler() if SCALER_MODE == "standard" else MinMaxScaler()

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("iqr", IQRClipper(factor=IQR_FACTOR)),
        ("scaler", scaler),
    ])

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop"
    )
    return pre

def build_model_pipeline(preprocess):
    # LGBM settings: quiet logs
    clf = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
        is_unbalance=True,
        verbose=-1,
        verbosity=-1,
    )

    pipe = Pipeline(steps=[
        ("preprocess", preprocess),
        ("clf", clf),
    ])
    return pipe

# ======================
# TRAIN
# ======================
def main():
    print("Loading data from UCI...")
    df = load_data_uci()

    # target
    y = (df["quality"] >= LABEL_THRESHOLD).astype(int)
    X = df.drop(columns=["quality"]).copy()

    print(f"Samples: {len(df)} | Missing values: {int(df.isna().sum().sum())}")
    print(f"Label ratio (1=high): {y.mean():.3f}")

    preprocess = make_preprocess(X)
    model = build_model_pipeline(preprocess)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    print("Training model...")
    model.fit(X_train, y_train)

    # Evaluate quickly
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred, zero_division=0)
    rec = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)
    auc = roc_auc_score(y_test, proba)

    print(f"Test metrics: Acc={acc:.4f} | Prec={prec:.4f} | Rec={rec:.4f} | F1={f1:.4f} | AUC={auc:.4f}")

    out_path = ART_DIR / "model.joblib"
    joblib.dump(model, out_path)
    print(f"✅ Saved model to: {out_path}")

if __name__ == "__main__":
    main()
