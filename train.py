import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

from lightgbm import LGBMClassifier, set_config

# --- CONFIG ---
RANDOM_STATE = 42
LABEL_THRESHOLD = 6          # 0:<=5, 1:>=6
SCALER_MODE = "standard"     # "standard" or "minmax"
IQR_FACTOR = 1.5

ART_DIR = Path("artifacts")
ART_DIR.mkdir(exist_ok=True)

# Tắt log LGBM
set_config(verbosity=-1)

class IQRClipper(BaseEstimator, TransformerMixin):
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

def load_data():
    # Option A: local CSV from UCI (you can store them in data/)
    red_path = "data/winequality-red.csv"
    white_path = "data/winequality-white.csv"

    if not (os.path.exists(red_path) and os.path.exists(white_path)):
        raise FileNotFoundError(
            "Không thấy data/winequality-red.csv và data/winequality-white.csv. "
            "Hãy tải từ UCI và đặt vào thư mục data/."
        )

    df_red = pd.read_csv(red_path, sep=";")
    df_white = pd.read_csv(white_path, sep=";")
    df_red["wine_type"] = "red"
    df_white["wine_type"] = "white"
    df = pd.concat([df_red, df_white], ignore_index=True)
    return df

def make_preprocess(X: pd.DataFrame):
    num = X.select_dtypes(include=[np.number]).columns.tolist()
    cat = X.select_dtypes(exclude=[np.number]).columns.tolist()

    scaler = StandardScaler() if SCALER_MODE == "standard" else MinMaxScaler()

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("iqr", IQRClipper(factor=IQR_FACTOR)),
        ("scaler", scaler),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", num_pipe, num),
        ("cat", cat_pipe, cat),
    ])

def main():
    df = load_data()
    y = (df["quality"] >= LABEL_THRESHOLD).astype(int)
    X = df.drop(columns=["quality"]).copy()

    pre = make_preprocess(X)

    model = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
        is_unbalance=True,
        verbose=-1,
        verbosity=-1
    )

    pipe = Pipeline([("preprocess", pre), ("clf", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    print(f"Saved model. Test AUC={auc:.4f}")

    joblib.dump(pipe, ART_DIR / "model.joblib")

if __name__ == "__main__":
    main()
