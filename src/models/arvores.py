"""
arvores.py
----------
Treina Decision Tree, Random Forest e Gradient Boosting usando o pipeline
de dados em src/data (load -> preprocess -> feature transform) e salva os
modelos em models/.

Uso:
    python src/models/arvores.py
"""

import json
import logging
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn as mlflow_sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # -> telco_fiap_01/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import build_feature_transformer, load_raw_data, preprocess_data, validate_raw  # noqa: E402

DATA_PATH = PROJECT_ROOT / "data" / "telco_churn_preprocessed.csv"
MODELS    = PROJECT_ROOT / "models"

TARGET_COL = "target"
ID_COL     = "customerID"
SEED       = 42
TEST_SIZE  = 0.15
VAL_SIZE   = 0.15


def load_data():
    """Carrega e pré-processa os dados. Retorna X_raw (DataFrame) e y sem transformar."""
    df = load_raw_data(str(DATA_PATH))
    validate_raw(df)
    df = preprocess_data(df)

    y = df[TARGET_COL].to_numpy(dtype=np.int32)
    X_raw = df.drop(columns=[c for c in (ID_COL, TARGET_COL) if c in df.columns])

    return X_raw, y


def make_split(X, y):
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    val_fraction = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, _, y_train, _ = train_test_split(
        X_trainval, y_trainval,
        test_size=val_fraction, random_state=SEED, stratify=y_trainval,
    )
    return X_train, X_test, y_train, y_test


def compute_metrics(y_true, y_pred, y_proba):
    try:
        auc    = roc_auc_score(y_true, y_proba)
        pr_auc = average_precision_score(y_true, y_proba)
    except ValueError:
        auc    = float("nan")
        pr_auc = float("nan")
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
        "AUC-ROC":  auc,
        "PR-AUC":   pr_auc,
    }


MODELS_CONFIG = {
    "decision_tree": DecisionTreeClassifier(
        max_depth=10,
        class_weight="balanced",
        random_state=SEED,
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    ),
    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        random_state=SEED,
    ),
}


def train_and_save(output_dir: Path = MODELS):
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Carregando dados...")
    X_raw, y = load_data()
    X_train, X_test, y_train, y_test = make_split(X_raw, y)
    logging.info(
        f"Train: {len(X_train):,}  |  Test: {len(X_test):,}  "
        f"|  Taxa de churn (test): {y_test.mean():.4f}"
    )

    results = {}
    for slug, clf in MODELS_CONFIG.items():
        label = slug.replace("_", " ").title()
        logging.info(f"Treinando {label}...")

        with mlflow.start_run(run_name=slug):
            pipeline = Pipeline([
                ("feat", build_feature_transformer()),
                ("clf",  clf),
            ])
            pipeline.fit(X_train, y_train)

            y_pred  = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            results[label] = compute_metrics(y_test, y_pred, y_proba)

            mlflow.log_param("algoritmo", type(clf).__name__)
            if hasattr(clf, "get_params"):
                mlflow.log_params(clf.get_params())
            mlflow.log_metric("Accuracy", float(results[label]["Accuracy"]))
            mlflow.log_metric("F1_Score", float(results[label]["F1-Score"]))
            mlflow.log_metric("ROC_AUC",  float(results[label]["AUC-ROC"]))
            mlflow.log_metric("PR_AUC",   float(results[label]["PR-AUC"]))
            mlflow_sklearn.log_model(pipeline, f"pipeline_{slug}")

            out = output_dir / f"{slug}.joblib"
            joblib.dump(pipeline, out)
            logging.info(f"  Pipeline completo salvo em: {out}")

            metrics_path = output_dir / f"{slug}_metrics.json"
            with open(metrics_path, "w") as fh:
                json.dump({k: float(v) for k, v in results[label].items()}, fh, indent=2)

    cols = ["Accuracy", "F1-Score", "AUC-ROC", "PR-AUC"]
    df = pd.DataFrame(results).T[cols].sort_values("AUC-ROC", ascending=False)

    sep = "=" * 68
    logging.info(f"\n{sep}")
    logging.info("  Modelos de Arvore - Test Set")
    logging.info(sep)
    logging.info("\n" + df.to_string(float_format=lambda x: f"{x:.4f}"))
    logging.info(f"{sep}\n")

    return df


def main():
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("TechChallenge_TelcoChurn")
    train_and_save()


if __name__ == "__main__":
    main()
