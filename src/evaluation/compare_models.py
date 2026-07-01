"""
compare_models.py
-----------------
Carrega as métricas salvas em models/ e gera uma tabela comparativa.

Cada script de treino salva um arquivo <modelo>_metrics.json com as
chaves: Accuracy, F1-Score, AUC-ROC, PR-AUC.

Uso:
    python src/evaluation/compare_models.py
"""

import json
import logging
from pathlib import Path

import mlflow
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS       = PROJECT_ROOT / "models"
REPORTS_DIR  = PROJECT_ROOT / "reports"

REGISTRY = [
    ("baseline_dummy_model_metrics.json",         "Dummy Baseline"),
    ("baseline_logistic_regression_metrics.json",  "Logistic Regression"),
    ("decision_tree_metrics.json",                 "Decision Tree"),
    ("random_forest_metrics.json",                 "Random Forest"),
    ("gradient_boosting_metrics.json",             "Gradient Boosting"),
    ("mlp_model_metrics.json",                     "MLP (PyTorch)"),
]


def main():
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("TechChallenge_TelcoChurn")

    results = {}
    for fname, label in REGISTRY:
        path = MODELS / fname
        if path.exists():
            with open(path) as fh:
                results[label] = json.load(fh)
        else:
            logging.warning(f"métricas não encontradas para {label} ({fname})")

    if not results:
        logging.warning("Nenhuma métrica encontrada. Execute os scripts de treino primeiro.")
        return

    cols = ["Accuracy", "F1-Score", "AUC-ROC", "PR-AUC"]
    df = (
        pd.DataFrame(results)
        .T[cols]
        .sort_values("AUC-ROC", ascending=False)
    )

    sep = "=" * 68
    logging.info(f"\n{sep}")
    logging.info("  Comparacao de Modelos - Test Set")
    logging.info(sep)
    logging.info("\n" + df.to_string(float_format=lambda x: f"{x:.4f}"))
    logging.info(f"{sep}\n")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORTS_DIR / "compare_models.csv"
    df.to_csv(csv_path)

    with mlflow.start_run(run_name="compare_models"):
        for label, row in df.iterrows():
            safe = label.replace(" ", "_").replace("(", "").replace(")", "")
            for metric, value in row.items():
                safe_metric = metric.replace("-", "_")
                mlflow.log_metric(f"{safe}__{safe_metric}", float(value))
        mlflow.log_artifact(str(csv_path))

    return df


if __name__ == "__main__":
    main()
