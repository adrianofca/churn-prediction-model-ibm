"""
mlp_persistence.py
------------------
Funções para salvar e carregar checkpoints do ChurnMLP.

O checkpoint (.pt) inclui:
  - model_state_dict : pesos do modelo
  - mlp_config       : arquitetura (MLPConfig como dict)
  - train_config     : hiperparâmetros de treino (TrainConfig como dict)
  - best_epoch / best_value / monitor : resumo do early stopping
  - test_metrics     : métricas do conjunto de teste

Um arquivo _metrics.json paralelo é salvo para comparação rápida
com os outros modelos via compare_models.py.
"""

import dataclasses
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import mlflow
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if TYPE_CHECKING:
    from .mlp import ChurnMLP, MLPConfig, TrainConfig


def save_model(
    model:        "ChurnMLP",
    mlp_cfg:      "MLPConfig",
    train_cfg:    "TrainConfig",
    history:      dict,
    test_metrics: dict,
    filename:     str = "mlp_model.pt",
) -> Path:
    """
    Salva o melhor modelo em <project_root>/models/<filename>.

    Também grava <filename_stem>_metrics.json para comparação com
    os modelos sklearn via compare_models.py.
    """
    models_dir = Path(__file__).resolve().parent.parent.parent / "models"
    models_dir.mkdir(exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "mlp_config":       dataclasses.asdict(mlp_cfg),
        "train_config":     dataclasses.asdict(train_cfg),
        "best_epoch":       history["best_epoch"],
        "best_value":       history["best_value"],
        "monitor":          train_cfg.monitor,
        "test_metrics":     test_metrics,
    }

    dest = models_dir / filename
    torch.save(checkpoint, dest)
    logging.info(f"Modelo salvo em: {dest}")

    metrics_path = models_dir / (Path(filename).stem + "_metrics.json")
    normalized = {
        "Accuracy": float(test_metrics.get("accuracy", float("nan"))),
        "F1-Score": float(test_metrics.get("f1",       float("nan"))),
        "AUC-ROC":  float(test_metrics.get("auc",      float("nan"))),
        "PR-AUC":   float(test_metrics.get("pr_auc",   float("nan"))),
    }
    with open(metrics_path, "w") as fh:
        json.dump(normalized, fh, indent=2)

    if mlflow.active_run() is not None:
        mlflow.log_artifact(str(dest))
        mlflow.log_artifact(str(metrics_path))

    return dest


def load_model(filename: str = "mlp_model.pt") -> "ChurnMLP":
    """
    Carrega o modelo salvo por save_model e retorna um ChurnMLP em modo eval.

    Uso:
        model = load_model()
        proba = model.predict_proba(x_tensor)
    """
    from .mlp import ChurnMLP, MLPConfig

    dest = Path(__file__).resolve().parent.parent.parent / "models" / filename
    if not dest.exists():
        raise FileNotFoundError(f"Checkpoint não encontrado em: {dest}")

    checkpoint = torch.load(dest, map_location="cpu", weights_only=False)

    mlp_cfg_data = checkpoint["mlp_config"]
    config = MLPConfig(**mlp_cfg_data) if isinstance(mlp_cfg_data, dict) else mlp_cfg_data

    model = ChurnMLP(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    logging.info(f"Modelo carregado de: {dest}")
    logging.info(
        f"Melhor época : {checkpoint['best_epoch']}  "
        f"({checkpoint['monitor']}={checkpoint['best_value']:.4f})"
    )
    if "test_metrics" in checkpoint:
        m = checkpoint["test_metrics"]
        logging.info(
            f"Test metrics : loss={m['loss']:.4f}  acc={m['accuracy']:.4f}  "
            f"f1={m['f1']:.4f}  auc={m['auc']:.4f}  pr_auc={m['pr_auc']:.4f}"
        )

    return model
