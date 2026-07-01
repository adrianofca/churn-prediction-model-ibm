"""
mlp_trainer.py
--------------
Pipeline de dados, training loop e avaliação do ChurnMLP.

Funções exportadas:
  - load_dataset      : carrega e transforma os dados brutos
  - make_dataloaders  : cria DataLoaders com split estratificado 70/15/15
  - compute_metrics   : calcula accuracy, F1, AUC-ROC e PR-AUC
  - run_epoch         : roda uma época (treino ou avaliação)
  - train             : training loop com early stopping
  - evaluate          : avalia modelo em um DataLoader
  - run_pipeline      : pipeline end-to-end
"""

import copy
import dataclasses
import logging
import sys
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from data import build_feature_transformer, load_raw_data, preprocess_data, validate_raw  # noqa: E402

from .mlp import ChurnMLP, MLPConfig, TrainConfig, build_loss
from .mlp_persistence import save_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ── Carregamento e split do dataset ────────────────────────────────────────────

def load_dataset(train_cfg: TrainConfig) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Carrega e pré-processa os dados brutos. Retorna X_raw (DataFrame) e y.
    A transformação de features ocorre APÓS o split em make_dataloaders,
    garantindo que o scaler seja fitado apenas nos dados de treino.
    """
    df = load_raw_data(train_cfg.data_path)
    validate_raw(df)
    df = preprocess_data(df)

    y = np.asarray(df[train_cfg.target_col], dtype=np.float32)
    drop_cols = [c for c in (train_cfg.id_col, train_cfg.target_col) if c in df.columns]
    X_raw = df.drop(columns=drop_cols)

    logging.info("load_dataset: ingestão e pré-processamento concluídos")
    logging.info(f"  Fonte         : {train_cfg.data_path}")
    logging.info(f"  Linhas        : {len(df):,}")
    logging.info(f"  Target ratio  : {y.mean():.4f} (positivos)")
    logging.info(f"  Class balance : {int((y==0).sum())} neg / {int((y==1).sum())} pos")

    return X_raw, y


def make_dataloaders(
    X_raw: pd.DataFrame,
    y: np.ndarray,
    train_cfg: TrainConfig,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Divide os dados brutos, fita o transformer apenas no treino e cria DataLoaders.

    Split: 70% train / 15% val / 15% test (estratificado por y, seed fixa).
    O transformer é fitado SOMENTE em X_train — val e test apenas transformados.
    """
    X_trainval_raw, X_test_raw, y_trainval, y_test = train_test_split(
        X_raw, y,
        test_size=train_cfg.test_size,
        random_state=train_cfg.seed,
        stratify=y,
    )

    val_fraction = train_cfg.val_size / (1.0 - train_cfg.test_size)
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_trainval_raw, y_trainval,
        test_size=val_fraction,
        random_state=train_cfg.seed,
        stratify=y_trainval,
    )

    transformer = build_feature_transformer()
    X_train = np.asarray(transformer.fit_transform(X_train_raw), dtype=np.float32)
    X_val   = np.asarray(transformer.transform(X_val_raw),       dtype=np.float32)
    X_test  = np.asarray(transformer.transform(X_test_raw),      dtype=np.float32)
    # salva o transformer já treinado
    joblib.dump(
        transformer,
    "models/transformer.pkl"
    )

    logging.info("Transformer salvo em models/transformer.pkl")

    logging.info(f"make_dataloaders: splits estratificados (seed={train_cfg.seed})")
    logging.info(f"  Train : {len(X_train):>5} amostras  ({y_train.mean():.4f} pos)  |  {X_train.shape[1]} features")
    logging.info(f"  Val   : {len(X_val):>5} amostras  ({y_val.mean():.4f} pos)")
    logging.info(f"  Test  : {len(X_test):>5} amostras  ({y_test.mean():.4f} pos)")

    def to_loader(X_arr, y_arr, shuffle, drop_last=False):
        ds = TensorDataset(
            torch.from_numpy(X_arr).float(),
            torch.from_numpy(y_arr).float(),
        )
        return DataLoader(ds, batch_size=train_cfg.batch_size, shuffle=shuffle, drop_last=drop_last)

    return (
        to_loader(X_train, y_train, shuffle=True,  drop_last=True),
        to_loader(X_val,   y_val,   shuffle=False),
        to_loader(X_test,  y_test,  shuffle=False),
    )


# ── Métricas ───────────────────────────────────────────────────────────────────

def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> dict:
    """Computa accuracy, F1, AUC-ROC e PR-AUC a partir de logits e rótulos verdadeiros."""
    proba  = torch.sigmoid(logits).detach().cpu().numpy()
    preds  = (proba >= threshold).astype(int)
    y_true = targets.detach().cpu().numpy().astype(int)

    acc = accuracy_score(y_true, preds)
    f1  = f1_score(y_true, preds, zero_division=0)

    try:
        auc    = roc_auc_score(y_true, proba)
        pr_auc = average_precision_score(y_true, proba)
    except ValueError:
        auc    = float("nan")
        pr_auc = float("nan")

    return {"accuracy": acc, "f1": f1, "auc": auc, "pr_auc": pr_auc}


# ── Loops de treino e avaliação ────────────────────────────────────────────────

def run_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    loss_fn:   nn.Module,
    device:    str,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    """
    Roda uma época em modo treino (optimizer != None) ou avaliação.
    Retorna dict com loss, accuracy, f1, auc e pr_auc agregados.
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss  = 0.0
    total_n     = 0
    all_logits  = []
    all_targets = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            logits = model(xb)
            loss   = loss_fn(logits, yb)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * xb.size(0)
            total_n    += xb.size(0)
            all_logits.append(logits.detach())
            all_targets.append(yb.detach())

    avg_loss = total_loss / total_n
    metrics  = compute_metrics(torch.cat(all_logits), torch.cat(all_targets))
    return {"loss": avg_loss, **metrics}


def train(
    model:        ChurnMLP,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    loss_fn:      nn.Module,
    train_cfg:    TrainConfig,
) -> dict:
    """
    Training loop com tracking de métricas por época e early stopping.

    Monitora `train_cfg.monitor` e para após `train_cfg.patience` épocas sem
    melhora de `train_cfg.min_delta`. Restaura os pesos da melhor época.
    Defina patience=0 para desativar o early stopping.
    """
    if train_cfg.monitor not in ("val_loss", "val_auc", "val_f1", "val_pr_auc"):
        raise ValueError(
            f"monitor deve ser 'val_loss', 'val_auc', 'val_f1' ou 'val_pr_auc', "
            f"recebeu '{train_cfg.monitor}'"
        )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )

    history = {
        "train_loss": [], "train_acc": [], "train_f1": [], "train_auc": [], "train_pr_auc": [],
        "val_loss":   [], "val_acc":   [], "val_f1":   [], "val_auc":   [], "val_pr_auc":   [],
    }

    maximize     = train_cfg.monitor in ("val_auc", "val_f1", "val_pr_auc")
    best_value   = float("-inf") if maximize else float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch   = 1
    wait         = 0
    early_stopped = False
    epoch        = 0

    es_info = (
        f"patience={train_cfg.patience}, min_delta={train_cfg.min_delta}, "
        f"monitor={train_cfg.monitor}"
        if train_cfg.patience > 0
        else "desativado"
    )

    logging.info(f"\n{'─'*90}")
    logging.info(
        f"  Training — max {train_cfg.epochs} épocas, batch={train_cfg.batch_size}, "
        f"lr={train_cfg.learning_rate}, device={train_cfg.device}"
    )
    logging.info(f"  Early stopping: {es_info}")
    logging.info(f"{'─'*90}")
    logging.info(
        f"  {'epoch':>5} | {'tr_loss':>8} {'tr_acc':>7} {'tr_f1':>7} {'tr_auc':>7} {'tr_prauc':>8}"
        f" | {'vl_loss':>8} {'vl_acc':>7} {'vl_f1':>7} {'vl_auc':>7} {'vl_prauc':>8}"
    )
    logging.info(f"{'─'*90}")

    for epoch in range(1, train_cfg.epochs + 1):
        tr = run_epoch(model, train_loader, loss_fn, train_cfg.device, optimizer)
        vl = run_epoch(model, val_loader,   loss_fn, train_cfg.device)

        history["train_loss"].append(tr["loss"])
        history["train_acc"].append(tr["accuracy"])
        history["train_f1"].append(tr["f1"])
        history["train_auc"].append(tr["auc"])
        history["train_pr_auc"].append(tr["pr_auc"])
        history["val_loss"].append(vl["loss"])
        history["val_acc"].append(vl["accuracy"])
        history["val_f1"].append(vl["f1"])
        history["val_auc"].append(vl["auc"])
        history["val_pr_auc"].append(vl["pr_auc"])

        current = {
            "val_loss":   vl["loss"],
            "val_auc":    vl["auc"],
            "val_f1":     vl["f1"],
            "val_pr_auc": vl["pr_auc"],
        }[train_cfg.monitor]

        improved = (current > best_value + train_cfg.min_delta) if maximize \
                   else (current < best_value - train_cfg.min_delta)

        marker = ""
        if improved:
            best_value   = current
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch   = epoch
            wait         = 0
            marker       = " *"
        elif train_cfg.patience > 0:
            wait += 1

        logging.info(
            f"  {epoch:>5} | {tr['loss']:>8.4f} {tr['accuracy']:>7.4f}"
            f" {tr['f1']:>7.4f} {tr['auc']:>7.4f} {tr['pr_auc']:>8.4f}"
            f" | {vl['loss']:>8.4f} {vl['accuracy']:>7.4f}"
            f" {vl['f1']:>7.4f} {vl['auc']:>7.4f} {vl['pr_auc']:>8.4f}{marker}"
        )

        if train_cfg.patience > 0 and wait >= train_cfg.patience:
            early_stopped = True
            break

    model.load_state_dict(best_weights)

    logging.info(f"{'─'*90}")
    if early_stopped:
        logging.info(
            f"  Early stopping na época {epoch} — melhor época: {best_epoch} "
            f"({train_cfg.monitor}={best_value:.4f})"
        )
    else:
        logging.info(
            f"  Treino concluído — melhor época: {best_epoch} "
            f"({train_cfg.monitor}={best_value:.4f})"
        )
    logging.info(f"{'─'*90}\n")

    return {
        **history,
        "best_epoch":    best_epoch,
        "best_value":    best_value,
        "early_stopped": early_stopped,
    }


def evaluate(
    model:   ChurnMLP,
    loader:  DataLoader,
    loss_fn: nn.Module,
    device:  str,
    name:    str = "test",
) -> dict:
    """Avalia modelo em um DataLoader e imprime métricas."""
    metrics = run_epoch(model, loader, loss_fn, device)
    logging.info(
        f"  [{name}] loss={metrics['loss']:.4f}  acc={metrics['accuracy']:.4f}  "
        f"f1={metrics['f1']:.4f}  auc={metrics['auc']:.4f}  pr_auc={metrics['pr_auc']:.4f}"
    )
    return metrics


# ── Pipeline completo ──────────────────────────────────────────────────────────

def run_pipeline(
    mlp_cfg:   MLPConfig | None   = None,
    train_cfg: TrainConfig | None = None,
) -> dict:
    """
    Pipeline end-to-end: carrega dados, treina e avalia no test set.

    Retorna dict com model, history e test_metrics.
    """
    mlp_cfg   = mlp_cfg   or MLPConfig()
    train_cfg = train_cfg or TrainConfig()

    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("TechChallenge_TelcoChurn")

    with mlflow.start_run(run_name="MLP_ChurnMLP"):
        torch.manual_seed(train_cfg.seed)
        np.random.seed(train_cfg.seed)

        X_raw, y = load_dataset(train_cfg)

        mlp_cfg = dataclasses.replace(
            mlp_cfg,
            pos_weight=float((y == 0).sum() / (y == 1).sum()),
        )

        train_loader, val_loader, test_loader = make_dataloaders(X_raw, y, train_cfg)

        # input_dim obtido após transformação (número real de features geradas)
        input_dim = next(iter(train_loader))[0].shape[1]
        mlp_cfg = dataclasses.replace(mlp_cfg, input_dim=input_dim)

        model   = ChurnMLP(mlp_cfg).to(train_cfg.device)
        loss_fn = build_loss(mlp_cfg, device=train_cfg.device)
        model.summary()

        mlflow.log_params(mlp_cfg.to_dict())
        mlflow.log_params(train_cfg.to_dict())

        history = train(model, train_loader, val_loader, loss_fn, train_cfg)

        for i, (tr_loss, vl_loss, vl_f1, vl_auc, vl_pr_auc) in enumerate(
            zip(
                history["train_loss"], history["val_loss"],
                history["val_f1"], history["val_auc"], history["val_pr_auc"],
            ),
            start=1,
        ):
            mlflow.log_metric("train_loss",  tr_loss,   step=i)
            mlflow.log_metric("val_loss",    vl_loss,   step=i)
            mlflow.log_metric("val_f1",      vl_f1,     step=i)
            mlflow.log_metric("val_auc",     vl_auc,    step=i)
            mlflow.log_metric("val_pr_auc",  vl_pr_auc, step=i)

        mlflow.log_metric("best_epoch", history["best_epoch"])

        logging.info("Avaliação final:")
        test_metrics = evaluate(model, test_loader, loss_fn, train_cfg.device, name="test")

        mlflow.log_metric("test_F1_Score", float(test_metrics["f1"]))
        mlflow.log_metric("test_ROC_AUC",  float(test_metrics["auc"]))
        mlflow.log_metric("test_PR_AUC",   float(test_metrics["pr_auc"]))
        mlflow.log_metric("test_Accuracy", float(test_metrics["accuracy"]))

        save_model(model, mlp_cfg, train_cfg, history, test_metrics)

        return {"model": model, "history": history, "test_metrics": test_metrics}


if __name__ == "__main__":
    run_pipeline()
