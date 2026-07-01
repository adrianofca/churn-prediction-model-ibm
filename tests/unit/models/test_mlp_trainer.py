from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from sklearn.preprocessing import FunctionTransformer
from torch.utils.data import DataLoader

from src.models.mlp import ChurnMLP, MLPConfig, TrainConfig
from src.models.mlp_trainer import (
    compute_metrics,
    load_dataset,
    make_dataloaders,
    run_epoch,
    run_pipeline,
    train,
)


@pytest.fixture
def base_train_config():
    """Configuração de treino reduzida para acelerar os testes unitários."""
    return TrainConfig(
        epochs=2,
        batch_size=4,
        patience=2,
        device="cpu",
        seed=42
    )


@patch("src.models.mlp_trainer.validate_raw")
@patch("src.models.mlp_trainer.preprocess_data")
@patch("src.models.mlp_trainer.load_raw_data")
def test_load_dataset_structure(mock_load, mock_preprocess, mock_validate, base_train_config):
    """Garante que a ingestão separa corretamente as features e remove IDs."""
    mock_df = pd.DataFrame({
        "customerID": ["1", "2", "3", "4"],
        "target": [0.0, 1.0, 0.0, 1.0],
        "feature_1": [10, 20, 30, 40]
    })
    mock_load.return_value = mock_df
    mock_preprocess.return_value = mock_df

    X_raw, y = load_dataset(base_train_config)

    assert "customerID" not in X_raw.columns
    assert "target" not in X_raw.columns
    assert list(X_raw.columns) == ["feature_1"]
    assert np.array_equal(y, np.array([0, 1, 0, 1], dtype=np.float32))


@patch("src.models.mlp_trainer.joblib.dump")
@patch("src.models.mlp_trainer.build_feature_transformer")
def test_make_dataloaders_splits(mock_build_feat, mock_joblib_dump, base_train_config):
    """Garante a geração correta dos 3 DataLoaders estruturados do PyTorch."""
    # 20 instâncias simuladas para garantir amostras em todos os splits (70/15/15)
    X_raw = pd.DataFrame({"feat": range(20)})
    y = np.array([0, 1] * 10, dtype=np.float32)

    # Dublê pass-through para o transformer não alterar o shape original
    mock_build_feat.return_value = FunctionTransformer()

    train_loader, val_loader, test_loader = make_dataloaders(X_raw, y, base_train_config)

    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)
    assert isinstance(test_loader, DataLoader)

    # Valida se os dados de teste e validação mantêm os registros íntegros
    # Split de 15% de 20 = 3 registros em cada
    assert len(val_loader.dataset) == 3
    assert len(test_loader.dataset) == 3


def test_compute_metrics_math_logic():
    """Valida a precisão matemática das métricas geradas a partir de logits puros."""
    logits = torch.tensor([2.0, -2.0, 2.0, -2.0])  # Sigmoids altos (~0.88) e baixos (~0.11)
    targets = torch.tensor([1.0, 0.0, 0.0, 1.0])    # 2 acertos, 2 erros

    metrics = compute_metrics(logits, targets, threshold=0.5)

    assert metrics["accuracy"] == 0.5
    assert "f1" in metrics
    assert metrics["auc"] == 0.5


def test_train_invalid_monitor_exception(base_train_config):
    """Garante que passar um monitor de Early Stopping inválido dispara ValueError."""
    model = ChurnMLP(MLPConfig(input_dim=5))
    config_errada = base_train_config
    config_errada.monitor = "invalid_metric"

    with pytest.raises(ValueError) as exc_info:
        train(model, MagicMock(), MagicMock(), MagicMock(), config_errada)

    assert "monitor deve ser" in str(exc_info.value)


def test_run_epoch_evaluation_mode():
    """Garante que a execução em modo avaliação não altera os pesos do modelo."""
    model = ChurnMLP(MLPConfig(input_dim=2))
    loss_fn = nn.BCEWithLogitsLoss()

    # Dataset contendo 1 único batch simulado
    X_batch = torch.randn(4, 2)
    y_batch = torch.tensor([0.0, 1.0, 0.0, 1.0])
    loader = [(X_batch, y_batch)]

    # Coleta o estado dos pesos antes de rodar a época
    pesos_iniciais = model.output.weight.clone().detach()

    # Executa em modo de avaliação (sem otimizador)
    metrics = run_epoch(model, loader, loss_fn, device="cpu", optimizer=None)

    assert "loss" in metrics
    # Garante imutabilidade dos parâmetros na avaliação
    assert torch.equal(model.output.weight, pesos_iniciais)


def test_train_early_stopping_trigger(base_train_config):
    """Garante que o loop interrompe o treino caso a validação pare de evoluir."""
    model = ChurnMLP(MLPConfig(input_dim=2))
    loss_fn = nn.BCEWithLogitsLoss()

    X_batch = torch.randn(4, 2)
    y_batch = torch.tensor([0.0, 1.0, 0.0, 1.0])
    loader = [(X_batch, y_batch)]

    # Forçamos o número máximo de épocas para 5, mas patience=2
    base_train_config.epochs = 5
    base_train_config.patience = 2
    base_train_config.monitor = "val_loss"

    # Criamos um mock estável para run_epoch retornar sempre a mesma loss fixa (sem melhoria)
    with patch("src.models.mlp_trainer.run_epoch") as mock_run_epoch:
        mock_run_epoch.return_value = {"loss": 0.50, "accuracy": 1.0, "f1": 1.0, "auc": 1.0, "pr_auc": 1.0}

        history = train(model, loader, loader, loss_fn, base_train_config)

        # Deve interromper o fluxo na época 3 (Época 1: base, Época 2: wait=1, Época 3: wait=2 -> Break)
        assert history["early_stopped"] is True
        assert history["best_epoch"] == 1


@patch("src.models.mlp_trainer.load_dataset")
@patch("src.models.mlp_trainer.make_dataloaders")
@patch("src.models.mlp_trainer.train")
@patch("src.models.mlp_trainer.evaluate")
@patch("src.models.mlp_trainer.save_model")
@patch("src.models.mlp_trainer.mlflow")
def test_run_pipeline_orchestration(mock_mlflow, mock_save, mock_eval, mock_train, mock_make_loaders, mock_load_ds):
    """Garante que a função mestre coordena todas as subetapas e registra no MLflow."""
    # Arrange
    X_mock = pd.DataFrame({"f": range(10)})
    y_mock = np.array([0, 1] * 5)
    mock_load_ds.return_value = (X_mock, y_mock)

    # Simulação dos loaders retornando batches rápidos de tensores
    ds_mock = [(torch.randn(2, 4), torch.tensor([0.0, 1.0]))]
    mock_make_loaders.return_value = (ds_mock, ds_mock, ds_mock)

    mock_train.return_value = {
        "train_loss": [0.6], "val_loss": [0.5], "val_f1": [0.7], "val_auc": [0.7], "val_pr_auc": [0.7],
        "best_epoch": 1, "best_value": 0.5, "early_stopped": False
    }
    mock_eval.return_value = {"loss": 0.4, "accuracy": 0.8, "f1": 0.8, "auc": 0.8, "pr_auc": 0.8}
    mock_mlflow.start_run.return_value.__enter__.return_value = MagicMock()

    # Act
    pipeline_results = run_pipeline(MLPConfig(), TrainConfig(epochs=1, device="cpu"))

    # Assert
    assert "model" in pipeline_results
    assert "test_metrics" in pipeline_results

    # Confirma se o modelo campeão e seus metadados foram despachados para salvamento
    mock_save.assert_called_once()
    # run_pipeline chama log_params duas vezes: uma para MLPConfig e outra para TrainConfig
    assert mock_mlflow.log_params.call_count == 2
