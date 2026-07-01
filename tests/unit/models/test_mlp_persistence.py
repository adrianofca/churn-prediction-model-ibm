from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from src.models.mlp import ChurnMLP, MLPConfig, TrainConfig
from src.models.mlp_persistence import load_model, save_model


@pytest.fixture
def dummy_model_components():
    """Fornece instâncias simplificadas das configurações e um modelo mínimo para persistência."""
    mlp_cfg = MLPConfig(input_dim=10, hidden_dims=[16], use_batch_norm=False)
    train_cfg = TrainConfig(epochs=1, monitor="val_loss")
    model = ChurnMLP(mlp_cfg)

    history = {"best_epoch": 1, "best_value": 0.456}
    test_metrics = {"loss": 0.4, "accuracy": 0.85, "f1": 0.80, "auc": 0.90, "pr_auc": 0.88}

    return model, mlp_cfg, train_cfg, history, test_metrics


@patch("src.models.mlp_persistence.Path.mkdir")
@patch("src.models.mlp_persistence.torch.save")
@patch("src.models.mlp_persistence.mlflow")
@patch("builtins.open", new_callable=mock_open)
def test_save_model_with_active_mlflow(mock_file, mock_mlflow, mock_torch_save, mock_mkdir, dummy_model_components):
    """Garante o empacotamento correto do checkpoint e o envio dos artefatos ao MLflow ativo."""
    model, mlp_cfg, train_cfg, history, test_metrics = dummy_model_components

    mock_mlflow.active_run.return_value = MagicMock()

    # Act
    saved_path = save_model(model, mlp_cfg, train_cfg, history, test_metrics, filename="test_mlp.pt")

    # Assert
    assert isinstance(saved_path, Path)
    assert saved_path.name == "test_mlp.pt"

    mock_torch_save.assert_called_once()
    checkpoint_argument = mock_torch_save.call_args[0][0]
    assert "model_state_dict" in checkpoint_argument
    assert checkpoint_argument["best_epoch"] == 1
    assert checkpoint_argument["monitor"] == "val_loss"
    assert checkpoint_argument["mlp_config"]["input_dim"] == 10

    mock_file.assert_called_once()
    assert mock_mlflow.log_artifact.call_count == 2


@patch("src.models.mlp_persistence.Path.exists")
@patch("src.models.mlp_persistence.torch.load")
def test_load_model_success(mock_torch_load, mock_exists):
    """Garante que a leitura reconstitui a arquitetura da rede e injeta os pesos em modo eval."""
    mock_exists.return_value = True

    # GERADOR DE PESOS REAIS: Instancia uma config e extrai um state_dict válido com as chaves que o PyTorch exige
    from src.models.mlp import ChurnMLP, MLPConfig
    cfg_duble = MLPConfig(input_dim=15, hidden_dims=[32], use_batch_norm=False)
    state_dict_valido = ChurnMLP(cfg_duble).state_dict()

    mock_checkpoint = {
        "mlp_config": {
            "input_dim": 15,
            "hidden_dims": [32],
            "dropout_rate": 0.1,
            "use_batch_norm": False,
            "activation": "relu",
            "pos_weight": 1.0
        },
        "model_state_dict": state_dict_valido,  # Corrigido: Agora mapeado com chaves estruturadas reais
        "best_epoch": 5,
        "monitor": "val_loss",
        "best_value": 0.35,
        "test_metrics": {"loss": 0.3, "accuracy": 0.9, "f1": 0.9, "auc": 0.9, "pr_auc": 0.9}
    }
    mock_torch_load.return_value = mock_checkpoint

    # Act
    loaded_model = load_model("test_mlp.pt")

    # Assert
    assert isinstance(loaded_model, ChurnMLP)
    assert loaded_model.config.input_dim == 15
    assert loaded_model.config.hidden_dims == [32]
    assert loaded_model.training is False


@patch("src.models.mlp_persistence.Path.exists")
def test_load_model_file_not_found(mock_exists):
    """Garante que a tentativa de carga de um arquivo inexistente dispara FileNotFoundError de forma limpa."""
    mock_exists.return_value = False

    with pytest.raises(FileNotFoundError) as exc_info:
        load_model("ghost_model.pt")

    assert "Checkpoint não encontrado em:" in str(exc_info.value)
