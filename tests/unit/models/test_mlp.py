import pytest
import torch
import torch.nn as nn

from src.models.mlp import ChurnMLP, HiddenBlock, MLPConfig, TrainConfig, build_loss


def test_configs_serialization():
    """Garante que os dicionários de configuração exportam os tipos corretos para o MLflow."""
    mlp_cfg = MLPConfig(input_dim=45, activation="leaky_relu")
    train_cfg = TrainConfig(batch_size=32)

    mlp_dict = mlp_cfg.to_dict()
    train_dict = train_cfg.to_dict()

    assert mlp_dict["input_dim"] == 45
    assert mlp_dict["activation"] == "leaky_relu"
    assert "input_dim" not in train_dict
    assert train_dict["batch_size"] == 32


def test_hidden_block_invalid_activation():
    """Garante que passar uma string de ativação inválida levanta uma exceção ValueError."""
    with pytest.raises(ValueError) as exc_info:
        HiddenBlock(
            in_features=10, out_features=5, dropout_rate=0.2, use_batch_norm=True, activation="invalid_activation"
        )

    assert "Ativação 'invalid_activation' inválida" in str(exc_info.value)


@pytest.mark.parametrize("use_bn", [True, False])
@pytest.mark.parametrize("act", ["relu", "leaky_relu", "elu"])
def test_hidden_block_forward_shapes(use_bn, act):
    """Garante que o bloco oculto processa tensores mantendo a saída esperada em múltiplos schemas."""
    batch_size = 16
    in_features = 20
    out_features = 10

    block = HiddenBlock(
        in_features=in_features,
        out_features=out_features,
        dropout_rate=0.1,
        use_batch_norm=use_bn,
        activation=act
    )

    # Gera um tensor aleatório simulando um lote de entrada
    x_input = torch.randn(batch_size, in_features)
    x_output = block(x_input)

    # A dimensão da saída deve ser obrigatoriamente (batch_size, out_features)
    assert x_output.shape == (batch_size, out_features)


def test_churn_mlp_forward_pass():
    """Garante que a rede processa os inputs e escoa em um vetor unidimensional de logits."""
    batch_size = 8
    config = MLPConfig(input_dim=30, hidden_dims=[64, 32])
    model = ChurnMLP(config)

    x_input = torch.randn(batch_size, 30)
    logits = model(x_input)

    # A saída deve ser um vetor plano contendo 1 logit por registro no lote
    assert logits.dim() == 1
    assert logits.shape[0] == batch_size


def test_churn_mlp_predict_proba_state_preservation():
    """Garante que predict_proba calcula probabilidades seguras e não corrompe o estado de treino."""
    config = MLPConfig(input_dim=20)
    model = ChurnMLP(config)

    # 1. Forçamos o modelo a entrar em modo de treinamento explicitamente
    model.train()
    assert model.training is True

    x_input = torch.randn(4, 20)
    probas = model.predict_proba(x_input)

    # As probabilidades do Sigmoid devem estar estritamente no intervalo [0, 1]
    assert torch.all(probas >= 0.0)
    assert torch.all(probas <= 1.0)

    # CRÍTICO: Garante que a função restaurou o modo de treinamento da rede com sucesso
    assert model.training is True


def test_churn_mlp_predict_binary_labels():
    """Garante que a função predict converte probabilidades em labels binários inteiros (0/1)."""
    config = MLPConfig(input_dim=10)
    model = ChurnMLP(config)

    x_input = torch.randn(5, 10)
    predictions = model.predict(x_input, threshold=0.5)

    assert predictions.shape[0] == 5
    assert predictions.dtype == torch.long
    # Elementos mapeados devem ser estritamente 0 ou 1
    assert set(predictions.tolist()).issubset({0, 1})


def test_build_loss_device_allocation():
    """Garante a instanciação correta da função de perda balanceada."""
    config = MLPConfig(pos_weight=3.5)
    loss_fn = build_loss(config, device="cpu")

    assert isinstance(loss_fn, nn.BCEWithLogitsLoss)
    # Garante que o pos_weight foi injetado internamente no tensor da Loss
    assert loss_fn.pos_weight.item() == 3.5
