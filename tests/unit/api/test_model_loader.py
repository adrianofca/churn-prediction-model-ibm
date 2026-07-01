import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch


@pytest.fixture(name="loader_env")
def fixture_loader_env():
    """
    Fixture de isolamento global. Intercepta o carregamento do disco
    antes de importar o módulo da API, evitando caminhos hardcoded reais.
    """
    with patch("src.models.mlp_persistence.load_model") as mock_load, \
         patch("joblib.load") as mock_joblib:

        # 1. Configura o comportamento do dublê do modelo PyTorch
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = torch.tensor([0.75])
        mock_load.return_value = mock_model

        # 2. Configura o comportamento do dublê do Transformer (Sklearn)
        mock_transformer = MagicMock()
        mock_ohe = MagicMock()

        # Cria categorias mockadas com o atributo .dtype exigido pelo bloco de DEBUG
        mock_cats = np.array(["No", "Yes"], dtype=object)
        mock_ohe.categories_ = [mock_cats]

        mock_transformer.named_steps = {
            "column_transformer": MagicMock(
                named_transformers_={"cat": mock_ohe}
            )
        }
        # Retorna uma matriz bidimensional genérica simulando as features pós-OHE
        mock_transformer.transform.return_value = np.array([[1.0, 0.0, 24.0, 75.0]])
        mock_joblib.return_value = mock_transformer

        # Remove o módulo do cache do Python para forçar o escopo global a
        # reexecutar usando os nossos patches novos em cada teste
        if "api.model_loader" in sys.modules:
            del sys.modules["api.model_loader"]

        import api.model_loader

        yield api.model_loader, mock_model, mock_transformer


def test_predict_churn_high_probability(loader_env):
    """Garante a classificação positiva (1) quando a probabilidade for maior ou igual a 0.5."""
    module, mock_model, _ = loader_env
    mock_model.predict_proba.return_value = torch.tensor([0.82])

    input_payload = {"SeniorCitizen": 0, "MonthlyCharges": 85.5, "tenure": 12}

    prob, pred = module.predict(input_payload)

    assert prob == pytest.approx(0.82)
    assert pred == 1


def test_predict_no_churn_low_probability(loader_env):
    """Garante a classificação negativa (0) quando a probabilidade for menor que o threshold (0.17)."""
    module, mock_model, _ = loader_env
    mock_model.predict_proba.return_value = torch.tensor([0.10])

    input_payload = {"SeniorCitizen": 1, "MonthlyCharges": 20.0, "tenure": 72}

    prob, pred = module.predict(input_payload)

    assert prob == pytest.approx(0.10)
    assert pred == 0


def test_predict_forces_senior_citizen_to_string(loader_env):
    """Verifica se a regra de tipo do schema força SeniorCitizen a virar string para o OHE."""
    module, _, mock_transformer = loader_env

    # Payload passando SeniorCitizen como um número inteiro puro
    input_payload = {"SeniorCitizen": 1, "MonthlyCharges": 50.0}

    _, _ = module.predict(input_payload)

    # Intercepta o argumento passado para o transformer.transform()
    # e garante que ele recebeu a coluna convertida de forma resiliente
    called_df = mock_transformer.transform.call_args[0][0]
    assert called_df["SeniorCitizen"].dtype == object or called_df["SeniorCitizen"].dtype == str
    assert called_df["SeniorCitizen"].iloc[0] == "1"
