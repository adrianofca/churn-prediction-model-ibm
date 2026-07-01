import sys
import types
from unittest.mock import MagicMock

import pydantic
import pytest
from fastapi.testclient import TestClient


# ── MODELOS DUMMY PARA ISOLAR O SCHEMAS.PY DO SEU COLEGA ────────────────────────
class DummyPredictRequest(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")

class DummyPredictResponse(pydantic.BaseModel):
    probability: float
    prediction: int

class DummyMiddleware:
    def __init__(self, app):
        self.app = app
    async def __call__(self, scope, receive, send):
        await self.app(scope, receive, send)


@pytest.fixture(name="api_env")
def fixture_api_env():
    """
    Injeta dublês direto no sys.modules para isolar o main.py de
    dependências físicas e evitar erros de resolução de pacotes.
    """
    mock_logger = MagicMock()
    mock_predict = MagicMock()

    # Guarda uma cópia do estado original para limpar após o teste
    original_modules = dict(sys.modules)

    # 1. Injeta dublê para api.schemas
    mock_schemas = types.ModuleType("api.schemas")
    mock_schemas.PredictRequest = DummyPredictRequest
    mock_schemas.PredictResponse = DummyPredictResponse
    sys.modules["api.schemas"] = mock_schemas

    # 2. Injeta dublê para api.logging_config
    mock_log_cfg = types.ModuleType("api.logging_config")
    mock_log_cfg.configure_logging = MagicMock(return_value=mock_logger)
    sys.modules["api.logging_config"] = mock_log_cfg

    # 3. Injeta dublê para api.middleware
    mock_mw = types.ModuleType("api.middleware")
    mock_mw.LatencyMiddleware = DummyMiddleware
    sys.modules["api.middleware"] = mock_mw

    # 4. Injeta dublê para api.model_loader
    mock_loader = types.ModuleType("api.model_loader")
    mock_loader.predict = mock_predict
    sys.modules["api.model_loader"] = mock_loader

    # Força a API a recarregar limpando o cache
    if "api.main" in sys.modules:
        del sys.modules["api.main"]

    import api.main

    # CORREÇÃO: raise_server_exceptions=False permite que o @app.exception_handler funcione no teste
    client = TestClient(api.main.app, raise_server_exceptions=False)

    # Retorna o cliente e o mock de predição para controle do teste
    yield client, mock_predict

    # Teardown: Restaura os módulos originais do sistema
    sys.modules.clear()
    sys.modules.update(original_modules)

# ── BATERIA DE TESTES ──────────────────────────────────────────────────────────

def test_get_health_endpoint(api_env):
    """Garante que a rota de verificação de integridade (/health) responde HTTP 200."""
    client, _ = api_env
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_post_predict_endpoint_success(api_env):
    """Garante que a rota /predict recebe o payload e retorna as probabilidades da IA."""
    client, mock_predict = api_env
    # Configura o comportamento do mock injetado
    mock_predict.return_value = (0.7854, 1)

    payload = {"tenure": 12, "MonthlyCharges": 70.5, "SeniorCitizen": 0}
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json() == {
        "probability": 0.7854,
        "prediction": 1
    }
    mock_predict.assert_called_once_with(payload)


def test_post_predict_endpoint_internal_error_handling(api_env):
    """Garante que o exception_handler captura falhas inesperadas do modelo e responde HTTP 500 amigável."""
    client, mock_predict = api_env
    # Simula uma pane catastrófica no motor do modelo
    mock_predict.side_effect = RuntimeError("PyTorch Tensor Corrupted")

    payload = {"tenure": 5}
    response = client.post("/predict", json=payload)

    assert response.status_code == 500
    assert response.json() == {"detail": "Erro interno"}
