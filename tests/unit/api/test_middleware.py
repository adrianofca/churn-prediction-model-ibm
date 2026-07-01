from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.middleware import LatencyMiddleware


@pytest.fixture(name="test_app_client")
def fixture_test_app_client():
    """Cria uma aplicação FastAPI efêmera configurada com o LatencyMiddleware para o escopo do teste."""
    app = FastAPI()
    app.add_middleware(LatencyMiddleware)

    @app.get("/mock-endpoint")
    def dummy_route():
        return {"message": "hello"}

    return TestClient(app)


def test_latency_middleware_logs_and_calculates_correct_delta(test_app_client, caplog):
    """
    Garante que o middleware intercepta a requisição, calcula a latência exata
    através de carimbos de tempo mockados e imprime os marcadores no stdout.
    """
    # Solução para o StopIteration: Como o módulo time é embutido e compartilhado,
    # usamos uma função dinâmica para responder às chamadas internas do TestClient/cookiejar.
    tempos_planejados = [999.0, 100.0, 100.123456]

    def relogio_sintetico():
        return tempos_planejados.pop(0) if tempos_planejados else 100.0

    with patch("api.middleware.time.time", side_effect=relogio_sintetico):
        with caplog.at_level("INFO", logger="api.middleware"):
        # Act: Dispara a requisição contra a nossa rota controlada
            response = test_app_client.get("/mock-endpoint")

    # Assert
    # 1. Garante que o fluxo de entrega da rota continuou íntegro e respondeu HTTP 200
    assert response.status_code == 200
    assert response.json() == {"message": "hello"}

    # 2. Captura as saídas de texto enviadas para o terminal pelo comando 'print'
    log_output = caplog.text

    # Valida se o formato do log inicial de captura foi impresso
    assert "[MIDDLEWARE] GET /mock-endpoint" in log_output

    # Valida se o cálculo de arredondamento em 4 casas decimais funcionou (0.1235)
    assert "[LATENCY] 0.1235s" in log_output

def test_latency_middleware_preserves_response_headers_and_body(test_app_client):
    """Garante que a passagem pelo middleware não corrompe ou altera o payload original de resposta."""
    response = test_app_client.get("/mock-endpoint")

    assert response.status_code == 200
    assert "content-type" in response.headers
    assert response.json() == {"message": "hello"}
