"""
Smoke tests: verificam que o pipeline completo (dados → features → modelo) executa
sem exceções com dados sintéticos mínimos. Não validam acurácia, apenas sobrevivência.
"""
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from api.main import app
from src.data.data_loader import load_raw_data
from src.data.features import build_feature_transformer
from src.data.preprocessing import preprocess_data
from src.data.schema import validate_raw
from src.models.mlp import ChurnMLP, MLPConfig


@pytest.fixture
def minimal_raw_dataframe():
    """DataFrame sintético mínimo que espelha o schema real do dataset IBM Telco."""
    return pd.DataFrame({
        "customerID":       [f"ID-{i:04d}" for i in range(20)],
        "gender":           (["Female", "Male"] * 10),
        "SeniorCitizen":    ([0, 1] * 10),
        "Partner":          (["Yes", "No"] * 10),
        "Dependents":       (["No", "Yes"] * 10),
        "tenure":           list(range(0, 20)),
        "PhoneService":     (["Yes", "No"] * 10),
        "MultipleLines":    (["No", "Yes"] * 10),
        "InternetService":  (["DSL", "Fiber optic"] * 10),
        "OnlineSecurity":   (["Yes", "No"] * 10),
        "OnlineBackup":     (["No", "Yes"] * 10),
        "DeviceProtection": (["No", "Yes"] * 10),
        "TechSupport":      (["No", "Yes"] * 10),
        "StreamingTV":      (["No", "Yes"] * 10),
        "StreamingMovies":  (["No", "Yes"] * 10),
        "Contract":         (["Month-to-month", "One year", "Two year", "Month-to-month", "One year"] * 4),
        "PaperlessBilling": (["Yes", "No"] * 10),
        "PaymentMethod":    (["Electronic check", "Mailed check", "Bank transfer", "Credit card"] * 5),
        "MonthlyCharges":   [round(20.0 + i * 3.5, 2) for i in range(20)],
        "TotalCharges":     [str(round(20.0 * (i + 1), 2)) for i in range(20)],
        "Churn":            (["No", "Yes"] * 10),
    })


def test_smoke_data_loader():
    """Garante que load_raw_data executa e retorna um DataFrame com colunas limpas."""
    fake_url = "https://fake-source.com/data.csv"
    mock_df = pd.DataFrame({"  col_a ": [1, 2], " col_b": [3, 4]})

    with patch("src.data.data_loader.pd.read_csv", return_value=mock_df):
        result = load_raw_data(fake_url)

    assert isinstance(result, pd.DataFrame)
    assert all(col == col.strip() for col in result.columns)


def test_smoke_preprocessing_pipeline(minimal_raw_dataframe):
    """Garante que preprocess_data executa do início ao fim sem exceção."""
    result = preprocess_data(minimal_raw_dataframe)

    assert isinstance(result, pd.DataFrame)
    assert "target" in result.columns
    assert "Churn" not in result.columns
    assert result["TotalCharges"].isnull().sum() == 0


def test_smoke_feature_pipeline(minimal_raw_dataframe):
    """Garante que o pipeline de features consome o DataFrame preprocessado sem quebrar."""
    processed = preprocess_data(minimal_raw_dataframe)
    X = processed.drop(columns=["target", "customerID"], errors="ignore")

    pipeline = build_feature_transformer()
    result = pipeline.fit_transform(X)

    assert isinstance(result, np.ndarray)
    assert result.shape[0] == len(minimal_raw_dataframe)
    assert result.shape[1] > 0


@pytest.fixture
def preprocessed_Xy(minimal_raw_dataframe):
    """DataFrame preprocessado e splitado em X/y, pronto para entrar no pipeline de features."""
    processed = preprocess_data(minimal_raw_dataframe)
    X = processed.drop(columns=["target", "customerID"], errors="ignore")
    y = processed["target"]
    return X, y


def test_smoke_end_to_end_with_dummy_classifier(preprocessed_Xy):
    """
    Teste de fumaça completo: dados brutos → preprocessing → features → treino → predição.
    Usa DummyClassifier para garantir que o pipeline mecânico funciona independente de acurácia.
    """
    X, y = preprocessed_Xy

    full_pipeline = Pipeline(steps=[
        ("features", build_feature_transformer()),
        ("classifier", DummyClassifier(strategy="stratified", random_state=42)),
    ])

    full_pipeline.fit(X, y)
    predictions = full_pipeline.predict(X)

    assert len(predictions) == len(y)
    assert set(predictions).issubset({0, 1})


def test_smoke_end_to_end_with_logistic_regression(preprocessed_Xy):
    """
    Garante que o pipeline completo funciona com Regressão Logística (modelo baseline real).
    Valida predict_proba além do predict para garantir compatibilidade com análise de custo.
    """
    X, y = preprocessed_Xy

    full_pipeline = Pipeline(steps=[
        ("features", build_feature_transformer()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])

    full_pipeline.fit(X, y)
    predictions = full_pipeline.predict(X)
    probas = full_pipeline.predict_proba(X)[:, 1]

    assert len(predictions) == len(y)
    assert probas.min() >= 0.0
    assert probas.max() <= 1.0


def test_smoke_validate_raw(minimal_raw_dataframe):
    """Garante que validate_raw aceita um DataFrame bruto bem-formado sem levantar exceção."""
    validated = validate_raw(minimal_raw_dataframe)
    assert len(validated) == len(minimal_raw_dataframe)


@pytest.mark.parametrize("clf,name", [
    (DecisionTreeClassifier(max_depth=3, random_state=42),       "DecisionTree"),
    (RandomForestClassifier(n_estimators=5, random_state=42),    "RandomForest"),
    (GradientBoostingClassifier(n_estimators=5, random_state=42),"GradientBoosting"),
])
def test_smoke_arvores(preprocessed_Xy, clf, name):
    """Garante que cada modelo de árvore treina e prediz sem exceção."""
    X, y = preprocessed_Xy

    pipeline = Pipeline(steps=[
        ("features", build_feature_transformer()),
        ("classifier", clf),
    ])

    pipeline.fit(X, y)
    predictions = pipeline.predict(X)
    probas = pipeline.predict_proba(X)[:, 1]

    assert len(predictions) == len(y), f"{name}: número de predições incorreto"
    assert set(predictions).issubset({0, 1}), f"{name}: predições fora do domínio binário"
    assert probas.min() >= 0.0 and probas.max() <= 1.0, f"{name}: probabilidades fora de [0, 1]"


def test_smoke_churn_mlp(preprocessed_Xy):
    """Garante que o ChurnMLP executa forward pass e retorna probabilidades válidas."""
    X, y = preprocessed_Xy

    feature_matrix = build_feature_transformer().fit_transform(X)
    input_dim = feature_matrix.shape[1]

    model = ChurnMLP(MLPConfig(input_dim=input_dim, hidden_dims=[32, 16], use_batch_norm=False))
    x_tensor = torch.tensor(feature_matrix, dtype=torch.float32)

    probas = model.predict_proba(x_tensor)
    predictions = model.predict(x_tensor)

    assert probas.shape == (len(X),), "ChurnMLP: shape das probabilidades incorreto"
    assert probas.min() >= 0.0 and probas.max() <= 1.0, "ChurnMLP: probabilidades fora de [0, 1]"
    assert set(predictions.numpy()).issubset({0, 1}), "ChurnMLP: predições fora do domínio binário"


@pytest.fixture
def api_client():
    """Cliente de teste para a aplicação FastAPI real (modelo e transformer do disco)."""
    return TestClient(app)


def test_smoke_api_health(api_client):
    """Garante que /health responde 200 com a aplicação real."""
    response = api_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_smoke_api_predict(api_client):
    """Garante que /predict executa o pipeline real (transformer + modelo) sem exceção."""
    payload = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.5,
        "TotalCharges": 845.0,
    }

    response = api_client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["prediction"] in (0, 1)
