import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.data.features import ServiceCountTransformer, TenureGrouper, build_feature_transformer


@pytest.fixture
def sample_customer_data():
    """Fixture que fornece um DataFrame sintético com os cenários necessários."""
    return pd.DataFrame({
        "gender": ["Female", "Male", "Male", "Female"],
        "SeniorCitizen": ["0", "0", "1", "0"],
        "Partner": ["Yes", "No", "No", "Yes"],
        "Dependents": ["No", "No", "No", "No"],
        "tenure": [0, 12, 25, 75],  # Cobre limites das faixas (0-12, 13-24, 25-48, 73+)
        "PhoneService": ["No", "Yes", "Yes", "Yes"],
        "MultipleLines": ["No", "Yes", "No", "1"],  # Testando strings variantes
        "InternetService": ["DSL", "Fiber optic", "No", "DSL"],
        "OnlineSecurity": ["Yes", "No", "No", "1"],   # CORRIGIDO: de 1 (int) para "1" (str) para o OHE
        "OnlineBackup": ["No", "Yes", "No", "No"],
        "DeviceProtection": ["No", "No", "No", "Yes"],
        "TechSupport": ["No", "No", "No", "No"],
        "StreamingTV": ["No", "Yes", "No", "No"],
        "StreamingMovies": ["No", "Yes", "No", "No"],
        "Contract": ["Month-to-month", "One year", "Month-to-month", "Two year"],
        "PaperlessBilling": ["Yes", "No", "Yes", "Yes"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        "MonthlyCharges": [29.85, 56.95, 53.85, 104.80]
    })


def test_tenure_grouper_boundaries(sample_customer_data):
    """Garante que a discretização do tenure respeita os limites exatos das faixas."""
    grouper = TenureGrouper()

    # Act
    transformed_df = grouper.transform(sample_customer_data)

    # Assert
    # 1. Garante que a coluna original foi eliminada
    assert "tenure" not in transformed_df.columns
    # 2. Garante que a nova coluna foi gerada
    assert "tenure_group" in transformed_df.columns

    # 3. Validação rigorosa dos limites dos bins: [0, 12, 25, 75] -> ["0-12m", "0-12m", "25-48m", "73+m"]
    expected_groups = ["0-12m", "0-12m", "25-48m", "73+m"]
    assert transformed_df["tenure_group"].tolist() == expected_groups


def test_service_count_transformer_logic(sample_customer_data):
    """Garante que a contagem de serviços aceita 'Yes', '1' e 1 e soma perfeitamente."""
    transformer = ServiceCountTransformer()

    # Act
    transformed_df = transformer.transform(sample_customer_data)

    # Assert
    assert "service_count" in transformed_df.columns

    # Verificação das somas linha por linha:
    # Linha 0: MultipleLines(No), OnlineSecurity(Yes) -> 1
    # Linha 1: MultipleLines(Yes), OnlineSecurity(No), OnlineBackup(Yes), StreamingTV(Yes), StreamingMovies(Yes) -> 4
    # Linha 2: Todos 'No' -> 0
    # Linha 3: MultipleLines('1'), OnlineSecurity(1), DeviceProtection('Yes') -> 3
    expected_counts = [1, 4, 0, 3]
    assert transformed_df["service_count"].tolist() == expected_counts


def test_transformers_do_not_mutate_inplace(sample_customer_data):
    """Garante que as operações mantêm a imutabilidade do DataFrame original."""
    original_copy = sample_customer_data.copy()

    grouper = TenureGrouper()
    _ = grouper.transform(sample_customer_data)

    # Compara se o DataFrame original passado continua idêntico à cópia antes do transform
    pd.testing.assert_frame_equal(sample_customer_data, original_copy)


def test_build_feature_transformer_pipeline_structure(sample_customer_data):
    """Garante que o pipeline completo é gerado com a estrutura correta e consegue dar fit/transform."""
    pipeline = build_feature_transformer()

    # Assert de estrutura
    assert isinstance(pipeline, Pipeline)
    assert "tenure_grouper" in pipeline.named_steps
    assert "service_count" in pipeline.named_steps
    assert "column_transformer" in pipeline.named_steps

    # Teste de fumaça (E2E) para garantir que os dados passam por todo o pipeline sem quebrar tipos/shapes
    # Como o pipeline é retornado não-ajustado, rodamos o fit_transform
    processed_matrix = pipeline.fit_transform(sample_customer_data)

    # Como aplicamos OneHotEncoder e StandardScaler, o retorno deve ser uma matriz numpy (ou array)
    assert isinstance(processed_matrix, np.ndarray)
    # Garante que gerou registros para as 4 linhas do mock
    assert processed_matrix.shape[0] == 4
