import pandas as pd

from src.data.preprocessing import preprocess_data


def test_preprocess_data_happy_path():
    """Garante o comportamento padrão com dados perfeitamente limpos e conversões corretas."""
    # Arrange
    input_df = pd.DataFrame({
        "TotalCharges": ["100.50", "2050.25"],
        "Churn": ["No", "Yes"],
        "SeniorCitizen": [0, 1]
    })

    # Act
    result_df = preprocess_data(input_df)

    # Assert
    # 1. Valida se TotalCharges virou numérico puro (float)
    assert pd.api.types.is_float_dtype(result_df["TotalCharges"])
    assert result_df["TotalCharges"].tolist() == [100.50, 2050.25]

    # 2. Valida se a coluna Churn foi renomeada e mapeada com sucesso
    assert "Churn" not in result_df.columns
    assert "target" in result_df.columns
    assert result_df["target"].tolist() == [0, 1]

    # 3. Valida a correção de schema para string em SeniorCitizen
    assert pd.api.types.is_string_dtype(result_df["SeniorCitizen"])
    assert result_df["SeniorCitizen"].tolist() == ["0", "1"]


def test_preprocess_data_coercion_and_imputation():
    """
    Garante que strings em branco ou inválidas em TotalCharges são coagidas
    para NaN e depois imputadas com 0.0 (regra de negócio de novos clientes).
    """
    # Arrange: Espaço em branco simulando o comportamento clássico do dataset do IBM Telco
    input_df = pd.DataFrame({
        "TotalCharges": [" ", "500.00", "None"],
        "Churn": ["No", "No", "Yes"],
        "SeniorCitizen": [0, 0, 0]
    })

    # Act
    result_df = preprocess_data(input_df)

    # Assert
    # O espaço em branco " " e a string "None" devem virar NaN e depois ser preenchidos com 0.0
    expected_values = [0.0, 500.00, 0.0]
    assert result_df["TotalCharges"].tolist() == expected_values
    assert result_df["TotalCharges"].isnull().sum() == 0


def test_preprocess_data_lowercase_churn():
    """Garante a resiliência do pipeline caso a coluna alvo venha em minúsculo ('churn')."""
    # Arrange
    input_df = pd.DataFrame({
        "TotalCharges": ["120.0"],
        "churn": ["Yes"],
        "SeniorCitizen": [1]
    })

    # Act
    result_df = preprocess_data(input_df)

    # Assert
    assert "churn" not in result_df.columns
    assert "target" in result_df.columns
    assert result_df["target"].tolist() == [1]


def test_preprocess_data_no_inplace_mutation():
    """Garante que a função trabalha em uma cópia segura e preserva o DataFrame de entrada."""
    # Arrange
    input_df = pd.DataFrame({
        "TotalCharges": [" "],
        "Churn": ["No"],
        "SeniorCitizen": [0]
    })
    input_copy = input_df.copy()

    # Act
    _ = preprocess_data(input_df)

    # Assert
    pd.testing.assert_frame_equal(input_df, input_copy)
