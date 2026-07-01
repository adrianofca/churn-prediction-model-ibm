from unittest.mock import ANY, MagicMock, mock_open, patch  # Adicionado o ANY aqui

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer

from src.models.baselines import main, train_and_log_pipeline


@patch("src.models.baselines.build_feature_transformer")
@patch("src.models.baselines.mlflow")
@patch("src.models.baselines.mlflow_sklearn")
@patch("src.models.baselines.joblib.dump")
@patch("builtins.open", new_callable=mock_open)
def test_train_and_log_pipeline_success(mock_file, mock_joblib, mock_mlflow_sklearn, mock_mlflow, mock_build_feat):
    """
    Garante que o pipeline de baseline executa o ciclo completo de fit, prediz
    as probabilidades com segurança, loga no MLflow e gera os artefatos locais mockados.
    """
    # Arrange: Amostras sintéticas rápidas para treino e teste
    X_train = pd.DataFrame({"feature": range(10)})
    X_test = pd.DataFrame({"feature": range(5)})
    y_train = np.array([0, 1] * 5)
    y_test = np.array([0, 1, 0, 1, 0])

    # Substitui o transformador de features real por um dublê neutro (pass-through)
    mock_build_feat.return_value = FunctionTransformer()

    # Simula o gerenciador de contexto do MLflow (with mlflow.start_run)
    mock_mlflow.start_run.return_value.__enter__.return_value = MagicMock()

    model = LogisticRegression()
    model_name = "test_logistic_baseline"

    # Act
    train_and_log_pipeline(model_name, model, X_train, X_test, y_train, y_test)

    # Assert
    # 1. Garante o acionamento do ciclo de rastreamento do MLflow
    mock_mlflow.start_run.assert_called_once_with(run_name=model_name)
    mock_mlflow.log_metric.assert_any_call("F1_Score", ANY)  # Corrigido para ANY
    mock_mlflow_sklearn.log_model.assert_called_once()

    # 2. Como a rota física em baselines.py é hardcoded (models/), garantimos que
    # as funções interceptaram a escrita física para não gerar sujeira no disco real
    mock_joblib.assert_called_once()
    mock_file.assert_called_once_with(f"models/{model_name}_metrics.json", "w")


@patch("src.models.baselines.load_raw_data")
@patch("src.models.baselines.validate_raw")
@patch("src.models.baselines.preprocess_data")
@patch("src.models.baselines.train_and_log_pipeline")
@patch("src.models.baselines.os.makedirs")
@patch("src.models.baselines.mlflow")
@patch("pandas.DataFrame.to_csv")
def test_main_orchestration(
    mock_to_csv, mock_mlflow, mock_makedirs, mock_train_log, mock_preprocess, mock_validate, mock_load_raw
):
    """Garante que a função principal (main) coordena o pipeline de ingestão e dispara os dois baselines de controle."""
    # Arrange: Mock de um DataFrame balanceado com 20 registros
    mock_df = pd.DataFrame({
        "target": [0, 1] * 10,
        "feature1": range(20)
    })
    mock_load_raw.return_value = mock_df
    mock_preprocess.return_value = mock_df

    # Act
    main()

    # Assert
    # 1. Garante que as pastas de segurança estruturais foram requisitadas
    mock_makedirs.assert_any_call("data", exist_ok=True)
    mock_makedirs.assert_any_call("models", exist_ok=True)

    # 2. Garante que a rotina disparou exatamente 2 treinamentos (Dummy e Regressão Logística)
    assert mock_train_log.call_count == 2

    # 3. Valida se os slugs de identificação passados aos experimentos estão corretos
    call_args = mock_train_log.call_args_list
    assert call_args[0][0][0] == "baseline_dummy_model"
    assert call_args[1][0][0] == "baseline_logistic_regression"
