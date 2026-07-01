from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer

from src.models.arvores import compute_metrics, load_data, make_split, train_and_save


def test_make_split_logic():
    """Garante que a divisão dos dados mantém a proporção de estratificação e tamanhos corretos."""
    # Arrange: Criamos 100 linhas simuladas balanceadas (50 de cada classe)
    X = pd.DataFrame({"feature1": range(100), "feature2": range(100)})
    y = np.array([0, 1] * 50)

    # Act
    X_train, X_test, y_train, y_test = make_split(X, y)

    # Assert
    # TEST_SIZE = 0.15 de 100 = 15 registros no teste
    assert len(X_test) == 15
    assert len(y_test) == 15
    # Devido à estratificação estrita em amostras pequenas, o split resulta em 69 linhas no treino
    assert len(X_train) == 69
    assert len(y_train) == 69

    # Garante o balanceamento próximo de classes no treino
    assert np.sum(y_train == 1) == 35


def test_compute_metrics_success():
    """Garante o cálculo correto das métricas de classificação em um cenário ideal."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_proba = np.array([0.1, 0.9, 0.8, 0.7])

    metrics = compute_metrics(y_true, y_pred, y_proba)

    assert "Accuracy" in metrics
    assert "F1-Score" in metrics
    assert "AUC-ROC" in metrics
    assert "PR-AUC" in metrics
    assert metrics["Accuracy"] == 0.75


def test_compute_metrics_single_class_fallback():
    """Garante que se a base de teste contiver apenas uma classe, as métricas respondem com resiliência."""
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([0, 0, 0, 0])
    y_proba = np.array([0.1, 0.2, 0.1, 0.3])

    metrics = compute_metrics(y_true, y_pred, y_proba)

    assert np.isnan(metrics["AUC-ROC"])
    # sklearn 1.8+ retorna 0.0 com warning; versões anteriores levantam ValueError -> nan
    assert metrics["PR-AUC"] == 0.0 or np.isnan(metrics["PR-AUC"])
    assert metrics["Accuracy"] == 1.0


@patch("src.models.arvores.validate_raw")
@patch("src.models.arvores.preprocess_data")
@patch("src.models.arvores.load_raw_data")
def test_load_data_pipeline(mock_load, mock_preprocess, mock_validate):
    """Garante que a carga descarta IDs e isola a label corretamente interceptando o módulo data."""
    # Arrange
    mock_df = pd.DataFrame({
        "customerID": ["123-ABC", "456-DEF"],
        "target": [0, 1],
        "feature_util": [10, 20]
    })
    mock_load.return_value = mock_df
    mock_preprocess.return_value = mock_df

    # Act
    X_raw, y = load_data()

    # Assert
    assert "customerID" not in X_raw.columns
    assert "target" not in X_raw.columns
    assert list(X_raw.columns) == ["feature_util"]
    assert np.array_equal(y, np.array([0, 1]))


@patch("src.models.arvores.load_data")
@patch("src.models.arvores.build_feature_transformer")
@patch("src.models.arvores.mlflow")
@patch("src.models.arvores.mlflow_sklearn")
@patch("src.models.arvores.joblib.dump")
@patch("builtins.open", new_callable=mock_open)
def test_train_and_save_execution(
    mock_file, mock_joblib, mock_mlflow_sklearn, mock_mlflow, mock_build_feat, mock_load_data, tmp_path
):
    """Testa o fluxo completo de treinamento salvando artefatos e integrando com o MLflow."""
    # Arrange
    X_dummy = pd.DataFrame({"feature": range(20)})
    y_dummy = np.array([0, 1] * 10)
    mock_load_data.return_value = (X_dummy, y_dummy)

    mock_build_feat.return_value = FunctionTransformer()
    mock_mlflow.start_run.return_value.__enter__.return_value = MagicMock()

    # Act
    df_results = train_and_save(output_dir=tmp_path)

    # Assert
    assert df_results.shape[0] == 3
    assert "Decision Tree" in df_results.index
    assert "Random Forest" in df_results.index
    assert "Gradient Boosting" in df_results.index
    assert mock_joblib.call_count == 3
    assert mock_file.call_count == 3
    assert mock_mlflow_sklearn.log_model.call_count == 3
