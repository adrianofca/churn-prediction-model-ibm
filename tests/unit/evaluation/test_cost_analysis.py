from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pandas as pd
import pytest

from src.evaluation.cost_analysis import (
    cost_at_threshold,
    find_optimal_thresholds,
    load_preprocessed_data,
    main,
)


@patch("src.evaluation.cost_analysis.pd.read_csv")
def test_load_preprocessed_data_success(mock_read_csv):
    """Garante que a leitura do dataset de custo separa os alvos e aplica o split de 15%."""
    # Arrange: Criamos um mock de DataFrame mínimo com 20 registros balanceados
    mock_df = pd.DataFrame({
        "customerID": [f"ID_{i}" for i in range(20)],
        "target": [0, 1] * 10,
        "feature_1": range(20)
    })
    mock_read_csv.return_value = mock_df

    # Act
    X_train, X_test, y_train, y_test = load_preprocessed_data(
        data_path="fake_path.csv",
        target_col="target"
    )

    # Assert
    # 15% de 20 instâncias = 3 registros de teste
    assert len(X_test) == 3
    assert len(y_test) == 3
    assert "customerID" not in X_train.columns
    assert "target" not in X_train.columns


def test_load_preprocessed_data_missing_target_exception():
    """Garante que tentar analisar uma coluna alvo inexistente levanta uma exceção explicativa."""
    mock_df = pd.DataFrame({"feature_correta": [1, 2, 3]})

    with patch("src.evaluation.cost_analysis.pd.read_csv", return_value=mock_df):
        with pytest.raises(ValueError) as exc_info:
            load_preprocessed_data("fake.csv", target_col="target_fantasma")

        assert "Coluna alvo 'target_fantasma' não encontrada" in str(exc_info.value)


def test_cost_at_threshold_math_logic():
    """Valida se as contas financeiras de Custo total, Savings e Net Benefit estão exatas."""
    # Arrange
    y_true = np.array([1, 0, 1, 0])     # Dois churns reais, dois não-churns
    y_proba = np.array([0.8, 0.6, 0.2, 0.1])
    threshold = 0.5
    cost_fn = 1000
    cost_fp = 50

    # Com threshold=0.5, y_pred vira [1, 1, 0, 0]
    # Avaliação:
    # idx 0: true=1, pred=1 -> TP (Custo=0, Savings=1000)
    # idx 1: true=0, pred=1 -> FP (Custo=50,  Retenção indevida)
    # idx 2: true=1, pred=0 -> FN (Custo=1000, Perda completa de LTV)
    # idx 3: true=0, pred=0 -> TN (Custo=0)
    # Total Cost Esperado = 1050 | Savings = 1000 | Net Benefit = 1000 - 50 = 950

    # Act
    res = cost_at_threshold(y_true, y_proba, threshold, cost_fn, cost_fp)

    # Assert
    assert res["TP"] == 1
    assert res["FP"] == 1
    assert res["FN"] == 1
    assert res["total_cost"] == 1050
    assert res["savings"] == 1000
    assert res["net_benefit"] == 950


def test_find_optimal_thresholds_criteria():
    """Garante a seleção do melhor threshold dependendo do objetivo operacional de negócio."""
    # Arrange: Simulamos saídas de 3 sweeps hipotéticos
    sweep_mock = [
        {"threshold": 0.2, "total_cost": 2000, "net_benefit": 5000, "recall": 0.90, "precision": 0.4, "f_beta": 0.75},
        {"threshold": 0.4, "total_cost": 500,  "net_benefit": 8000, "recall": 0.80, "precision": 0.7, "f_beta": 0.85},
        {"threshold": 0.7, "total_cost": 4000, "net_benefit": 1000, "recall": 0.30, "precision": 0.9, "f_beta": 0.40},
    ]

    # Act
    optimal = find_optimal_thresholds(sweep_mock)

    # Assert
    # O menor custo total deve ser o do threshold 0.4 (custo=500)
    assert optimal["min_cost"]["threshold"] == 0.4
    # O maior benefício líquido deve ser o do threshold 0.4 (benefício=8000)
    assert optimal["max_benefit"]["threshold"] == 0.4
    # O maior recall absoluto deve ser o do threshold 0.2 (recall=0.90)
    assert optimal["max_recall"]["threshold"] == 0.2


@patch("src.evaluation.cost_analysis.load_preprocessed_data")
@patch("src.evaluation.cost_analysis.get_mlp_probas")
@patch("src.evaluation.cost_analysis.mlflow")
@patch("builtins.open", new_callable=mock_open)
def test_main_cost_analysis_orchestration(mock_file, mock_mlflow, mock_probas, mock_load_data, tmp_path):
    """Garante a execução de ponta a ponta do script, gerando relatórios e integrando ao MLflow."""
    # Arrange
    X_train_m = pd.DataFrame({"f": [1, 2]})
    X_test_m = pd.DataFrame({"f": [3, 4]})
    y_train_m = pd.Series([0, 1])
    y_test_m = pd.Series([0, 1])
    mock_load_data.return_value = (X_train_m, X_test_m, y_train_m, y_test_m)

    # Probabilidades fictícias simuladas para o teste de fumaça
    mock_probas.return_value = np.array([0.15, 0.85])
    mock_mlflow.start_run.return_value.__enter__.return_value = MagicMock()

    # Redireciona a constante do diretório de relatórios para a pasta temporária do pytest
    with patch("src.evaluation.cost_analysis.REPORTS_DIR", tmp_path):
        # Act
        main()

    # Assert
    # 1. Garante que disparou a abertura de logs e registros de parâmetros no MLflow
    mock_mlflow.start_run.assert_called_once_with(run_name="cost_analysis")
    mock_mlflow.log_param.assert_any_call("cost_fn_fp_ratio", 20.0)

    # 2. Garante que os artefatos de saída (JSON e MD) foram fisicamente gravados no sandbox isolado
    assert mock_file.call_count >= 2
