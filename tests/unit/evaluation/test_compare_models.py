import json
from unittest.mock import MagicMock, patch

from src.evaluation.compare_models import main


def test_compare_models_happy_path(tmp_path):
    """
    Garante que o consolidador lê os arquivos JSON existentes, monta o DataFrame,
    ordena pelo maior AUC-ROC, gera o CSV de relatório e desbanca os dados pro MLflow.
    """
    # Arrange: Criamos subpastas temporárias isoladas para simular o ambiente
    fake_models_dir = tmp_path / "models"
    fake_reports_dir = tmp_path / "reports"
    fake_models_dir.mkdir()

    # 1. Criamos dados de métricas sintéticos para 2 modelos (um fraco e um forte)
    dummy_metrics = {
        "Accuracy": 0.50, "F1-Score": 0.40, "AUC-ROC": 0.50, "PR-AUC": 0.30
    }
    rf_metrics = {
        "Accuracy": 0.85, "F1-Score": 0.82, "AUC-ROC": 0.92, "PR-AUC": 0.88
    }

    # 2. Gravamos fisicamente na nossa pasta temporária simulando o fim do treino
    with open(fake_models_dir / "baseline_dummy_model_metrics.json", "w") as f:
        json.dump(dummy_metrics, f)
    with open(fake_models_dir / "random_forest_metrics.json", "w") as f:
        json.dump(rf_metrics, f)

    # 3. Aplicamos o patch nas constantes de caminho globais dentro do módulo alvo
    with patch("src.evaluation.compare_models.MODELS", fake_models_dir), \
         patch("src.evaluation.compare_models.REPORTS_DIR", fake_reports_dir), \
         patch("src.evaluation.compare_models.mlflow") as mock_mlflow:

        mock_mlflow.start_run.return_value.__enter__.return_value = MagicMock()

        # Act
        df_result = main()

        # Assert
        assert df_result is not None
        # Garante que o DataFrame processou apenas as duas bases existentes
        assert "Dummy Baseline" in df_result.index
        assert "Random Forest" in df_result.index
        assert len(df_result) == 2

        # REGRA DE NEGÓCIO: Garante que a ordenação colocou o melhor modelo (RF) no topo
        assert df_result.index[0] == "Random Forest"

        # Garante que o relatório consolidado físico foi salvo com sucesso
        assert (fake_reports_dir / "compare_models.csv").exists()

        # Garante a formatação e limpeza de strings enviada ao MLflow tracking
        mock_mlflow.start_run.assert_called_once_with(run_name="compare_models")
        mock_mlflow.log_metric.assert_any_call("Random_Forest__AUC_ROC", 0.92)


def test_compare_models_no_metrics_found(tmp_path):
    """Garante que se nenhuma métrica for localizada, o script avisa no log e aborta sem quebrar."""
    fake_models_dir = tmp_path / "models"
    fake_reports_dir = tmp_path / "reports"
    fake_models_dir.mkdir() # Pasta vazia sem nenhum arquivo JSON

    with patch("src.evaluation.compare_models.MODELS", fake_models_dir), \
         patch("src.evaluation.compare_models.REPORTS_DIR", fake_reports_dir), \
         patch("src.evaluation.compare_models.mlflow") as mock_mlflow:

        # Act
        df_result = main()

        # Assert
        assert df_result is None
        # Não deve abrir execução no MLflow se não há dados para computar
        mock_mlflow.start_run.assert_not_called()
