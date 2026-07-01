from unittest.mock import patch

import pandas as pd
import pytest

from src.data.data_loader import load_raw_data


def test_load_raw_data_success():
    """
    Garante que os dados são carregados corretamente e que os espaços em branco
    nos nomes das colunas são limpos (strip).
    """
    # Arrange: Criamos um DataFrame simulado com espaços propositais nas colunas
    mock_data = {
        "  id  ": [1, 2, 3],
        "feature_v1 ": [0.1, 0.2, 0.3],
        " target": [1, 0, 1]
    }
    mock_df = pd.DataFrame(mock_data)
    fake_url = "https://fake-datasource.com/data.csv"

    # Act: Interceptamos o pd.read_csv para retornar nosso DataFrame mockado
    with patch("src.data.data_loader.pd.read_csv") as mock_read_csv:
        mock_read_csv.return_value = mock_df

        result_df = load_raw_data(fake_url)

        # Assert: Verificações rigorosas
        # 1. Garante que o pandas foi chamado com a URL correta
        mock_read_csv.assert_called_once_with(fake_url)

        # 2. Garante que o strip funcionou em todas as colunas
        expected_columns = ["id", "feature_v1", "target"]
        assert list(result_df.columns) == expected_columns

        # 3. Garante que as dimensões do DataFrame não mudaram
        assert result_df.shape == (3, 3)


def test_load_raw_data_exception_handling():
    """
    Garante que, caso ocorra um erro de I/O, a função faz o log e propaga a exceção (raise).
    """
    fake_url = "https://invalid-url-that-causes-error.com/data.csv"

    # Act & Assert
    with patch("src.data.data_loader.pd.read_csv") as mock_read_csv:
        mock_read_csv.side_effect = FileNotFoundError("Arquivo não encontrado.")

        with pytest.raises(FileNotFoundError) as exc_info:
            load_raw_data(fake_url)

        assert "Arquivo não encontrado." in str(exc_info.value)
