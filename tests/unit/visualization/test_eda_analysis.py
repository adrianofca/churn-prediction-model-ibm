from unittest.mock import MagicMock, patch

import pandas as pd

from src.visualization.eda_analysis import (
    calculate_iqr_outliers,
    plot_continuous_boxplots,
    plot_continuous_histograms,
    plot_target_distribution,
)


def test_calculate_iqr_outliers_math_logic():
    """Garante a precisão matemática do cálculo de outliers via IQR."""
    # Q1 (25%) = 21.5, Q3 (75%) = 28.5 -> IQR = 7.0
    # Limite Inferior = 21.5 - (1.5 * 7) = 11.0
    # Limite Superior = 28.5 + (1.5 * 7) = 39.0
    # Os valores 10 (abaixo de 11) e 100 (acima de 39) são outliers detectados. Total = 2.
    series = pd.Series([10, 20, 22, 24, 26, 28, 30, 100])

    outlier_count, limite_inf, limite_sup = calculate_iqr_outliers(series)

    assert outlier_count == 2
    assert limite_inf == 11.0
    assert limite_sup == 39.0


@patch("src.visualization.eda_analysis.plt.show")
@patch("src.visualization.eda_analysis.plt.subplots")
def test_plot_target_distribution_rendering(mock_subplots, mock_show):
    """Garante que a distribuição do target monta as duas sub-figuras e chama a exibição."""
    mock_ax1 = MagicMock()
    mock_ax2 = MagicMock()
    mock_subplots.return_value = (MagicMock(), [mock_ax1, mock_ax2])

    df_mock = pd.DataFrame({"target": [0, 0, 0, 1]})

    plot_target_distribution(df_mock, target_col="target")

    mock_subplots.assert_called_once_with(1, 2, figsize=(14, 5))
    mock_ax1.bar.assert_called_once()
    mock_ax2.pie.assert_called_once()
    mock_show.assert_called_once()


@patch("src.visualization.eda_analysis.plt.show")
@patch("src.visualization.eda_analysis.plt.subplots")
@patch("src.visualization.eda_analysis.sns.boxplot")
def test_plot_continuous_boxplots_grid(mock_boxplot, mock_subplots, mock_show):
    """Garante que múltiplos boxplots disparam as chamadas do Seaborn sem iniciar GUI."""
    mock_subplots.return_value = (MagicMock(), [MagicMock(), MagicMock()])
    df_mock = pd.DataFrame({
        "tenure": [12, 24, 36],
        "MonthlyCharges": [50.0, 70.0, 90.0]
    })

    plot_continuous_boxplots(df_mock, columns=["tenure", "MonthlyCharges"])

    assert mock_boxplot.call_count == 2
    mock_subplots.assert_called_once()
    mock_show.assert_called_once()


@patch("src.visualization.eda_analysis.plt.show")
@patch("src.visualization.eda_analysis.plt.subplots")
@patch("src.visualization.eda_analysis.sns.histplot")
def test_plot_continuous_histograms_grid(mock_histplot, mock_subplots, mock_show):
    """Garante que os histogramas com KDE processam o schema sem quebras e sem GUI."""
    # CORREÇÃO: Como n=1, o subplots real retorna um único objeto Axes, e não uma lista.
    mock_subplots.return_value = (MagicMock(), MagicMock())
    df_mock = pd.DataFrame({"tenure": [10, 20, 30]})

    # Act
    plot_continuous_histograms(df_mock, columns=["tenure"], bins=10)

    # Assert
    mock_histplot.assert_called_once()
    mock_subplots.assert_called_once()
    mock_show.assert_called_once()

def test_plot_target_distribution_missing_column_safeguard():
    """Garante resiliência caso a coluna target informada não exista no DataFrame."""
    df_vazio = pd.DataFrame({"coluna_aleatoria": [1, 2, 3]})
    result = plot_target_distribution(df_vazio, target_col="target_fantasma")
    assert result is None
