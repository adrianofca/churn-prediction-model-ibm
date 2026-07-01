
import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Configuração global estética para manter o padrão visual do projeto
sns.set_theme(style="whitegrid", rc={"axes.spines.top": False, "axes.spines.right": False})

def plot_target_distribution(df: pd.DataFrame, target_col: str = 'target') -> None:
    """
    Calcula as estatísticas e plota a distribuição da variável alvo (Target).
    Exibe um gráfico de barras e um gráfico de piza lado a lado.
    """
    if target_col not in df.columns:
        logger.warning("Coluna %s não encontrada no DataFrame.", target_col)
        return

    target_counts = df[target_col].value_counts()
    target_percentages = df[target_col].value_counts(normalize=True) * 100

    logger.info("=== DISTRIBUIÇÃO DA VARIÁVEL TARGET ===")
    for idx, pct in target_percentages.items():
        label = "No (0)" if idx == 0 else "Yes (1)"
        logger.info("%s: %d ocorrências (%.2f%%)", label, target_counts[idx], pct)

    # Verificação de desbalanceamento
    ratio = target_counts.min() / target_counts.max()
    logger.info("Ratio de balanceamento: %.2f", ratio)
    if ratio < 0.5:
        logger.warning("Dataset desbalanceado! Considere usar técnicas como SMOTE ou class_weight no modelo.")
    else:
        logger.info("Dataset razoavelmente balanceado.")

    # Construção dos Gráficos
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Gráfico de Barras
    axes[0].bar(['No', 'Yes'], target_counts.values, color=['#A3D2CA', '#FF9F9F'])
    axes[0].set_ylabel('Frequência')
    axes[0].set_title('Contagem da Variável Target', fontweight='bold', pad=10)

    # Gráfico de Piza
    axes[1].pie(target_counts.values, labels=['No', 'Yes'], autopct='%1.1f%%',
                colors=['#A3D2CA', '#FF9F9F'], startangle=90)
    axes[1].set_title('Proporção de Classes', fontweight='bold', pad=10)

    plt.tight_layout()
    plt.show()


def calculate_iqr_outliers(series: pd.Series):
    """Função utilitária interna para calcular limites e contagem de outliers via IQR."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    limite_inf = q1 - 1.5 * iqr
    limite_sup = q3 + 1.5 * iqr
    outlier_count = ((series < limite_inf) | (series > limite_sup)).sum()
    return outlier_count, limite_inf, limite_sup


def plot_continuous_boxplots(df: pd.DataFrame, columns: list[str]) -> None:
    """
    Gera Boxplots elegantes para as variáveis contínuas especificadas,
    com uma etiqueta (badge) indicando a quantidade de outliers calculados via IQR.
    """
    n = len(columns)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    # Garante que axes seja um array mesmo se houver apenas 1 coluna
    if n == 1:
        axes = [axes]

    cores = ['#FF9F9F', '#A3D2CA', '#5EAAA8', '#D3B5E5']

    for i, col in enumerate(columns):
        outlier_count, _, _ = calculate_iqr_outliers(df[col])
        color_idx = i % len(cores)

        sns.boxplot(
            x=df[col],
            ax=axes[i],
            color=cores[color_idx],
            width=0.4,
            linewidth=1.5,
            flierprops={"marker": "o", "markersize": 6, "markerfacecolor": "#e74c3c", "alpha": 0.6}
        )

        axes[i].set_title(col.upper(), fontsize=13, pad=15, fontweight='bold', color='#4a4a4a')
        axes[i].set_xlabel('')

        # Anotação de Outliers (Badge)
        axes[i].text(
            0.95, 0.95, f'Outliers (IQR): {outlier_count}',
            transform=axes[i].transAxes, fontsize=10, fontweight='bold', color='#333333',
            ha='right', va='top',
            bbox=dict(boxstyle="round,pad=0.5", edgecolor='#d3d3d3', facecolor='#f8f9fa', alpha=0.9)
        )

    plt.tight_layout()
    plt.show()


def plot_continuous_histograms(df: pd.DataFrame, columns: list[str], bins: int = 20) -> None:
    """
    Gera Histogramas com curvas KDE para as variáveis contínuas,
    mostrando graficamente o comportamento estatístico.
    """
    n = len(columns)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for i, col in enumerate(columns):
        outlier_count, _, _ = calculate_iqr_outliers(df[col])

        sns.histplot(df[col], ax=axes[i], color='coral', kde=True, bins=bins)

        axes[i].set_title(f'Distribuição: {col}', fontweight='bold', fontsize=12)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequência')

        axes[i].text(
            0.95, 0.95, f'Outliers (IQR): {outlier_count}',
            transform=axes[i].transAxes, fontsize=9, fontweight='bold',
            va='top', ha='right',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray', boxstyle='round,pad=0.5')
        )

    plt.tight_layout()
    plt.show()
