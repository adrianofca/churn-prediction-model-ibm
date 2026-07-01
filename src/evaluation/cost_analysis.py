"""
cost_analysis.py
----------------
Analisa o trade-off financeiro entre Falsos Positivos (FP)
e Falsos Negativos (FN) para o modelo de churn.

Conceitos
---------
FP (False Positive): cliente que NÃO ia churnar, mas foi acionado.
    → Custo = ação de retenção desnecessária (desconto, ligação)

FN (False Negative): cliente que IA churnar, mas não foi detectado.
    → Custo = perda do LTV completo do cliente

A análise percorre todos os thresholds possíveis e calcula
o custo total esperado de cada ponto da curva PR.

Saídas
------
  reports/cost_analysis.json     — custo por threshold
  reports/optimal_threshold.json — threshold ótimo por critério
  reports/cost_analysis.md       — relatório narrativo

Uso
---
    cd churn_project/
    python src/cost_analysis.py
"""

import json
import logging
import os
import sys
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

_ROOT = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(_ROOT / "src"))
from models.mlp import ChurnMLP, MLPConfig  # noqa: E402

# ── Configurações ──────────────────────────────────────────────────────────────
DATA_PATH    = _ROOT / "data" / "telco_churn_preprocessed.csv"
TARGET_COL   = "target"
ID_COL       = "customerID"
MODELS_DIR   = _ROOT / "models"
REPORTS_DIR  = _ROOT / "reports"
RANDOM_STATE = 42

os.makedirs(REPORTS_DIR, exist_ok=True)

# ── Premissas de custo (valores conceituais, unidade = "unidades monetárias") ──

COST_FN = 1000   # perda de LTV: cliente churna sem intervenção
                 # representativo de ~12-18x mensalidade média (~$65)
COST_FP = 50     # custo de retenção desnecessária: desconto ou ligação
                 # ratio FN/FP ≈ 20x → forte incentivo a maximizar Recall

THRESHOLDS = np.linspace(0.01, 0.99, 199)   # varre toda a curva
BETA       = 2                               # F-beta: penaliza FN 2× mais que FP


# ── Carrega e divide os dados pré-processados ──────────────────────────────────

def load_preprocessed_data(
    data_path: Path | str,
    target_col: str,
    id_col: str = ID_COL,
    test_size: float = 0.15,
):
    """
    Lê o CSV já pré-processado e retorna X_train, X_test, y_train, y_test.

    Usa o mesmo split estratificado 70/15 do model.py (test_size=0.15, seed=42).
    Remove a coluna de ID se presente para evitar vazamento de informação.
    """
    df = pd.read_csv(data_path)

    if target_col not in df.columns:
        available = list(df.columns)
        raise ValueError(
            f"Coluna alvo '{target_col}' não encontrada no CSV.\n"
            f"Colunas disponíveis: {available}\n"
            f"Ajuste a variável TARGET_COL no topo do script."
        )

    drop_cols = [c for c in (id_col, target_col) if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    logging.info(f"CSV carregado: {len(df)} linhas, {X.shape[1]} features")
    logging.info(f"Treino: {len(X_train)} | Teste: {len(X_test)}")
    return X_train, X_test, y_train, y_test


# ── Carrega probabilidades do MLP ──────────────────────────────────────────────

def get_mlp_probas(X_train, X_test):
    """Carrega o MLP salvo e retorna probabilidades no test set."""
    ckpt_path = MODELS_DIR / "mlp_model.pt"
    if not ckpt_path.exists():
        logging.warning("mlp_model.pt não encontrado. Usando Logistic Regression como fallback.")
        return get_lr_probas(X_test)

    from data import build_feature_transformer
    transformer = build_feature_transformer()
    transformer.fit(X_train)
    X_test_arr = np.asarray(transformer.transform(X_test), dtype=np.float32)

    ckpt         = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mlp_cfg_data = ckpt["mlp_config"]
    config       = MLPConfig(**mlp_cfg_data) if isinstance(mlp_cfg_data, dict) else mlp_cfg_data
    model        = ChurnMLP(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    X_t = torch.tensor(X_test_arr, dtype=torch.float32)
    with torch.no_grad():
        probas = model.predict_proba(X_t).numpy()

    logging.info(f"MLP carregado — {len(probas)} exemplos no test set")
    return probas


def get_lr_probas(X_test):
    """Fallback: usa Logistic Regression se MLP não estiver disponível."""
    lr_path = MODELS_DIR / "baseline_logistic_regression.joblib"
    if not lr_path.exists():
        raise FileNotFoundError("Rode train_baselines.py e train_mlp.py primeiro.")
    model = joblib.load(lr_path)
    return model.predict_proba(X_test)[:, 1]


# ── Análise de custo por threshold ────────────────────────────────────────────

def cost_at_threshold(y_true, y_proba, threshold, cost_fn, cost_fp):
    """
    Calcula custo total, TP, FP, FN para um dado threshold.

    Custo total = (n_FN × cost_FN) + (n_FP × cost_FP)
    Savings     = TP × cost_FN  (clientes churn que evitamos perder)
    Net benefit = Savings − (FP × cost_FP)
    """
    y_pred = (y_proba >= threshold).astype(int)
    y_true = np.array(y_true)

    TP = int(((y_pred == 1) & (y_true == 1)).sum())
    FP = int(((y_pred == 1) & (y_true == 0)).sum())
    FN = int(((y_pred == 0) & (y_true == 1)).sum())
    TN = int(((y_pred == 0) & (y_true == 0)).sum())

    total_cost  = (FN * cost_fn) + (FP * cost_fp)
    savings     = TP * cost_fn
    net_benefit = savings - (FP * cost_fp)
    precision   = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall      = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1          = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    f_beta_denom = BETA**2 * precision + recall
    f_beta = (1 + BETA**2) * precision * recall / f_beta_denom if f_beta_denom > 0 else 0.0

    return {
        "threshold":   round(float(threshold), 4),
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "total_cost":   int(total_cost),
        "savings":      int(savings),
        "net_benefit":  int(net_benefit),
        "precision":    round(precision, 4),
        "recall":       round(recall, 4),
        "f1":           round(f1, 4),
        "f_beta":       round(f_beta, 4),
    }


def sweep_thresholds(y_true, y_proba, cost_fn, cost_fp):
    """Varre todos os thresholds e retorna lista de resultados."""
    results = []
    for t in THRESHOLDS:
        results.append(cost_at_threshold(y_true, y_proba, t, cost_fn, cost_fp))
    return results


# ── Encontra thresholds ótimos por critério ────────────────────────────────────

def find_optimal_thresholds(sweep_results: list) -> dict:
    """
    Retorna o threshold ótimo para cada critério de negócio.

    Critérios
    ---------
    min_cost     : minimiza custo total (FN×cost_FN + FP×cost_FP)
    max_benefit  : maximiza benefício líquido (savings − custo_FP)
    max_recall   : maximiza Recall (com Recall ≥ 0.75, SLO mínimo)
    max_f_beta   : maximiza F-beta(β=2) — balanceia com peso duplo no Recall
    slo_min      : maior threshold que ainda atende Recall ≥ 0.75 (ponto mais conservador)
    """
    best = {}

    best["min_cost"] = min(sweep_results, key=lambda x: x["total_cost"])
    best["max_benefit"] = max(sweep_results, key=lambda x: x["net_benefit"])
    best["max_recall"]  = max(sweep_results, key=lambda x: x["recall"])
    best["max_f_beta"]  = max(sweep_results, key=lambda x: x["f_beta"])

    # SLO mínimo: maior threshold que ainda atinge Recall ≥ 0.75 (mais conservador)
    slo_candidates = [r for r in sweep_results if r["recall"] >= 0.75]
    if slo_candidates:
        best["slo_min"] = max(slo_candidates, key=lambda x: x["threshold"])
    else:
        best["slo_min"] = best["max_recall"]

    return best


# ── Relatório Markdown ─────────────────────────────────────────────────────────

def save_markdown_report(optimal: dict, cost_fn: int, cost_fp: int):
    ratio = cost_fn / cost_fp
    lines = [
        "# Análise de Custo — Trade-off FP vs FN\n",
        "## Premissas de custo\n",
        "| Evento | Custo unitário | Justificativa |",
        "|--------|---------------|---------------|",
        f"| Falso Negativo (FN) | {cost_fn} u.m. | LTV perdido: cliente churna sem intervenção |",
        f"| Falso Positivo (FP) | {cost_fp} u.m. | Ação de retenção desnecessária (desconto/ligação) |",
        f"| **Ratio FN/FP** | **{ratio:.0f}x** | Perder um cliente custa {ratio:.0f}x mais"
        " que reter desnecessariamente |\n",
        "## Thresholds ótimos por critério\n",
        "| Critério | Threshold | Recall | Precision | FN | FP | Custo Total | Benefício Líquido |",
        "|----------|-----------|--------|-----------|----|----|-------------|-------------------|",
    ]
    for name, r in optimal.items():
        lines.append(
            f"| {name} | {r['threshold']:.2f} | {r['recall']:.3f} | "
            f"{r['precision']:.3f} | {r['FN']} | {r['FP']} | "
            f"{r['total_cost']:,} | {r['net_benefit']:,} |"
        )
    lines += [
        "\n## Interpretação\n",
        f"Com ratio FN/FP = {ratio:.0f}x, o threshold ótimo por custo tende a ser **baixo** (0.2–0.4),",
        "pois é muito mais barato acionar 20 clientes desnecessariamente do que perder 1 cliente real.",
        "\nO threshold padrão de 0.5 não é ótimo para churn — ele equilibra FP e FN como se",
        "tivessem o mesmo custo, o que contradiz a realidade do negócio.\n",
        "## Recomendação operacional\n",
        "Usar o threshold de `max_benefit` como ponto de partida para produção,",
        "revisando com a equipe de retenção o custo real de cada ação.",
    ]
    path = os.path.join(REPORTS_DIR, "cost_analysis.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logging.info(f"Relatório salvo em: {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("TechChallenge_TelcoChurn")

    sep = "=" * 60
    logging.info(sep)
    logging.info("  CHURN — ANÁLISE DE CUSTO FP vs FN")
    logging.info(sep)
    logging.info("Premissas:")
    logging.info(f"  Custo FN (cliente perdido)   : {COST_FN} u.m.")
    logging.info(f"  Custo FP (retenção indevida) : {COST_FP} u.m.")
    logging.info(f"  Ratio FN/FP                 : {COST_FN/COST_FP:.0f}x")

    # Dados: lê diretamente o CSV pré-processado (sem build_pipeline)
    X_train, X_test, y_train, y_test = load_preprocessed_data(
        data_path=DATA_PATH,
        target_col=TARGET_COL,
    )

    y_proba = get_mlp_probas(X_train, X_test)

    # Sweep de thresholds
    sweep = sweep_thresholds(y_test.values, y_proba, COST_FN, COST_FP)

    # Thresholds ótimos
    optimal = find_optimal_thresholds(sweep)

    # Exibe tabela
    header = (
        f"  {'Critério':<18} {'Threshold':>10} {'Recall':>8} {'Precision':>10} "
        f"{'FN':>5} {'FP':>5} {'Custo Total':>12} {'Benef. Líq.':>12}"
    )
    logging.info(header)
    logging.info("  " + "─" * 85)
    for name, r in optimal.items():
        logging.info(
            f"  {name:<18} {r['threshold']:>10.2f} {r['recall']:>8.3f} "
            f"{r['precision']:>10.3f} {r['FN']:>5} {r['FP']:>5} "
            f"{r['total_cost']:>12,} {r['net_benefit']:>12,}"
        )

    # Highlight: threshold recomendado
    rec = optimal["max_benefit"]
    logging.info(f"→ Threshold recomendado (max benefit): {rec['threshold']:.2f}")
    logging.info(
        f"  Recall={rec['recall']:.3f} | Precision={rec['precision']:.3f} "
        f"| FN={rec['FN']} | FP={rec['FP']}"
    )
    logging.info(
        f"  Custo total: {rec['total_cost']:,} u.m. | "
        f"Benefício líquido: {rec['net_benefit']:,} u.m."
    )

    # Compara com threshold padrão 0.5
    default = cost_at_threshold(y_test.values, y_proba, 0.5, COST_FN, COST_FP)
    logging.info("→ Threshold padrão (0.50):")
    logging.info(f"  Recall={default['recall']:.3f} | FN={default['FN']} | FP={default['FP']}")
    logging.info(
        f"  Custo total: {default['total_cost']:,} u.m. | "
        f"Benefício líquido: {default['net_benefit']:,} u.m."
    )
    delta = default["total_cost"] - rec["total_cost"]
    logging.info(
        f"→ Economia ao usar threshold ótimo vs 0.5: {delta:,} u.m. "
        f"({delta/max(default['total_cost'],1)*100:.1f}%)"
    )

    # Persiste
    out = {
        "cost_assumptions": {"cost_fn": COST_FN, "cost_fp": COST_FP},
        "threshold_default_0.5": default,
        "optimal_thresholds": optimal,
        "sweep_sample": sweep[::10],  # 1 em cada 10 para não inflar o JSON
    }
    path = os.path.join(REPORTS_DIR, "cost_analysis.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    logging.info(f"Resultados salvos em: {path}")

    save_markdown_report(optimal, COST_FN, COST_FP)

    md_path = os.path.join(REPORTS_DIR, "cost_analysis.md")
    with mlflow.start_run(run_name="cost_analysis"):
        mlflow.log_param("cost_fn", COST_FN)
        mlflow.log_param("cost_fp", COST_FP)
        mlflow.log_param("cost_fn_fp_ratio", COST_FN / COST_FP)
        for criterion, r in optimal.items():
            mlflow.log_metric(f"{criterion}__threshold",   r["threshold"])
            mlflow.log_metric(f"{criterion}__recall",      r["recall"])
            mlflow.log_metric(f"{criterion}__precision",   r["precision"])
            mlflow.log_metric(f"{criterion}__total_cost",  r["total_cost"])
            mlflow.log_metric(f"{criterion}__net_benefit", r["net_benefit"])
        mlflow.log_metric("default_0_5__total_cost",  default["total_cost"])
        mlflow.log_metric("default_0_5__net_benefit", default["net_benefit"])
        mlflow.log_artifact(path)
        mlflow.log_artifact(md_path)

    logging.info("cost_analysis concluído.")


if __name__ == "__main__":
    main()
