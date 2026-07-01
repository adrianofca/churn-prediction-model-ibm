"""Carrega o modelo treinado e o pipeline de transformação, e executa a predição de churn."""

import logging
import os

import joblib
import pandas as pd
import torch

from src.models.mlp_persistence import load_model

logger = logging.getLogger(__name__)

# Threshold de decisão calibrado pela análise de custo (reports/cost_analysis.md):
# FN custa 20x mais que FP, então o threshold que minimiza o custo esperado é 0.17
# (vs. 0.5 padrão). Pode ser sobrescrito via variável de ambiente CHURN_THRESHOLD.
CHURN_THRESHOLD = float(os.getenv("CHURN_THRESHOLD", "0.17"))

model = load_model()

transformer = joblib.load(
    "models/transformer.pkl"
)

ohe = (
    transformer.named_steps["column_transformer"]
    .named_transformers_["cat"]
)

for i, cats in enumerate(ohe.categories_):
    logger.debug(f"Coluna {i}: {cats} (dtype={cats.dtype})")

def predict(data):

    df = pd.DataFrame([data])
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)

    logger.debug(f"DataFrame:\n{df}")
    logger.debug(f"Dtypes:\n{df.dtypes}")

    x = transformer.transform(df)

    x = torch.tensor(
        x,
        dtype=torch.float32
    )

    probability = (
        model
        .predict_proba(x)
        .item()
    )

    prediction = int(
        probability >= CHURN_THRESHOLD
    )

    return probability, prediction
