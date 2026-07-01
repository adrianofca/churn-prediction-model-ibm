import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class TenureGrouper(BaseEstimator, TransformerMixin):
    """Discretiza 'tenure' em faixas de contrato e remove a coluna original."""

    _bins = [0, 12, 24, 48, 72, float("inf")]
    _labels = ["0-12m", "13-24m", "25-48m", "49-72m", "73+m"]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["tenure_group"] = pd.cut(
            X["tenure"], bins=self._bins, labels=self._labels, include_lowest=True
        ).astype(str)
        return X.drop(columns=["tenure"])


class ServiceCountTransformer(BaseEstimator, TransformerMixin):
    """Soma quantos serviços adicionais o cliente contratou (feature de engagement)."""

    _service_cols = [
        "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["service_count"] = (
            X[self._service_cols]
            .apply(lambda col: col.isin(["Yes", "1", 1]))
            .sum(axis=1)
            .astype(int)
        )
        return X


def build_feature_transformer() -> Pipeline:
    """
    Constrói o pipeline completo de transformações de features.

    Etapas:
      1. TenureGrouper       — discretiza 'tenure' em faixas e adiciona 'tenure_group'
      2. ServiceCountTransformer — conta serviços adicionais contratados ('service_count')
      3. ColumnTransformer   — escala numéricas e aplica OHE nas categóricas

    Returns:
        Pipeline: pipeline de features sklearn não ajustado (unfitted).
    """
    # Colunas numéricas após TenureGrouper (tenure foi removida, MonthlyCharges permanece)
    numeric_features = ["MonthlyCharges", "service_count"]

    # Colunas categóricas incluindo a nova faixa de tenure
    categorical_features = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
        "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
        "PaperlessBilling", "PaymentMethod", "tenure_group",
    ]

    column_transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, dtype=int), categorical_features),
        ],
        remainder="drop",
    )

    return Pipeline(steps=[
        ("tenure_grouper", TenureGrouper()),
        ("service_count", ServiceCountTransformer()),
        ("column_transformer", column_transformer),
    ])
