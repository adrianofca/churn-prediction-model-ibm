"""Fixtures compartilhadas entre todos os testes do projeto."""
import pandas as pd
import pytest


@pytest.fixture
def sample_row() -> dict:
    """Linha realista que respeita o schema bruto do Telco Churn."""
    return {
        "customerID":       "0001-ABC",
        "gender":           "Female",
        "SeniorCitizen":    0,
        "Partner":          "Yes",
        "Dependents":       "No",
        "tenure":           12,
        "PhoneService":     "Yes",
        "MultipleLines":    "No",
        "InternetService":  "Fiber optic",
        "OnlineSecurity":   "No",
        "OnlineBackup":     "Yes",
        "DeviceProtection": "No",
        "TechSupport":      "No",
        "StreamingTV":      "Yes",
        "StreamingMovies":  "No",
        "Contract":         "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod":    "Electronic check",
        "MonthlyCharges":   79.85,
        "TotalCharges":     "958.20",
        "Churn":            "No",
    }


@pytest.fixture
def sample_df(sample_row) -> pd.DataFrame:
    """DataFrame com 3 linhas válidas variando tenure, contrato e charges."""
    rows = []
    for tenure, charges, contract, churn in [
        (1,  50.0, "Month-to-month", "No"),
        (24, 70.0, "One year",       "Yes"),
        (60, 95.0, "Two year",       "No"),
    ]:
        r = dict(sample_row)
        r["tenure"] = tenure
        r["MonthlyCharges"] = charges
        r["TotalCharges"] = str(round(tenure * charges, 2))
        r["Contract"] = contract
        r["Churn"] = churn
        rows.append(r)
    return pd.DataFrame(rows)
