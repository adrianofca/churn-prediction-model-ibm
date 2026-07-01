import pandas as pd
from pandera.pandas import Check, Column, DataFrameSchema

RAW_SCHEMA = DataFrameSchema(
    columns={
        "customerID":        Column(str),
        "gender":            Column(str, Check.isin(["Male", "Female"])),
        "SeniorCitizen":     Column(int, Check.isin([0, 1])),
        "Partner":           Column(str, Check.isin(["Yes", "No"])),
        "Dependents":        Column(str, Check.isin(["Yes", "No"])),
        "tenure":            Column(int, Check.greater_than_or_equal_to(0)),
        "PhoneService":      Column(str, Check.isin(["Yes", "No"])),
        "InternetService":   Column(str),
        "Contract":          Column(str, Check.isin(["Month-to-month", "One year", "Two year"])),
        "PaperlessBilling":  Column(str, Check.isin(["Yes", "No"])),
        "MonthlyCharges":    Column(float, Check.greater_than(0)),
        "TotalCharges":      Column(str),
    },
    coerce=False,
    strict=False,
)

PREPROCESSED_SCHEMA = DataFrameSchema(
    columns={
        "SeniorCitizen":  Column(str),
        "tenure":         Column(int, Check.greater_than_or_equal_to(0)),
        "MonthlyCharges": Column(float, Check.greater_than(0)),
        "TotalCharges":   Column(float, Check.greater_than_or_equal_to(0)),
        "target":         Column(int, Check.isin([0, 1])),
    },
    coerce=False,
    strict=False,
)


def validate_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Valida o DataFrame bruto contra o RAW_SCHEMA. Levanta SchemaErrors se inválido."""
    return RAW_SCHEMA.validate(df, lazy=True)


def validate_preprocessed(df: pd.DataFrame) -> pd.DataFrame:
    """Valida o DataFrame pós-preprocessing contra o PREPROCESSED_SCHEMA."""
    return PREPROCESSED_SCHEMA.validate(df, lazy=True)
