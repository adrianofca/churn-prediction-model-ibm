import pandas as pd
import pandera.pandas as pa
import pytest

from src.data.preprocessing import preprocess_data
from src.data.schema import validate_preprocessed, validate_raw

# ── Testes do schema bruto ────────────────────────────────────────────────────

def test_valid_rows_pass(sample_df):
    """Garante que um DataFrame bruto bem-formado passa na validação."""
    validated = validate_raw(sample_df)
    assert len(validated) == len(sample_df)


def test_invalid_gender_fails(sample_row):
    """Garante que gender fora do domínio ['Male','Female'] é rejeitado."""
    bad = dict(sample_row)
    bad["gender"] = "Outro"
    with pytest.raises(pa.errors.SchemaErrors):
        validate_raw(pd.DataFrame([bad]))


def test_negative_tenure_fails(sample_row):
    """Garante que tenure negativo levanta SchemaErrors."""
    bad = dict(sample_row)
    bad["tenure"] = -1
    with pytest.raises(pa.errors.SchemaErrors):
        validate_raw(pd.DataFrame([bad]))


def test_invalid_contract_fails(sample_row):
    """Garante que um tipo de contrato fora do domínio é rejeitado."""
    bad = dict(sample_row)
    bad["Contract"] = "Weekly"
    with pytest.raises(pa.errors.SchemaErrors):
        validate_raw(pd.DataFrame([bad]))


def test_invalid_senior_citizen_fails(sample_row):
    """Garante que SeniorCitizen fora de [0, 1] é rejeitado."""
    bad = dict(sample_row)
    bad["SeniorCitizen"] = 2
    with pytest.raises(pa.errors.SchemaErrors):
        validate_raw(pd.DataFrame([bad]))


def test_negative_monthly_charges_fails(sample_row):
    """Garante que MonthlyCharges <= 0 é rejeitado."""
    bad = dict(sample_row)
    bad["MonthlyCharges"] = -10.0
    with pytest.raises(pa.errors.SchemaErrors):
        validate_raw(pd.DataFrame([bad]))


def test_invalid_partner_fails(sample_row):
    """Garante que Partner fora de ['Yes', 'No'] é rejeitado."""
    bad = dict(sample_row)
    bad["Partner"] = "Maybe"
    with pytest.raises(pa.errors.SchemaErrors):
        validate_raw(pd.DataFrame([bad]))


# ── Testes do schema pós-preprocessing ───────────────────────────────────────

def test_preprocessed_schema_valid(sample_df):
    """Garante que o output de preprocess_data respeita o schema pós-processamento."""
    processed = preprocess_data(sample_df)
    validated = validate_preprocessed(processed)
    assert validated is not None


def test_preprocessed_senior_citizen_is_string(sample_df):
    """Garante que SeniorCitizen vira string após o preprocessing (regra de negócio)."""
    processed = preprocess_data(sample_df)
    assert processed["SeniorCitizen"].dtype == object
    validate_preprocessed(processed)


def test_preprocessed_target_is_binary(sample_df):
    """Garante que o target só contém 0 e 1 após o mapeamento."""
    processed = preprocess_data(sample_df)
    assert set(processed["target"].unique()).issubset({0, 1})
    validate_preprocessed(processed)


def test_preprocessed_no_nulls_in_total_charges(sample_row):
    """Garante que TotalCharges não contém nulos após o preprocessing."""
    bad = dict(sample_row)
    bad["TotalCharges"] = " "
    processed = preprocess_data(pd.DataFrame([bad]))
    assert processed["TotalCharges"].isnull().sum() == 0
    validate_preprocessed(processed)


def test_preprocessed_negative_monthly_charges_fails(sample_df):
    """Garante que MonthlyCharges <= 0 no schema pós-processamento é rejeitado."""
    processed = preprocess_data(sample_df).copy()
    processed.loc[0, "MonthlyCharges"] = -5.0
    with pytest.raises(pa.errors.SchemaErrors):
        validate_preprocessed(processed)
