import logging

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Executa a limpeza estrutural e o tratamento das variáveis do dataset Telco Churn.
    Garante a conversão de tipos, tratamento de nulos por regra de negócio e codificação do target.

    Args:
        df (pd.DataFrame): DataFrame original carregado.

    Returns:
        pd.DataFrame: Novo DataFrame totalmente limpo e pronto para análise.
    """
    df_clean = df.copy()

    # 1. Converter TotalCharges para numérico
    logging.info("A converter 'TotalCharges' para tipo numérico...")
    df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")

    # 2. Tratar nulos em TotalCharges (clientes novos com tenure zero)
    null_count = df_clean["TotalCharges"].isnull().sum()
    if null_count > 0:
        logging.info(f"Foram encontrados {null_count} valores nulos em 'TotalCharges'. A imputar valor 0.")
        df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(0.0)

    # 3. Processar e mapear a variável Target
    if "Churn" in df_clean.columns:
        logging.info("A mapear e a renomear a variável alvo 'Churn' para 'target'...")
        df_clean.rename(columns={"Churn": "target"}, inplace=True)
        df_clean["target"] = df_clean["target"].map({"No": 0, "Yes": 1})
    elif "churn" in df_clean.columns:
        df_clean.rename(columns={"churn": "target"}, inplace=True)
        df_clean["target"] = df_clean["target"].map({"No": 0, "Yes": 1})

    # 4. Correção de Schema: SeniorCitizen é categórica (Regra de Negócio)
    if "SeniorCitizen" in df_clean.columns:
        logging.info("A converter 'SeniorCitizen' para tipo string (categórica)...")
        df_clean["SeniorCitizen"] = df_clean["SeniorCitizen"].astype(str)

    logging.info(f"Pré-processamento concluído com sucesso. Dimensões: {df_clean.shape}")
    return df_clean
