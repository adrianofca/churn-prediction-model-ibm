import logging

import pandas as pd

# Configuração básica do logger para monitorização do pipeline
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_raw_data(url: str) -> pd.DataFrame:
    """
    Carrega os dados brutos a partir de uma URL ou caminho de ficheiro local.
    Remove também espaços em branco dos nomes das colunas.

    Args:
        url (str): Link ou caminho do ficheiro CSV.

    Returns:
        pd.DataFrame: DataFrame do Pandas com os dados carregados e colunas normalizadas.
    """
    try:
        logging.info(f"A carregar dados a partir de: {url}")
        df = pd.read_csv(url)

        # Correção inicial: remove espaços ocultos nos nomes das colunas (Célula 4)
        df.columns = df.columns.str.strip()

        logging.info(f"Dados carregados com sucesso. Dimensões: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Erro crítico ao carregar os dados de {url}: {str(e)}")
        raise e
