"""

Este módulo é o ponto de entrada para a execução do projeto Telco Churn.
Ele integra as etapas de pré-processamento, treinamento e avaliação dos modelos,
garantindo que todo o pipeline funcione de forma coesa.
"""

__version__ = "0.1.0"
__author__ = "Grupo Tech Challenge - FIAP Pós-Graduação em Machine Learning Engineering"

from .data_loader import load_raw_data
from .features import build_feature_transformer
from .preprocessing import preprocess_data
from .schema import validate_preprocessed, validate_raw

__all__ = [
    "__version__",
    "__author__",
    "load_raw_data",
    "preprocess_data",
    "build_feature_transformer",
    "validate_raw",
    "validate_preprocessed",
]
