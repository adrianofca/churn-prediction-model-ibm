FROM python:3.11-slim

WORKDIR /app

# Torch CPU-only: o wheel padrão do PyPI inclui bibliotecas CUDA (~2GB) que não
# servem para nada em produção sem GPU. O índice da CPU reduz a imagem drasticamente.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Dependências de runtime da API (treino/EDA usam outras libs que não entram aqui)
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    pydantic \
    pandas \
    numpy \
    scikit-learn==1.8.0 \
    joblib \
    mlflow \
    pandera \
    python-json-logger

COPY api/ api/
COPY src/ src/
COPY models/ models/

EXPOSE 8080

CMD exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080}
