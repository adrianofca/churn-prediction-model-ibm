# Telco Churn Prediction — Tech Challenge Fase 1

> Pipeline completo de Machine Learning para previsão de churn de clientes de telecomunicações, do dado bruto ao modelo servido via API REST.

---

## Contexto de Negócio

Uma operadora de telecomunicações enfrenta perda acelerada de clientes. Este projeto constrói um sistema preditivo que classifica clientes com risco de cancelamento, permitindo à equipe de retenção priorizar ações proativas antes que o churn aconteça.

- **Dataset**: IBM Telco Customer Churn — 7.043 clientes, 20 features
- **Problema**: Classificação binária (`Churn = 1` / `Não Churn = 0`)
- **Desbalanceamento**: 73,5% não-churn vs 26,5% churn
- **Métrica principal**: AUC-ROC e F1-Score (classe minoritária)
- **Modelo em produção**: MLP (Multi-Layer Perceptron) com PyTorch — AUC-ROC 0.843, F1 0.620

---

## Estrutura do Repositório

```
telco_fiap_01/
├── api/                              # API FastAPI de inferência
│   ├── __init__.py
│   ├── logging_config.py             # Logging estruturado em JSON
│   ├── main.py                       # Endpoints /predict e /health
│   ├── middleware.py                 # Middleware de latência
│   ├── model_loader.py               # Carregamento do modelo e predição
│   └── schemas.py                    # Validação de entrada/saída (Pydantic)
├── data/
│   └── telco_churn_preprocessed.csv  # Dataset pré-processado
├── docs/
│   ├── deploy_architecture.md        # Arquitetura de deploy (batch vs. real-time) + justificativa
│   ├── model_card.md                 # Model Card completo do modelo
│   └── monitoring_plan.md            # Plano de monitoramento em produção
├── models/                           # Artefatos de todos os modelos treinados
│   ├── baseline_dummy_model.joblib
│   ├── baseline_dummy_model_metrics.json
│   ├── baseline_logistic_regression.joblib
│   ├── baseline_logistic_regression_metrics.json
│   ├── decision_tree.joblib
│   ├── decision_tree_metrics.json
│   ├── gradient_boosting.joblib
│   ├── gradient_boosting_metrics.json
│   ├── mlp_model.pt                  # Modelo MLP treinado (PyTorch)
│   ├── mlp_model_metrics.json        # Métricas do modelo em produção
│   ├── random_forest.joblib
│   ├── random_forest_metrics.json
│   └── transformer.pkl               # Pipeline de pré-processamento
├── notebooks/
│   └── DEA.ipynb                     # Análise exploratória de dados (EDA)
├── reports/
│   ├── cost_analysis.json            # Dados brutos da análise de custo
│   └── cost_analysis.md              # Análise de custo FN vs FP e threshold
├── src/                              # Código modularizado
│   ├── data/                         # Scripts de pré-processamento
│   ├── evaluation/                   # Métricas e avaliação de modelos
│   ├── models/                       # Treinamento e persistência
│   └── visualization/                # Gráficos e relatórios
├── tests/                            # Testes automatizados (pytest)
├── .dockerignore
├── .gitignore
├── .python-version
├── Dockerfile                        # Imagem da API para deploy em nuvem
├── Makefile                          # Atalhos de execução
├── pyproject.toml                    # Dependências e configuração
├── requirements.txt
└── uv.lock
```

---

## Pré-requisitos

- Python 3.11+
- pip ou [uv](https://docs.astral.sh/uv/)
- [Docker](https://docs.docker.com/get-docker/) e [gcloud CLI](https://cloud.google.com/sdk/docs/install) — apenas para reproduzir o deploy em nuvem (opcional)

---

## Instalação

### Opção 1 — pip

```bash
# Clone o repositório
git clone https://github.com/charlescoutinho85/telco_fiap_01.git
cd telco_fiap_01

# Crie e ative o ambiente virtual
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
.venv\Scripts\activate           # Windows

# Instale as dependências
pip install -r requirements.txt
```

### Opção 2 — uv (recomendado)

```bash
git clone https://github.com/charlescoutinho85/telco_fiap_01.git
cd telco_fiap_01
uv sync
```

---

## Execução

### 1. Treinar o modelo

```bash
python -m src.models.mlp_trainer
```

O script treina a rede neural MLP, compara com modelos baseline (Dummy, Logistic Regression, Decision Tree, Random Forest, Gradient Boosting), registra todos os experimentos no MLflow e salva os artefatos em `models/`.

### 2. Visualizar experimentos no MLflow

```bash
mlflow ui
```

Acesse [http://localhost:5000](http://localhost:5000) para comparar métricas, parâmetros e artefatos de todos os experimentos registrados.

### 3. Subir a API

```bash
uvicorn api.main:app --reload
```

A API estará disponível em [http://localhost:8000](http://localhost:8000).

### 4. Testar a API interativamente

Acesse [http://localhost:8000/docs](http://localhost:8000/docs) para a interface Swagger gerada automaticamente pelo FastAPI.

### 5. Rodar via Makefile

```bash
make help     # Lista todos os comandos disponíveis
make lint     # Roda o linter (ruff)
make test     # Roda os testes (pytest)
make run      # Sobe a API (uvicorn)
```

---

## Endpoints da API

### `GET /health`

Verifica se a API está no ar.

```bash
curl http://localhost:8000/health
```

**Resposta:**
```json
{ "status": "healthy" }
```

---

### `POST /predict`

Recebe os dados de um cliente e retorna a probabilidade de churn e a classificação.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35,
    "TotalCharges": 844.2
  }'
```

**Resposta:**
```json
{
  "probability": 0.889,
  "prediction": 1
}
```

- `probability`: probabilidade de churn (0.0 a 1.0)
- `prediction`: `1` = cliente em risco de churn / `0` = cliente estável
- **Threshold de decisão**: `0.17` — calibrado pela análise de custo FN/FP (ver `reports/cost_analysis.md`)

---

## Configuração via Variáveis de Ambiente

| Variável | Padrão | Descrição |
|---|---|---|
| `CHURN_THRESHOLD` | `0.17` | Threshold de classificação. Ajuste conforme custo de negócio. |

Exemplo:
```bash
CHURN_THRESHOLD=0.3 uvicorn api.main:app --reload
```

---

## Testes

```bash
# Rodar todos os testes
pytest tests/ -v

# Com cobertura
pytest tests/ -v --cov=src --cov=api
```

Os testes cobrem (`tests/unit/`):
- **Pipeline de dados**: carregamento, pré-processamento, engenharia de features e validação de schema
- **Modelos baseline**: Dummy, Logistic Regression, Decision Tree, Random Forest e Gradient Boosting
- **MLP (PyTorch)**: forward pass, probabilidades e predições do `ChurnMLP`
- **API**: `/health` retorna 200 e `/predict` executa o pipeline real retornando probability e prediction válidos

Todos os testes são smoke tests — verificam que o pipeline executa sem exceções com dados sintéticos, sem validar acurácia.

---

## Resultados — Comparação de Modelos

| Modelo | Accuracy | F1-Score | AUC-ROC | PR-AUC |
|---|---|---|---|---|
| Dummy (baseline) | 0.622 | 0.290 | 0.516 | 0.272 |
| Logistic Regression | 0.737 | 0.614 | 0.837 | 0.613 |
| Decision Tree | 0.717 | 0.569 | 0.757 | 0.463 |
| Random Forest | 0.757 | 0.615 | 0.834 | 0.598 |
| Gradient Boosting | 0.777 | 0.546 | 0.829 | 0.598 |
| **MLP (produção)** | **0.731** | **0.620** | **0.843** | **0.612** |

O MLP foi escolhido para produção por ter o melhor AUC-ROC (0.843) e F1-Score (0.620) — métricas prioritárias dado o desbalanceamento da classe de churn.

---

## Arquitetura de Deploy

**Estratégia escolhida**: Inferência em Tempo Real (Online Inference) via API REST com FastAPI — a equipe de retenção precisa de resposta imediata no momento do atendimento ao cliente, o que descarta processamento em batch.

Comparação detalhada Batch vs. Real-time, justificativa de negócio, fluxo de inferência e limitações: [`docs/deploy_architecture.md`](docs/deploy_architecture.md)

### Deploy em Produção (Google Cloud Run)

A API está publicada em ambiente de nuvem (Google Cloud Run), com endpoint público:

🔗 **https://churn-api-944466888000.us-central1.run.app**

```bash
curl https://churn-api-944466888000.us-central1.run.app/health

curl -X POST https://churn-api-944466888000.us-central1.run.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35,
    "TotalCharges": 844.2
  }'
```

#### Como reproduzir o deploy

A imagem é definida pelo [`Dockerfile`](Dockerfile) (Python 3.11-slim + torch CPU-only para reduzir o tamanho da imagem). Com o [gcloud CLI](https://cloud.google.com/sdk/docs/install) instalado e autenticado:

```bash
gcloud run deploy churn-api --source . --region us-central1 --allow-unauthenticated
```

O comando builda a imagem remotamente (Cloud Build) e publica no Cloud Run, retornando a URL pública. Scale-to-zero está habilitado por padrão — sem tráfego, não há custo.

---

## Decisões Técnicas

| Decisão | Escolha | Justificativa |
|---|---|---|
| Modelo principal | MLP (PyTorch) | Melhor AUC-ROC e F1 entre todos os modelos testados |
| Threshold | 0.17 (não 0.5) | FN custa 20x mais que FP — calibrado pela análise de custo |
| Métrica de avaliação | AUC-ROC + F1 | Accuracy mascara desempenho na classe minoritária |
| Arquitetura de deploy | Online Inference (API REST) | Equipe de retenção precisa de resposta imediata no momento do atendimento |
| Rastreamento | MLflow | Reprodutibilidade e auditoria de todos os experimentos |
| Serving | FastAPI + Pydantic | Baixa latência, validação automática, documentação automática |
| Logging | JSON estruturado | Facilita monitoramento e detecção de drift em produção |

---

## Documentação Adicional

- **ML Canvas** (proposta de valor, stakeholders, métricas de negócio): [`docs/ml_canvas.md`](docs/ml_canvas.md)
- **Arquitetura de Deploy** (batch vs. real-time + justificativa): [`docs/deploy_architecture.md`](docs/deploy_architecture.md)
- **Model Card** (limitações, vieses, cenários de falha): [`docs/model_card.md`](docs/model_card.md)
- **Plano de Monitoramento** (alertas, drift, runbooks): [`docs/monitoring_plan.md`](docs/monitoring_plan.md)
- **Análise de custo FN/FP e threshold**: [`reports/cost_analysis.md`](reports/cost_analysis.md)
- **Experimentos registrados**: `mlflow ui` → [http://localhost:5000](http://localhost:5000)

---

## Equipe

Projeto desenvolvido como Tech Challenge da Fase 1 — Machine Learning Engineering  
PosTech FIAP · Grupo 26 · 2026
