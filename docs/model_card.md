# Model Card — Telco Customer Churn Predictor

**Versão**: 1.0.0  
**Data**: 2026-06-17  
**Mantido por**: Equipe FIAP Telco Challenge  
**Contato**: adriano731@gmail.com

---

## Sumário

| Campo | Valor |
|---|---|
| Tarefa | Classificação binária (churn / não churn) |
| Modelo | MLP (Multi-Layer Perceptron) — PyTorch |
| Dataset | IBM Telco Customer Churn (7.043 clientes) |
| Threshold de produção | 0.17 |
| AUC-ROC (test) | **0.843** |
| F1-Score (test) | **0.620** |
| PR-AUC (test) | **0.612** |
| Servido via | FastAPI — `POST /predict` |

---

## 1. Descrição do Modelo

### 1.1 Objetivo

Prever a probabilidade de cancelamento (churn) de clientes de uma operadora de telecomunicações antes que o evento ocorra, permitindo que a equipe de retenção intervenha proativamente durante interações de atendimento ou campanhas direcionadas.

### 1.2 Tipo de Modelo

**Multi-Layer Perceptron (MLP)** — rede neural binária implementada em PyTorch.

| Componente | Configuração |
|---|---|
| Camada de entrada | 30 features (pós feature engineering) |
| Camadas ocultas | [128, 64, 32] neurônios |
| Camada de saída | 1 neurônio (logit) |
| Ativação | ReLU + BatchNormalization + Dropout (0.2) |
| Inicialização de pesos | Kaiming Uniform |
| Função de perda | BCEWithLogitsLoss (`pos_weight = 2.77`) |
| Otimizador | Adam (lr=1e-3, weight_decay=1e-5) |

### 1.3 Uso Pretendido

**Casos de uso aprovados:**
- Pontuação em tempo real de clientes via API durante atendimento ao cliente
- Pontuação em batch para campanhas de retenção proativas
- Apoio à decisão da equipe de retenção (humano no loop)

**Casos de uso fora do escopo:**
- Decisões automatizadas de rescisão de contrato sem revisão humana
- Uso para discriminação ou decisão adversa sobre clientes
- Aplicação a operadoras de setores diferentes de telecomunicações sem revalidação
- Clientes de regiões ou perfis demográficos substancialmente distintos dos dados de treinamento

---

## 2. Dados de Treinamento

### 2.1 Dataset

| Atributo | Valor |
|---|---|
| Nome | IBM Telco Customer Churn |
| Registros totais | 7.043 clientes |
| Features originais | 21 |
| Features após engenharia | ~30 |
| Variável alvo | `Churn` (0 = permanece, 1 = cancela) |
| Distribuição de classes | 73,5% não-churn (5.174) / 26,5% churn (1.869) |

### 2.2 Divisão Treino / Validação / Teste

| Conjunto | Proporção | Amostras | % Churn |
|---|---|---|---|
| Treino | 70% | 4.930 | 25,65% |
| Validação | 15% | 1.057 | 25,62% |
| Teste | 15% | 1.056 | 26,07% |

Divisão estratificada pela variável alvo (`seed=42`) para garantir distribuição proporcional das classes.

### 2.3 Features

**Demográficas:**
- `gender` — gênero do cliente
- `SeniorCitizen` — flag binária (idoso)
- `Partner` — possui cônjuge/parceiro
- `Dependents` — possui dependentes

**Serviços contratados:**
- `PhoneService`, `MultipleLines`
- `InternetService` (DSL / Fiber optic / No)
- `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`
- `StreamingTV`, `StreamingMovies`

**Contrato e pagamento:**
- `Contract` (Month-to-month / One year / Two year)
- `PaperlessBilling`
- `PaymentMethod`

**Financeiras e temporais:**
- `MonthlyCharges` — valor mensal cobrado
- `tenure` — meses como cliente
- `TotalCharges` — removida do modelo (alta correlação com `tenure` e `MonthlyCharges`, risco de multicolinearidade)

**Features derivadas (feature engineering):**
- `tenure_group` — `tenure` discretizado em faixas [0-12m, 13-24m, 25-48m, 49-72m, 73+m]
- `service_count` — contagem de serviços adicionais contratados

### 2.4 Pré-processamento

1. Conversão de `TotalCharges` de string para numérico (valores nulos → 0 para clientes com `tenure=0`)
2. Tratamento de `SeniorCitizen` como variável categórica (regra de negócio)
3. Mapeamento do alvo: `'No'` → 0, `'Yes'` → 1
4. **StandardScaler** em `MonthlyCharges` e `service_count`
5. **OneHotEncoder** com `drop='first'` em todas as 15 variáveis categóricas
6. Validação de schema via **Pandera** (`RAW_SCHEMA` e `PREPROCESSED_SCHEMA`)

> **Prevenção de data leakage**: o `ColumnTransformer` é ajustado exclusivamente no conjunto de treino; validação e teste utilizam apenas `transform`.

---

## 3. Performance

### 3.1 Métricas no Conjunto de Teste

| Métrica | Valor |
|---|---|
| Accuracy | 73,13% |
| F1-Score | 0,620 |
| AUC-ROC | **0,843** |
| PR-AUC | 0,612 |
| Recall (threshold 0.17) | 98,9% |
| Precision (threshold 0.17) | 36,84% |

> Métricas de accuracy e F1 calculadas no threshold de produção (0.17). AUC-ROC e PR-AUC são independentes de threshold.

### 3.2 Comparação com Modelos Baseline

| Modelo | Accuracy | F1-Score | AUC-ROC | PR-AUC |
|---|---|---|---|---|
| Dummy (baseline) | 62,2% | 0,290 | 0,516 | 0,272 |
| Logistic Regression | 73,74% | 0,614 | 0,837 | 0,613 |
| Decision Tree | 71,7% | 0,569 | 0,757 | 0,463 |
| Random Forest | 75,68% | 0,615 | 0,834 | 0,598 |
| Gradient Boosting | 77,7% | 0,546 | 0,829 | 0,598 |
| **MLP (produção)** | **73,13%** | **0,620** | **0,843** | **0,612** |

**Critério de seleção**: O MLP foi selecionado por apresentar o maior **AUC-ROC (0.843)** e o maior **F1-Score (0.620)** entre todos os modelos avaliados. Para problemas com classes desbalanceadas, AUC-ROC e F1 são métricas prioritárias em relação à accuracy.

### 3.3 Configuração de Treinamento

| Parâmetro | Valor |
|---|---|
| Batch size | 64 |
| Máximo de épocas | 100 |
| Early stopping (paciência) | 10 épocas |
| Early stopping (min_delta) | 1e-4 |
| Métrica monitorada | `val_loss` |
| Épocas típicas até convergência | ~30–40 |

---

## 4. Threshold e Análise de Custo

### 4.1 Threshold de Produção: 0.17

O threshold padrão de 0.5 foi substituído por **0.17** com base em análise de custo-benefício:

| Tipo de erro | Custo unitário | Impacto |
|---|---|---|
| Falso Negativo (FN) | 1.000 u.m. | Cliente churna sem intervenção (LTV perdido) |
| Falso Positivo (FP) | 50 u.m. | Ação de retenção desnecessária (desconto/ligação) |
| **Razão FN/FP** | **20×** | Penaliza fortemente não-detecção |

| Threshold | Recall | Precision | Custo total | Benefício líquido |
|---|---|---|---|---|
| 0.50 | ~72% | ~62% | 59.800 u.m. | — |
| **0.17** | **98,9%** | **36,84%** | **26.750 u.m.** | **253.250 u.m.** |

> O threshold de 0.17 reduz o custo total em **127%** em relação ao threshold padrão.

**Atenção**: qualquer alteração nos custos de FN/FP exige reavaliação do threshold ótimo antes da implantação.

---

## 5. Limitações

### 5.1 Dados e Representatividade

- **Escopo geográfico e temporal restrito**: o dataset representa uma operadora específica em um período específico. O modelo pode não generalizar para outras operadoras, regiões ou janelas de tempo sem revalidação.
- **Dataset estático**: treinado sobre snapshot histórico. Padrões de churn evoluem — o modelo requer retreinamento periódico (mensal recomendado ou quando PSI > 0.25 for detectado).
- **Ausência de variáveis comportamentais**: logs de interações, histórico de reclamações, e padrões de uso granulares não estão presentes no dataset e poderiam melhorar o desempenho.

### 5.2 Modelo e Técnicas

- **Balanceamento de classes parcial**: a distribuição 73,5%/26,5% é tratada via `pos_weight` na função de perda, mas precisão no segmento minoritário permanece limitada (~37% no threshold de produção).
- **Dependência temporal implícita**: clientes com `tenure` baixo (<12 meses) apresentam churn concentrado nesse período. O modelo pode superestimar risco para novos clientes por razões que não se sustentam no longo prazo.
- **Multicolinearidade residual**: `TotalCharges` foi removida, mas correlações entre serviços contratados permanecem (ex.: clientes com fibra ótica tendem a contratar mais serviços adicionais).
- **Interpretabilidade limitada**: redes neurais são caixas-pretas. Não há feature importance diretamente derivável. Para decisões que exigem explicabilidade regulatória, considere modelos lineares alternativos documentados nos baselines.

### 5.3 Operacional

- **Latência fria**: carregamento do modelo no startup da API (~1–2 segundos); sem cache de inferência.
- **Sensibilidade ao threshold**: threshold calibrado para a estrutura de custo atual. Mudanças nos processos de retenção (custo de ações, valor de LTV) invalidam a calibração.
- **Ausência de retreinamento automatizado**: pipeline de retreinamento deve ser acionado manualmente ou por alertas de drift.

---

## 6. Vieses e Considerações de Equidade

### 6.1 Variáveis Sensíveis Presentes no Dataset

| Variável | Tipo | Observação |
|---|---|---|
| `gender` | Sensível (gênero) | Distribuição ~51% feminino, 49% masculino. Impacto no modelo não auditado por subgrupo. |
| `SeniorCitizen` | Sensível (idade) | Flag binária. Idosos tendem a ter padrão de churn distinto. Impacto diferencial não medido. |
| `PaymentMethod` | Proxy econômico | Método de pagamento correlaciona com faixa de renda. Não é variável protegida, mas merece monitoramento. |
| `MonthlyCharges` | Proxy econômico | Pode atuar como proxy de nível socioeconômico. |

### 6.2 Auditorias Realizadas

- Divisão estratificada preserva proporção de churn em todos os conjuntos — não há auditoria explícita de equidade por subgrupo demográfico.
- Monitoramento de PSI em produção cobre `tenure`, `MonthlyCharges`, `Contract`, `InternetService`, `PaymentMethod`, `TechSupport` e `OnlineSecurity`.

### 6.3 Lacunas de Equidade Identificadas

- **Sem métricas de equidade por subgrupo** (paridade demográfica, odds equalizadas, etc.) documentadas no treinamento.
- **Recomendação**: antes de escalar o uso do modelo para decisões de alto impacto, conduzir auditoria de equidade com equalized odds entre grupos de `gender` e `SeniorCitizen`.

---

## 7. Cenários de Falha

### 7.1 Falsos Negativos em Massa (Recall cai abruptamente)

**Trigger**: mudança estrutural no perfil de clientes (nova oferta de concorrente, sazonalidade atípica, campanha agressiva de portabilidade).

**Sintoma**: taxa de churn real aumenta enquanto o modelo mantém scores baixos.

**Detecção**: monitorar churn rate real semanal vs. taxa de alertas gerados pelo modelo.

**Ação**: ativar retreinamento imediato; escalar threshold temporariamente para 0.10 até novo modelo estar disponível.

### 7.2 Deriva de Dados (Feature Drift)

**Trigger**: mudança no mix de planos, novo produto, alteração no sistema de cobrança.

**Sintoma**: PSI > 0.25 em `MonthlyCharges`, `Contract` ou `InternetService`.

**Detecção**: pipeline de monitoramento semanal com alertas automáticos.

**Ação**: investigar feature com PSI elevado; retreinar se confirmado drift de conceito.

### 7.3 Degradação Silenciosa de Performance

**Trigger**: variações graduais no comportamento dos clientes sem evento abrupto.

**Sintoma**: AUC-ROC < 0.75 ou F1 < 0.50 no conjunto de validação contínua.

**Detecção**: avaliação mensal com dados rotulados do mês anterior.

**Ação**: retreinamento com dados do último trimestre; comparar com modelo em produção antes de implantar.

### 7.4 Falha na Pipeline de Preprocessamento

**Trigger**: mudança de schema na fonte de dados (nova coluna obrigatória, renomeação de campo).

**Sintoma**: erro 500 na API, logs de `PanderaSchemaError` ou `KeyError`.

**Detecção**: validação de schema via Pandera no pré-processamento; alertas de erro na API.

**Ação**: validar schema de entrada; atualizar `RAW_SCHEMA` e `PREPROCESSED_SCHEMA` conforme necessário.

### 7.5 Degradação por Distribuição de Novos Clientes

**Trigger**: campanha de aquisição que traz perfil de clientes novo (ex.: expansão geográfica, segmento premium).

**Sintoma**: alta taxa de falsos positivos em novos clientes (retention team sobrecarregada com alarmes).

**Detecção**: analisar precision por coorte de aquisição mensalmente.

**Ação**: segmentar modelo por coorte ou retreinar com dados do novo segmento representado.

