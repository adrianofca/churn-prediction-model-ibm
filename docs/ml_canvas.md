# Machine Learning Canvas

**Designed for:** Tech Challenge Fase 01
**Designed by:** Grupo 26
**Date:** 15/05/2026
**Iteration:** 1

---

## Tarefas de Predição

**Qual é o tipo de tarefa?**
Classificação binária via rede neural.

**Sobre qual entidade as predições são feitas?**
Serão avaliados os clientes ativos.

**Quais são os possíveis resultados a serem previstos?**
Risco de cancelamento (Churn / Classe 1) ou Permanência (Retido / Classe 0).

**Quando os resultados são observados?**
No momento do encerramento do contrato ou quando o cliente formaliza o cancelamento da assinatura.

---

## Decisões

**Como as predições são transformadas em recomendações ou decisões acionáveis para o usuário final?**
As predições serão servidas por meio de uma API de inferência construída com FastAPI, utilizando um endpoint específico (como `/predict`).

---

## Proposta de Valor

**Quem é o beneficiário final e quais pontos de dor específicos são resolvidos?**
A diretoria da operadora de telecomunicações. A dor principal resolvida é a perda de clientes em ritmo acelerado e, consequentemente, perda financeira.

**Como a solução de ML se integrará ao fluxo de trabalho deles, e através de quais interfaces de usuário?**
A área de negócios consumirá as predições da API para classificar clientes de risco, permitindo intervenções direcionadas.

---

## Coleta de Dados

**Como o conjunto inicial de entidades e resultados é obtido?**
Dataset Telco Customer Churn (IBM).

**Quais estratégias estão em vigor para atualizar os dados continuamente, controlando custos e mantendo a atualização?**
Será construído um pipeline reprodutível (usando pipelines do Scikit-Learn e transformadores customizados) organizado em módulos na pasta `src/`.

---

## Fonte de Dados

**Onde podemos obter dados sobre entidades e resultados observados?**
No dataset fornecido (IBM Telco Customer Churn), contendo 7.043 registros (observações), com 21 colunas.

---

## Simulação de Impacto

**Quais são os valores de custo/ganho para decisões (in)corretas?**
Análise do trade-off de custo (peso do falso positivo versus o impacto do falso negativo), focada na métrica de negócio de custo do churn evitado:
- **Impacto de Falsos Positivos**: a empresa pode oferecer benefícios para clientes que não cancelariam.
- **Impacto de Falsos Negativos**: clientes realmente propensos ao churn podem cancelar sem ação preventiva.

**Quais dados são usados para simular o impacto antes da implantação?**
Divisões do dataset avaliadas por meio de validação cruzada estratificada e utilizando seeds fixados para garantir a reprodutibilidade.

**Quais são os critérios para implantação?**
A Rede Neural (MLP) deve superar os baselines em um conjunto de 4 métricas. Além disso, a aplicação deve passar em testes automatizados (3 tipos: smoke test, schema, API) e possuir código validado por linting com o Ruff, sem acusar erros.

- **Qualidade do modelo**: Recall ≥ 75%, F1-Score ≥ 70%, ROC-AUC ≥ 0.80, baseline (LR simples) superado.
- **Performance da API**: latência p95 ≤ 500ms, uptime ≥ 99%, throughput mínimo de 10 req/s.

---

## Geração de Predições

**As predições são feitas em lote (batch) ou em tempo real?**
API — a arquitetura final (batch vs. real-time) e suas justificativas estão formalmente documentadas em [`docs/deploy_architecture.md`](deploy_architecture.md).

**Com qual frequência?**
Conforme a demanda for enviada ao endpoint de predição.

**Quais recursos computacionais são usados?**
Infraestrutura em nuvem, caso a equipe opte pela entrega opcional de deploy da API em produção utilizando AWS, Azure ou GCP.

---

## Monitoramento

**Quais métricas e KPIs são usados para rastrear o impacto da solução de ML uma vez implantada?**
Métricas técnicas (AUC-ROC, PR-AUC, F1) aliadas à métrica primária de negócio: o custo de churn evitado em R$ (ticket médio).

**Com que frequência eles devem ser revisados?**
O monitoramento em ambiente operacional (API) acontecerá de forma contínua, valendo-se da implementação de logging estruturado e de um middleware de latência incorporados na FastAPI. Detalhamento completo em [`docs/monitoring_plan.md`](monitoring_plan.md).

---

## Construção de Modelos

**Quantos modelos são necessários em produção?**
Uma Rede Neural (MLP) desenvolvida em PyTorch, que deve ser comparada com baselines (DummyClassifier e Regressão Logística).

**Quando eles devem ser atualizados?**
Será definida por um plano de monitoramento contendo alertas e um playbook de resposta predefinido.

**Quanto tempo está disponível para isso?**
O processo de treinamento será otimizado utilizando técnicas como early stopping e batching no loop de treinamento.

**Quais recursos computacionais são usados?**
Rastreamento de experimentos (parâmetros, métricas e artefatos) utilizando o MLflow, suportado por um ambiente estruturado com o `pyproject.toml`.

---

## Features / Atributos

**Quais representações são usadas para entidades no momento da predição?**
Vetores padronizados que passarão por validação rigorosa de schema utilizando as bibliotecas Pydantic na API e Pandera no pipeline de dados.

**Quais agregações ou transformações são aplicadas às fontes de dados brutas?**
Aplicações de pré-processamento acopladas diretamente nos pipelines do Scikit-Learn para padronizar formatos antes de serem consumidos pela Rede Neural.

---
