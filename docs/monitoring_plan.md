# Plano de Monitoramento do Modelo de Predição de Churn

**Entrega Acadêmica — Trabalho de Conclusão** **Curso:** Pós Tech - Machine Learning Engineering - MLET  
**Grupo:** 26  
**Contexto:** Engenharia de Machine Learning em Produção (MLOps)  

---

## 1. Objetivo
Garantir que o modelo de Machine Learning para previsão de *churn* mantenha níveis adequados de desempenho, disponibilidade e confiabilidade em ambiente produtivo, permitindo identificar rapidamente anomalias, desvios estatísticos e executar ações corretivas estruturadas.

---

## 2. Métricas Monitoradas

### 2.1 Métricas Operacionais da API
Indicadores de infraestrutura dedicados a avaliar a integridade da aplicação FastAPI que serve as predições do modelo.

| Métrica | Objetivo Principal | Meta Operacional |
| :--- | :--- | :--- |
| **Disponibilidade da API** | Garantir que o serviço permaneça ativo e acessível | > 99,5% (*Uptime*) |
| **Latência Média** | Monitorar o tempo de resposta nas requisições de predição | < 500 ms |
| **Taxa de Erros HTTP** | Identificar falhas na aplicação ou infraestrutura (erros 4xx/5xx) | < 1% |
| **Número de Requisições** | Acompanhar a volumetria e a curva de utilização do serviço | Monitoramento contínuo |

### 2.2 Métricas de Desempenho do Modelo
Métricas estatísticas calculadas periodicamente com base no acoplamento das predições efetuadas com os reais desfechos observados no negócio (*ground truth*).

| Métrica | Objetivo Analítico | Valor Mínimo Aceitável |
| :--- | :--- | :--- |
| **Accuracy** | Avaliar a taxa global de acertos do modelo | ≥ 70% |
| **F1-Score** | Garantir o balanceamento ideal entre Precisão e *Recall* | ≥ 0,60 |
| **ROC-AUC** | Medir a capacidade de separação das classes (*Churn* vs. Não-*Churn*) | ≥ 0,80 |
| **PR-AUC** | Avaliar o desempenho em classes altamente desbalanceadas | ≥ 0,60 |

> 💡 **Valores de referência obtidos durante os experimentos (Baseline):** > * **Accuracy:** 73,1%  
> * **F1-Score:** 0,62  
> * **ROC-AUC:** 0,84  
> * **PR-AUC:** 0,61  

### 2.3 Monitoramento dos Dados (Data Drift)
Acompanhamento rigoroso de possíveis alterações na natureza estatística das variáveis preditoras originais em ambiente produtivo.

#### Variáveis Críticas Monitoradas:
* `tenure`
* `MonthlyCharges`
* `Contract`
* `InternetService`
* `PaymentMethod`
* `TechSupport`
* `OnlineSecurity`

#### Indicadores Adotados:
* **PSI** (*Population Stability Index*);
* Mudança na distribuição estatística das variáveis;
* Alteração significativa da proporção de clientes classificados em *churn*.

#### Faixas de Controle do PSI:
| Faixa de Controle (PSI) | Interpretação Técnica | Ação Recomendada |
| :--- | :--- | :--- |
| **< 0,1** | Sem drift detectável | Nenhuma ação necessária; manter fluxo básico. |
| **0,1 a 0,25** | Atenção / Mudança moderada | Investigar variáveis e planejar validações. |
| **> 0,25** | Drift significativo constatado | Gatilho obrigatório para execução do pipeline de retreinamento. |

---

## 3. Matriz de Alertas

### 🚨 Alerta Crítico
* **Condição:** API indisponível; Taxa de erro superior a 5%; Latência acima de 1 segundo; ROC-AUC inferior a 0,75; ou PSI superior a 0,25.
* **Ação:** Notificação imediata via canais automatizados para a equipe responsável e início instantâneo do procedimento de resposta a incidentes.

### ⚠️ Alerta Moderado
* **Condição:** F1-Score inferior a 0,60; Accuracy inferior a 70%; PSI entre 0,10 e 0,25; ou aumento gradual e consistente da latência.
* **Ação:** Investigação aprofundada do comportamento analítico do modelo e avaliação da necessidade de reprocessamento/retreinamento.

### ℹ️ Alerta Informativo
* **Condição:** Crescimento orgânico no número de requisições ou alterações residuais na distribuição dos dados.
* **Ação:** Apenas registro em log interno e acompanhamento nos relatórios mensais de capacidade.

---

## 4. Frequência do Monitoramento

| Item / Componente Monitorado | Frequência / Periodicidade | Mecanismo de Captura |
| :--- | :--- | :--- |
| Disponibilidade da API | `Tempo Real` | Prometheus / Grafana |
| Latência Média | `Tempo Real` | Prometheus / Grafana |
| Taxa de Erros HTTP | `Tempo Real` | Mapeamento de *status code* do FastAPI |
| Métricas do Modelo (Accuracy, F1, ROC-AUC) | `Semanal` | Job assíncrono acoplado às *ground truth* |
| Análise de Data Drift (Cálculo do PSI) | `Semanal` | Pipeline via Evidently AI |
| Retraining do modelo | `Mensal ou sob demanda` | Gatilho automático ou rotina cíclica ordinária |

---

## 5. Playbook de Resposta a Incidentes (Runbooks)

### Cenário 1 – API indisponível ou instável
1. Verificar logs em tempo real da aplicação FastAPI buscando exceções impeditivas.
2. Confirmar disponibilidade física ou em nuvem do servidor de aplicação (Kubernetes/VM).
3. Executar reinicialização (*restart*) controlada do serviço afetado.
4. Validar estabilidade estrutural do endpoint nativo de checagem: `/health`.
5. Executar testes sintéticos automatizados de previsão em ambiente produtivo.
* ⏱️ **Tempo máximo de recuperação alvo (RTO):** 30 minutos.

### Cenário 2 – Degradação das métricas analíticas do modelo
1. Avaliar detalhadamente as métricas de performance das janelas de dias mais recentes.
2. Verificar a existência concomitante de *data drift*.
3. Comparar distribuições dos dados atuais com os dados de treinamento original.
4. Executar pipeline automatizado de retreinamento.
5. Validar exaustivamente o desempenho do novo modelo candidato.
6. Publicar nova versão utilizando práticas consolidadas de versionamento (ex: *MLflow Model Registry*).
* ⏱️ **Tempo máximo de recuperação alvo (RTO):** 24 horas.

### Cenário 3 – Data Drift identificado
1. Medir e isolar o PSI individual das variáveis monitoradas.
2. Identificar quais atributos foram mais severamente impactados pela mudança comportamental.
3. Coletar novos dados históricos atualizados e higienizados da operação.
4. Executar novo treinamento utilizando a massa amostral recente.
5. Conduzir testes de validação cruzada avaliando o ganho de performance entre as versões.
6. Efetuar a promoção transparente da nova versão otimizada para o ambiente produtivo.

### Cenário 4 – Aumento anômalo da latência
1. Verificar consumo instantâneo de infraestrutura (utilização volumétrica de CPU e Memória RAM).
2. Avaliar volumetria total de requisições concorrentes recebidas pela API.
3. Revisar logs de aplicação em busca de gargalos lógicos ou concorrência de I/O.
4. Acionar políticas de *autoscaling* horizontal ou realizar o upgrade vertical temporário de recursos do servidor.

---

## 6. Ferramentas Recomendadas
* **MLflow:** Rastreamento minucioso de parâmetros, experimentos e versionamento centralizado de modelos (*Model Registry*).
* **Prometheus:** Coleta ativa de séries temporais e telemetria operacional da aplicação.
* **Grafana:** Centralização visual através de dashboards analíticos interativos e sistemas de alerta.
* **Evidently AI:** Biblioteca especializada para cálculo automatizado de *data drift* e geração de relatórios de estabilidade.
* **FastAPI Logs:** Estruturação padronizada de logs de eventos e de erros de requisições.
* **GitHub / GitLab:** Controle de versão do código do pipeline produtivo e da infraestrutura (*GitOps*).
* **Canais Integrados:** Disparos nativos de alertas para e-mail institucional e Microsoft Teams / Slack.

---

## 7. Estratégia de Manutenção de Longo Prazo

### 7.1 Manutenção Preventiva
* Inspeção semanal e consolidação periódica de relatórios sobre o comportamento estatístico geral;
* Auditoria programada sobre as transformações de dados em produção e checagem de integridade de *data-stores*;
* Rotina automatizada diária de cópias de segurança (*backup*) de todos os pesos e metadados dos modelos em produção.

### 7.2 Manutenção Corretiva
* Adoção de ciclos reativos de readequação estatística (retreinamento) frente à queda validada de métricas *core*;
* Ajustes pontuais no código do pipeline para tratamento de anomalias imprevistas na chegada dos dados brutos;
* Publicação ágil de correções arquiteturais (*hotfixes*) com incrementos controlados nas tags semânticas das versões.

---

## 8. Indicadores de Sucesso (KPIs do Monitoramento)
* Manutenção da disponibilidade da API em patamar rigorosamente superior a **99,5%** ao mês;
* Latência de entrega de score preditivo mantida abaixo de **500 ms** no 95º percentil (p95);
* Preservação do desempenho analítico do modelo com **ROC-AUC > 0,80** e **F1-Score > 0,60**;
* Cumprimento estrito do tempo de mitigação (RTO) de falhas críticas de infraestrutura em até **30 minutos**;
* Atualização controlada e transparente do modelo produtivo sempre que anomalias persistentes de *drift* forem constatadas.