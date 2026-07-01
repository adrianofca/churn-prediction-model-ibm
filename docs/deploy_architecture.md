# Arquitetura de Deploy

## Estratégia escolhida: Inferência em Tempo Real (Online Inference)

O modelo é servido via **API REST com FastAPI**, respondendo a requisições individuais com latência média abaixo de 500ms.

## Por que tempo real e não batch?

| Critério | Batch (Processamento em Lote) | **Real-time / Online (escolhido)** |
|---|---|---|
| **Latência** | Alta — processa agendado (ex: diário) | Baixa — resposta imediata por requisição |
| **Caso de uso** | Scoring periódico de toda a base | Scoring sob demanda, cliente a cliente |
| **Integração** | Arquivo/banco de dados de saída | API REST consumível por qualquer sistema |
| **Adequação ao problema** | ❌ Não — equipe de retenção precisa agir no momento do contato | ✅ Sim — consulta imediata durante atendimento ao cliente |

## Justificativa de negócio

O problema de churn em telecomunicações exige **intervenção no momento certo**: quando um atendente está em contato com o cliente, ou quando o sistema de CRM detecta uma ação de risco (ex: cliente acessando página de cancelamento). Processar em batch diário significaria agir tarde demais — o cliente já teria cancelado.

A arquitetura escolhida permite que qualquer sistema da operadora (CRM, central de atendimento, app) consulte a API em tempo real e tome ação imediata.

## Fluxo de inferência

```
Sistema externo         API FastAPI              Modelo
(CRM / Atendimento)
        │                    │                      │
        │── POST /predict ──>│                      │
        │   {dados cliente}  │── transformer.pkl ──>│
        │                    │   (pré-processamento) │
        │                    │<── MLP (PyTorch) ────│
        │<── {probability,   │   predict_proba()    │
        │     prediction} ───│                      │
        │                    │                      │
```

## Limitações desta arquitetura

- **Escalabilidade**: para volumes muito altos de requisições simultâneas, seria necessário adicionar múltiplas instâncias da API com load balancer.
- **Latência de cold start**: o modelo é carregado em memória na inicialização da API — se o container reiniciar, há um breve delay inicial.
- **Sem cache**: cada requisição executa a inferência completa. Para perfis de clientes repetidos, um cache (ex: Redis) reduziria latência e custo computacional.
