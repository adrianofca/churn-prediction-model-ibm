# Análise de Custo — Trade-off FP vs FN

## Premissas de custo

| Evento | Custo unitário | Justificativa |
|--------|---------------|---------------|
| Falso Negativo (FN) | 1000 u.m. | LTV perdido: cliente churna sem intervenção |
| Falso Positivo (FP) | 50 u.m. | Ação de retenção desnecessária (desconto/ligação) |
| **Ratio FN/FP** | **20x** | Perder um cliente custa 20x mais que reter desnecessariamente |

## Thresholds ótimos por critério

| Critério | Threshold | Recall | Precision | FN | FP | Custo Total | Benefício Líquido |
|----------|-----------|--------|-----------|----|----|-------------|-------------------|
| min_cost | 0.17 | 0.989 | 0.368 | 3 | 475 | 26,750 | 253,250 |
| max_benefit | 0.17 | 0.989 | 0.368 | 3 | 475 | 26,750 | 253,250 |
| max_recall | 0.01 | 1.000 | 0.265 | 0 | 777 | 38,850 | 241,150 |
| max_f_beta | 0.29 | 0.968 | 0.411 | 9 | 389 | 28,450 | 251,550 |
| slo_min | 0.57 | 0.754 | 0.537 | 69 | 182 | 78,100 | 201,900 |

## Interpretação

Com ratio FN/FP = 20x, o threshold ótimo por custo tende a ser **baixo** (0.2–0.4),
pois é muito mais barato acionar 20 clientes desnecessariamente do que perder 1 cliente real.

O threshold padrão de 0.5 não é ótimo para churn — ele equilibra FP e FN como se
tivessem o mesmo custo, o que contradiz a realidade do negócio.

## Recomendação operacional

Usar o threshold de `max_benefit` como ponto de partida para produção,
revisando com a equipe de retenção o custo real de cada ação.