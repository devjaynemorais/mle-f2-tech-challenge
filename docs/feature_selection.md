# Seleção de Features — Racional e Candidatas

## Objetivo

Este documento explica **por que** cada feature usada hoje no treino do
modelo foi escolhida e **como** ela é calculada, além de listar features
**candidatas** que ainda não entraram no pipeline. Para o desempenho atual
do modelo ver [`model_card.md`](model_card.md); para como treinar/ajustar
ver [`training.md`](training.md).

> **Nota histórica:** este documento nasceu como uma análise de gaps entre
> o design planejado (embeddings, negative sampling, features causais,
> avaliação por ranking) e uma implementação inicial que não seguia esse
> design (o que produzia AUC≈0.5). Esses gaps foram todos resolvidos pela
> feature [`specs/001-recommender-quality/`](../specs/001-recommender-quality/spec.md)
> (2026-07-13) — ver Seção 4 para o que mudou. O conteúdo abaixo descreve o
> **estado atual** do pipeline.

---

## 1. Features em produção hoje

Calculadas em `src/data/feature_engineering.py` (agregações causais) e
`src/features/build_features.py` (categoria + split), na ordem definida
pelo contrato único `src/data/feature_contract.py`
(`ID_COLS + CONT_COLS`). Consumidas por `src/training/trainer.py`,
`src/evaluation/evaluate.py` e `src/serving/` — os três leem do mesmo
contrato, então não há descasamento treino/serving (FR-007).

| Feature | Nível | Fórmula (causal / *as-of*) | Racional (fonte) |
|---------|-------|------------------------------|-------------------|
| `user_idx` | identidade | Índice contíguo (0-based) do usuário — **índice de `nn.Embedding`**, não valor numérico | EDA/`preprocessing.md`: *"necessário para embeddings no modelo"* — hoje implementado em `src/models/mlp.py` (`_NCFNet`) |
| `item_idx` | identidade | Índice contíguo (0-based) do item — idem, embedding | Idem |
| `cat_idx` | identidade | Índice contíguo de `categoryid` (extraído em `data/content/` pelo estágio `content`); `-1` (`UNKNOWN_CAT_IDX`) se o item não tem categoria conhecida | `eda.md`: *"categoryid como feature de conteúdo — único metadado estruturado confiável disponível"*; embedding próprio, dá sinal de conteúdo mesmo para itens com pouco histórico |
| `hour` | evento | Hora do timestamp (0–23) | EDA identificou pico de tráfego 10h–16h ([eda.md](eda.md#principais-achados)) |
| `day_of_week` | evento | Dia da semana (0–6) | EDA identificou padrão semanal de volume ([eda.md](eda.md#principais-achados)) |
| `frequency` | usuário, *as-of* | Nº de eventos **anteriores** do usuário (`groupby().cumcount()`) | Distribuição power-law de interações — usuários ativos concentram ~70% dos eventos ([eda.md](eda.md#comportamento)) |
| `engagement_score` | usuário, *as-of* | Soma dos pesos (view=1, addtocart=3, transaction=5) dos eventos **anteriores** do usuário | Pondera intensidade de interação, refletindo a hierarquia de intenção validada no EDA |
| `recency_days` | usuário, *as-of* | Dias desde o evento anterior do usuário; `-1.0` (`NO_HISTORY_RECENCY`) se é o primeiro evento | Sinal de engajamento recente ([eda.md](eda.md#decisões-para-o-pré-processamento)) |
| `view_count` | item, *as-of* | Nº de views **anteriores** do item | Árvore exploratória apontou `view` como principal driver de popularidade ([eda.md](eda.md#feature-importance-árvore-exploratória)) |

**"*as-of*" é o ponto central**: cada agregação usa `groupby().cumcount()`/
`cumsum().shift()` para enxergar só eventos estritamente anteriores à
linha — por construção não há vazamento temporal, independente de onde o
split cronológico corta os dados (`chronological_split`). Isso também
resolve o problema original de features "constantes por usuário/item":
como o acumulado varia a cada evento novo, duas linhas do mesmo usuário
já não têm mais o mesmo vetor de features.

`frequency`, `engagement_score` e `view_count` têm cauda pesada
(power-law); o modelo aplica `log1p` nelas antes do `StandardScaler`
(fitado só no treino e serializado no `model.pkl` — D5), senão a
informação de popularidade fica espremida numa faixa estreita de z-score.

## 2. Rótulo e negative sampling (`src/data/labeling.py`)

Fonte única da definição de rótulo — nenhum outro módulo redeclara
threshold (a inconsistência de threshold apontada na revisão do TC1 não se
repete aqui, ver `feedback_tc1_professor_review` na memória do projeto):

- **Positivo** = evento `addtocart` ou `transaction` (sinal forte de
  preferência; `POSITIVE_EVENTS`).
- **Negativo** = pares (user, item) não vistos pelo usuário, amostrados do
  catálogo do treino a 8:1, com rejeição de itens já vistos
  (`_sample_negative_rows`); `view_count` do negativo é recalculado
  *as-of* o timestamp do positivo, para não vazar o futuro.
- Amostragem **uniforme** (`popularity_alpha: 0.0`). Foi testada
  amostragem por popularidade^0.75 (estilo word2vec) e o modelo **perdeu**
  para o próprio baseline de popularidade — ele aprendia a punir itens
  populares, enquanto a avaliação usa candidatos uniformes. Detalhe da
  descoberta: `research.md` (decisão D1) e `model_card.md`.

Isso resolve o gap original de "vetores de features quase idênticos com
rótulos conflitantes": o modelo agora aprende "interagiu fortemente vs.
não interagiu", não mais um threshold de peso sobre o evento da própria
linha.

## 3. Avaliação alinhada ao caso de uso (`src/evaluation/`)

Recomendação é ranking Top-K, não classificação binária — a avaliação
reflete isso: `evaluate_ranking_per_user()` roda por usuário (todos os
positivos do split + 100 negativos não vistos, mesmos candidatos para
modelo e baseline) e reporta HitRate/Precision/Recall/NDCG@{10,20}, sob
relevância **forte** (addtocart/transaction) e **ampla** (+view),
segmentado **warm** (usuário com histórico de treino) vs. **cold**. AUC/AP/
F1 continuam reportados sobre o teste rotulado, mas como métrica
complementar — a métrica de promoção no Registry é `val_ndcg_at_20`, não
AUC. Números atuais e limitações: [`model_card.md`](model_card.md).

---

## 4. O que mudou desde a versão anterior deste documento

| Gap identificado (versão anterior) | Resolução | Onde |
|---|---|---|
| `user_idx`/`item_idx` tratados como float cru em vez de índice de embedding | NCF com `nn.Embedding` de usuário/item/categoria | `src/models/mlp.py` |
| Vazamento temporal nas agregações (fitadas no dataset inteiro antes do split) | Agregações causais/*as-of*, seguras por construção | `src/data/feature_engineering.py` |
| Sem negative sampling (label = `weight >= 3` sobre a própria linha) | Rótulo único + negative sampling 8:1 | `src/data/labeling.py` |
| `categoryid` planejado no EDA, nunca implementado | Estágio DVC `content` (ETL chunked dos `item_properties_*`) + embedding de categoria | `src/data/content_etl.py`, `src/models/mlp.py` |
| Métrica de treino/avaliação (AUC/F1) descasada do caso de uso (ranking Top-K) | Avaliação por ranking sampled, warm/cold, forte/ampla | `src/evaluation/ranking.py`, `evaluate.py` |
| `RetailRocketDataset` existia mas não era usado (0% coberto) | Usado pelo `NCFRecommender.fit()` via `DataLoader` | `src/data/dataset.py` |
| Sem cold-start explícito para usuário/item novo | Último índice de cada embedding = "unknown", treinado via id-dropout | `src/models/mlp.py` (`unknown_dropout`) |
| `popularity_tier` calculado mas não usado | Descartado deliberadamente — `view_count` causal (contínuo) carrega a mesma informação | — |

---

## 5. Features candidatas (ainda não implementadas)

Organizadas por esforço/impacto esperado — trabalho futuro, não bugs.

### 5.1 Sinal por par (user, item)

| Feature candidata | O que captura | Esforço |
|---|---|---|
| **`user_item_view_count`** — nº de views deste usuário *neste item específico* | Sinal de interesse repetido, mais específico que `view_count` (agregado por item) | Baixo — groupby (`user_idx`, `item_idx`), causal como as demais |
| **Tempo desde a última vez que o usuário viu *este item*** | Recência específica do par, não recência geral do usuário | Baixo |

### 5.2 Conteúdo do item (além de `cat_idx`)

| Feature candidata | O que captura | Esforço |
|---|---|---|
| Profundidade/categoria-pai na hierarquia (`category_tree.csv`, 1670 categorias) | Generalização entre itens de categorias relacionadas (útil para cold-start de item novo em categoria conhecida) | Médio |
| `available` (propriedade item ativo/inativo por período) | Evita recomendar itens indisponíveis no momento do evento | Médio — série temporal por item, exige join por timestamp |

### 5.3 Comportamento de curto prazo

| Feature candidata | O que captura | Esforço |
|---|---|---|
| **Taxa de conversão histórica do usuário** (`addtocart_rate`, `transaction_rate` — normalizada por `frequency`, não só soma de peso) | Propensão relativa a converter, independente de volume de interações | Baixo |
| **Sessionização** (gap de tempo > N minutos = nova sessão) + posição do evento na sessão | Comportamento de navegação de curto prazo — não modelado hoje (limitação documentada no `model_card.md`) | Médio |

### 5.4 Multi-task para a relevância ampla

O modelo hoje é treinado só para prever interação **forte**
(addtocart/transaction) e por isso fica ~12% abaixo do baseline de
popularidade quando a relevância ampla (que inclui `view`) é usada para
medir — ver limitação em `model_card.md`. Uma cabeça auxiliar treinada
para prever `view` (multi-task) é a candidata mais direta para fechar
esse gap, mas não foi implementada (Fase 3 do tuning priorizou
regularização/negative sampling, que tinha maior impacto imediato).
