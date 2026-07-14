# Seleção de Features — Racional, Gaps e Candidatas

## Objetivo

Este documento consolida **por que** cada feature usada hoje no treino do modelo foi
escolhida, **onde** essa decisão está documentada, o **descompasso** encontrado entre
o design planejado e a implementação em produção, e quais features **candidatas**
poderiam ser adicionadas para dar sinal real ao modelo.

Serve de referência para a próxima rodada de trabalho no modelo (ver
[model_card.md](model_card.md) para o estado de performance atual).

---

## 1. Features em produção hoje

Implementadas em `src/data/feature_engineering.py` e consumidas via `FEATURE_COLS` em
`src/training/trainer.py`.

| Feature | Nível | Fórmula | Racional (fonte) |
|---------|-------|---------|-------------------|
| `hour` | evento | Hora do timestamp (0–23) | EDA identificou pico de tráfego 10h–16h ([eda.md](eda.md#principais-achados)) |
| `day_of_week` | evento | Dia da semana (0–6) | EDA identificou padrão semanal de volume ([eda.md](eda.md#principais-achados)) |
| `frequency` | usuário | `COUNT(*)` por `user_idx` | Distribuição power-law de interações — usuários ativos concentram ~70% dos eventos ([eda.md](eda.md#comportamento)) |
| `recency_days` | usuário | Dias desde o último evento do usuário | Sinal de engajamento recente ([eda.md](eda.md#decisões-para-o-pré-processamento)) |
| `engagement_score` | usuário | `SUM(weight)` por `user_idx` | Pondera intensidade de interação (view=1, addtocart=3, transaction=5), refletindo hierarquia de intenção validada no EDA |
| `view_count` | item | `COUNT(event=='view')` por `item_idx` | Árvore exploratória apontou `view` como principal driver de popularidade ([eda.md](eda.md#feature-importance-árvore-exploratória)) |
| `user_idx` | evento | Índice contíguo (0-based) do usuário | **Planejado para ser índice de `nn.Embedding`, não valor numérico direto** — ver Seção 2 |
| `item_idx` | evento | Índice contíguo (0-based) do item | Idem |

### Observação sobre `frequency` / `engagement_score` / `recency_days` / `view_count`

Essas quatro são agregações **constantes por usuário ou por item** — não variam entre
as linhas de um mesmo usuário (ou item), independentemente do tipo de evento daquela
linha específica. Isso significa que elas descrevem "quem é o usuário" / "quão popular
é o item" em geral, mas não "por que esta interação específica teve este resultado".
É uma limitação relevante para o próximo ponto.

> **Consequência mecânica (causa direta do AUC≈0.5):** o label é `weight >= 3`
> calculado sobre o evento *daquela mesma linha* (`trainer.py:52`), mas **nenhuma
> feature codifica o tipo de evento da linha**. Como as features de usuário/item são
> constantes, duas linhas do mesmo usuário têm vetor de features praticamente idêntico
> (só variam `user_idx`/`item_idx`, que entram como float sem significado), porém labels
> diferentes (uma `view`=0, outra `addtocart`=1). Vetores iguais com rótulos
> conflitantes → o modelo não tem como separar as classes. Isso é a explicação
> mecânica do AUC≈0.5, mais precisa do que "as features não discriminam bem".

---

## 2. Gap: `user_idx` / `item_idx` foram desenhados para embeddings, mas viram input numérico cru

Três documentos deixam este plano explícito:

- `eda.md` → *"Encoding visitorid → user_idx (int sequencial): necessário para
  embeddings no modelo"* (mesma frase repetida para `item_idx`).
- `preprocessing.md` → tabela "Decisões de Design": *"`user_idx`/`item_idx`
  contíguos base-0 — Compatíveis diretamente com `nn.Embedding` do PyTorch"*.
- `exploration_doc.md`, seção 8 (Neural Collaborative Filtering) — descreve
  exatamente a arquitetura NCF que deveria ter sido levada a produção:

  ```text
  user_id -> user embedding
  item_id -> item embedding
  concatenação -> MLP -> probabilidade de interação
  ```

  incluindo negative sampling explícito (`addtocart`/`transaction` = positivo,
  `view`-only = negativo, com amostragem controlada de negativos).

**O que existe em produção** (`src/models/mlp.py`) é um MLP simples que recebe as 8
colunas de `FEATURE_COLS` concatenadas como float — `user_idx` e `item_idx` entram
como número cru, sem `nn.Embedding`, e o label é `weight >= 3` sobre os eventos reais
(sem negative sampling).

Evidência de que a implementação com embeddings foi iniciada e abandonada:
`src/data/dataset.py` (`RetailRocketDataset`) existe, mas **não é usado em nenhum
lugar** do `trainer.py` (0% de cobertura nos testes) — parece ser o resquício da
tentativa de portar o notebook de exploração (`notebooks/02_exploration_MLP_Architecture.ipynb`)
para o pipeline de produção.

---

## 3. Feature planejada e nunca implementada: `categoryid`

`eda.md` lista `categoryid` como decisão de pré-processamento: *"categoryid como
feature de conteúdo — único metadado estruturado confiável disponível"*. Nunca foi
adicionada a `feature_engineering.py` nem a `FEATURE_COLS`. Os dados brutos que a
sustentariam já estão disponíveis e não são usados por nenhum estágio do pipeline:

| Arquivo | Conteúdo | Uso atual |
|---------|----------|-----------|
| `data/raw/item_properties_part1/2.csv` | `(timestamp, itemid, property, value)` — inclui `categoryid`, `available` e ~1000 propriedades anonimizadas | Só explorado no EDA, não entra no pipeline DVC |
| `data/raw/category_tree.csv` | Hierarquia `categoryid → parentid` (1670 categorias) | Idem |

---

## 4. Gap: vazamento temporal no cálculo das agregações

Este é um **bug de correção**, não só de sinal. Em `src/features/build_features.py`
(`run()` → `build_features(df)` → `split_data(df)`), as agregações de usuário e item são
calculadas sobre o **dataset inteiro** e só *depois* o `chronological_split` é aplicado:

```text
build_interaction_features(df)   # frequency/engagement/recency/view_count sobre TODOS os eventos
        ↓
chronological_split(df)          # split só acontece aqui
```

Ou seja, as features das linhas de **treino** incluem eventos de **val/test (futuro)**.
Pior ainda: `compute_user_features` usa `reference_date = max(timestamp) + 1 dia`
**global** (`feature_engineering.py:60`), então a `recency_days` do treino é medida
contra o fim do período de teste. Isso é data leakage clássico — justamente o que o
split cronológico deveria evitar. Além de inflar artificialmente qualquer métrica, torna
as features menos úteis do que seriam se calculadas de forma causal.

**Correção:** calcular as agregações apenas sobre o treino e, de preferência, de forma
*as-of* (acumulado até o timestamp do evento). Isso mata o leakage **e** resolve a
"constância por usuário" da Seção 1 — um acumulado até `t` varia linha a linha, dando
sinal por interação.

---

## 5. Gap: descasamento treino/serving e métrica de avaliação

- **Treino vs. serving:** no serving (`src/serving/recommender.py`, `_model_recommend`)
  o modelo pontua **itens que o usuário nunca viu** (candidatos). No treino ele só vê
  eventos que aconteceram — nunca aprendeu o conceito de "não-interação". É exatamente o
  negative sampling ausente (Seção 6.1), mas o ponto adicional é que, sem ele, o serving
  opera **fora da distribuição de treino**.
- **Métrica:** treino e avaliação usam ROC-AUC/F1 sobre `weight>=3` (`evaluate.py`),
  enquanto todo o `exploration_doc.md` (seção 3) define o problema como **ranking Top-K**
  (HitRate@K, Precision@K, Recall@K, NDCG@K). Mesmo consertando o classificador binário,
  essas métricas não medem qualidade de recomendação. É preciso avaliar o ranking.

> **Nota de conformidade:** o threshold `weight>=3` está duplicado em `trainer.py:52` e
> `evaluate.py:71`. Ao trocar a definição de label (Seção 6.1), centralizar em um único
> lugar para não repetir o erro de "threshold inconsistente" apontado na revisão do TC1.

---

## 6. Features candidatas para adicionar

Organizadas por esforço/impacto esperado.

### 6.1 Alto impacto — resolvem a falta de sinal por linha

| Feature candidata | O que captura | Esforço |
|---|---|---|
| **Embeddings de `user_idx`/`item_idx`** (via `nn.Embedding`, arquitetura NCF já descrita em `exploration_doc.md`) | Preferências latentes de usuário/item — substitui o uso incorreto dos IDs como float | Médio — arquitetura já documentada, só falta portar pro `mlp.py`/`trainer.py` |
| **Contagem de views do usuário *neste item específico*** (`user_item_view_count`) | Sinal de interesse repetido — quantas vezes esse usuário já visitou esse item antes deste evento | Baixo — groupby (`user_idx`, `item_idx`) |
| **Tempo desde a última vez que o usuário viu *este item*** | Recência específica do par, não recência geral do usuário | Baixo |
| **Negative sampling explícito** (como descrito na seção 8.2 do `exploration_doc.md`) em vez do threshold `weight >= 3` | Label mais alinhado a "interagiu vs. não interagiu", problema mais tratável para um recomendador | Médio |

### 6.2 Médio impacto — conteúdo do item

| Feature candidata | O que captura | Esforço |
|---|---|---|
| `categoryid` (one-hot ou embedding) | Similaridade de conteúdo entre itens | Baixo–médio — dado já existe em `item_properties_*.csv`, falta ETL |
| Profundidade/categoria-pai na hierarquia (`category_tree.csv`) | Generalização entre itens de categorias relacionadas | Médio |
| `available` (propriedade item ativo/inativo por período) | Evita recomendar itens indisponíveis no momento do evento | Médio — é uma série temporal por item, exige join por timestamp |
| `popularity_tier` (já calculado em `compute_item_features`, mas não usado no treino) | Categórica de popularidade (long_tail/mid_tier/top_tier) | Muito baixo — só adicionar à lista de features e fazer encoding |

### 6.3 Baixo esforço, impacto incremental

| Feature candidata | O que captura | Esforço |
|---|---|---|
| **Taxa de conversão histórica do usuário** (`addtocart_rate`, `transaction_rate` — não só soma de peso) | Propensão relativa do usuário a converter, normalizada por volume de interações | Baixo |
| **Sessionização** (gap de tempo > N minutos = nova sessão) + posição do evento na sessão | Comportamento de navegação de curto prazo | Médio |
| **Normalização/scaling** das features contínuas atuais (`StandardScaler` em `frequency`, `recency_days`, `view_count`, `engagement_score`) | Nenhum sinal novo, mas remove problema de escala (esses valores hoje entram crus, em ordens de grandeza muito diferentes, no MLP) | Muito baixo |

---

## 7. Resumo

- **Causa raiz do AUC≈0.5 não é falta de features, é a formulação da tarefa.** Vetores
  de features quase idênticos recebem rótulos conflitantes (Seção 1), porque o label
  depende do tipo de evento da linha e nenhuma feature codifica isso.
- As 6 features comportamentais atuais têm racional sólido vindo do EDA, mas são
  **constantes por usuário/item** e ainda são calculadas **com vazamento temporal**
  (Seção 4) — precisam virar agregações causais/*as-of* fitadas só no treino.
- `user_idx`/`item_idx` foram desenhados para embeddings (documentado em 3 lugares) mas
  são tratados como float cru (Seção 2).
- Falta **negative sampling** e há **descasamento treino/serving** e de **métrica**
  (Seção 5): treina em `weight>=3`/AUC, mas serve ranking Top-K.
- `categoryid` foi planejada no EDA e nunca implementada, apesar do dado bruto já estar
  em `data/raw/` (Seção 3).
- A lista de features candidatas (Seção 6) está ordenada por esforço vs. impacto.

---

## 8. Plano de execução ordenado

> **Status (2026-07-13): IMPLEMENTADO** pela feature
> [`specs/001-recommender-quality/`](../specs/001-recommender-quality/spec.md).
> Fase 0: `src/data/labeling.py` (rótulo único + negative sampling 4:1 por
> popularidade^0.75), agregações causais em `src/data/feature_engineering.py`,
> métricas de ranking em `src/evaluation/ranking.py`. Fase 1: NCF com
> embeddings (+unknown no último índice) em `src/models/mlp.py`, early
> stopping por val-AUC, promoção por `val_ndcg_at_20`. Fase 2: scaler
> serializado no `model.pkl`, `categoryid` via estágio `content` do DVC,
> segmentação warm/cold na avaliação. `popularity_tier` foi descartado como
> feature (o `view_count` causal contém a mesma informação, contínua).
> Fase 3 (tuning) permanece como trabalho futuro.

Ordenado por impacto/esforço. As Fases 0–1 são pré-requisito para o modelo aprender
qualquer coisa; features novas (Fase 2) só valem depois disso.

### Fase 0 — Corrigir a tarefa, não o modelo *(maior alavanca)*

1. **Negative sampling explícito** substituindo o label `weight>=3`: positivo =
   `addtocart`/`transaction`; negativo = pares (user, item) não interagidos, amostrados
   ~4:1 (já prometido em `model_card.md`), removendo dos negativos os pares que também
   são positivos. Centralizar a definição de label num único módulo.
2. **Agregações causais / as-of** para `frequency`, `engagement_score`, `view_count`:
   acumulado até o `timestamp` do evento, **fitadas apenas no treino** (val/test usam
   estatística do treino). Resolve o leakage da Seção 4 **e** a constância da Seção 1.
3. **Métricas de ranking** no `evaluate.py`: HitRate@K, Precision@K, Recall@K, NDCG@K
   (K=10/20) além de AUC/AP, para medir recomendação de fato.

### Fase 1 — Arquitetura NCF com embeddings *(alvo já documentado, Seção 2)*

4. Reescrever `MLPRecommender` (`src/models/mlp.py`): `nn.Embedding(n_users, d)` +
   `nn.Embedding(n_items, d)` → concatena com features contínuas → MLP → logit.
   Passar `n_users`/`n_items`/`embedding_dim` via `params.yaml`/factory.
5. Ligar o `RetailRocketDataset` (hoje 0% usado) no `trainer.py`, separando
   `user_idx`/`item_idx` (long, p/ embedding) das contínuas (float).
6. Early stopping por **métrica de validação (NDCG/AUC)** e não pela loss de treino
   (hoje `mlp.py:102-110` olha só a loss de treino — não previne overfitting real).

### Fase 2 — Escala e conteúdo *(esforço baixo–médio)*

7. **StandardScaler** nas contínuas, fitado só no treino e serializado com o modelo
   (o serving precisa usar o mesmo scaler).
8. **`categoryid`** (embedding por categoria) + `popularity_tier` (já calculado em
   `compute_item_features`, só falta usar). Dado bruto em `data/raw/item_properties_*`.

### Fase 3 — Tuning *(depois que o pipeline aprender)*

9. Buscar `embedding_dim`, `hidden_dims`, `lr`, `dropout`, `weight_decay` e razão de
   negativos.

### Metas de métrica (referência RetailRocket)

- O baseline de **popularidade** precisa ser superado para justificar o modelo.
- Pós-Fase 1: **AUC 0.75–0.85** no binário com negative sampling e **NDCG@20 acima do
  baseline de popularidade**. NCF + categoria tende a empurrar mais.
