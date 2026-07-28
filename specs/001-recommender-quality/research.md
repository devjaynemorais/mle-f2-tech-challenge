# Research: decisões técnicas (Fase 0)

**Feature**: `001-recommender-quality`
**Plan**: [`plan.md`](./plan.md)

Fecha as 5 decisões abertas na Fase 0 do plano. Formato por decisão:
**Decisão / Justificativa / Alternativas consideradas**.

---

## D1 — Geração de negativos em escala

**Decisão.** Negative sampling **por popularidade amortecida** (`freq(item)^0.75`,
estilo word2vec), na razão **4:1** por positivo. Para cada usuário positivo, amostro itens
do vetor global de frequência e **rejeito** os que estão no conjunto de itens já vistos
por aquele usuário (positivos + histórico), reamostrando os rejeitados. Semente fixa.

**Justificativa.** Uniforme gera negativos "fáceis demais" (quase tudo é cauda longa numa
base com esparsidade >99,99%), e o modelo aprende só popularidade. A amostragem por
popularidade^0.75 produz negativos mais informativos (padrão em NCF/BPR/word2vec). Como os
usuários passaram por `min_interactions >= 5` e a mediana de itens vistos é baixa, o
conjunto "visto" por usuário é pequeno → rejeição custa O(1) amortizado, **sem** matriz
densa usuário×item (235K itens).

**Alternativas.** (a) Uniforme — descartada por gerar sinal fraco. (b) Todos os negativos
/ full-softmax — inviável a 235K itens por usuário.

> **Revisão empírica (Gate F1).** Com α=0.75 o modelo perdeu para o baseline de
> popularidade (NDCG@20 0.105 vs 0.348; AUC teste 0.59): negativos amostrados por
> popularidade ensinam o modelo a **punir** itens populares, enquanto os candidatos
> da avaliação (D3) são uniformes — descasamento de distribuição. Decisão revista
> para **α=0 (uniforme)**, o padrão em NCF (He et al.); o expoente permanece
> parametrizado em `params.labeling.popularity_alpha` para tuning (Fase 3).

---

## D2 — Agregação *as-of* eficiente (sem vazamento)

**Decisão.** Ordenar por `(entidade, timestamp)` e calcular de forma **cumulativa
excluindo o evento corrente**:
- `frequency` = `groupby(user).cumcount()`
- `engagement_score` = `groupby(user).weight.cumsum().shift(1)` (por usuário)
- `recency_days` = diferença para o **timestamp anterior** do mesmo usuário
- `view_count` (item) = cumulativo de views por item, `shift(1)`

O **scaler** e quaisquer estatísticas de referência são ajustados **só no treino**.

**Justificativa.** Agregação causal é, por construção, livre de vazamento — cada linha usa
apenas eventos **anteriores** a ela, independentemente do split (isso corrige o bug da
Seção 4, que era agregação **global não-causal**). Custo O(n log n) pela ordenação +
operações vetorizadas; evita join O(n²). Bônus: deixa de ser constante por usuário/item,
dando sinal por interação (resolve a Seção 1).

**Alternativas.** (a) Janelas temporais com self-join — custo alto. (b) Recalcular por
split — ainda exigiria causalidade; sem ganho.

---

## D3 — Conjunto de candidatos na avaliação Top-K

**Decisão.** **Sampled ranking metrics por usuário**: para cada usuário do teste,
agrupar **todos os seus positivos de teste** numa única lista de candidatos junto com
**100 negativos** não vistos, ranquear e computar HR/NDCG/Precision/Recall @{10,20} por
usuário, depois agregar (média sobre usuários). O **mesmo** conjunto de candidatos (mesma
semente) é usado pelo modelo **e** pelo baseline de popularidade. Reportar sob relevância
**forte** e **ampla**.

**Por que agrupar por usuário (e não 1 positivo por lista).** Com um único positivo por
lista (protocolo leave-one-out clássico), `HitRate@K ≡ Recall@K` e
`Precision@K = HitRate@K / K` — das 4 métricas exigidas pelo FR-005 sobrariam só 2
independentes. Agrupar os positivos do usuário mantém as 4 métricas informativas.

**Custo sob relevância ampla.** Sob relevância ampla os `view` do teste viram positivos e
dominam o volume (~2.8M eventos). Para conter o custo, aplicar **teto de positivos por
usuário** na avaliação ampla (default: **20 mais recentes**, parametrizável em
`params.yaml`); relevância forte avalia todos os positivos (addtocart/transaction são
raros).

**Justificativa.** Ranquear o catálogo inteiro (235K) por usuário é caro e, na esparsidade
desta base, sampled metrics são o padrão (He et al., NCF). Usar candidatos idênticos entre
métodos garante comparação justa (AC-1).

**Ressalva documentada.** Sampled metrics têm viés conhecido (Krichene & Rendle, 2020);
aceitável para comparação **interna** desde que consistente entre modelos. Deixar um
caminho opcional de avaliação full-catalog (mais lento) para validação final.

**Alternativas.** Full-catalog — maior fidelidade, custo alto; fica como opção.

---

## D4 — Estratégia de cold-start (FR-011)

**Decisão.** Definir **cold** = entidade ausente do treino (usuários já filtrados por
`min_interactions>=5`; item cold = `item_idx` não visto no treino). Reservar o **último
índice** (`n_users` / `n_items`) como embedding "unknown" para as torres de usuário e de
item — as tabelas de embedding têm tamanho `n+1` e ids não vistos são roteados para o
índice `n`. Itens ganham também **embedding de `categoryid`**, então um item novo ainda
recebe sinal de conteúdo. No serving, usuário unknown → **fallback de popularidade** (já
existe). Na avaliação, **segmentar** métricas em `warm` vs `cold`.

**Por que o último índice (e não o 0).** `user_idx`/`item_idx` nascem de `pd.factorize`
(`preprocessor.py`) e começam em **0**. Reservar o índice 0 exigiria deslocar todos os ids
em +1 — migrando parquets de `data/processed/`, o `store.py` do serving e mapeamentos
persistidos, com risco de off-by-one silencioso. Usar o índice `n` não re-mapeia nada.

**Justificativa.** Escopo mínimo, sem reescrever o serving; aproveita conteúdo para itens
novos e mantém o fallback atual para usuários novos.

**Alternativas.** Two-tower puramente de conteúdo — maior escopo, adiado para trabalho
futuro (consistente com o exploration_doc).

---

## D5 — Persistência do modelo (scaler + metadados)

**Decisão.** Manter **um único `model.pkl`** do `MLPRecommender`, agora **auto-contido**:
passa a incluir um `scaler` ajustado no treino (**hoje não existe scaler algum** — treino
e serving usam features cruas, e `recommender._build_matrix` monta o vetor à mão) e os
metadados (`n_users`, `n_items`, `embedding_dim`, versão do contrato de features,
vocabulário de categoria). Serving carrega o pkl e **reutiliza** o mesmo scaler e contrato
(garante FR-007). Continua sendo o artefato rastreado por DVC/MLflow.

**Justificativa.** Menor alteração em `save_model`/`load_model`/`model_loader`; elimina a
possibilidade de treino e serving usarem transformações diferentes.

**Alternativas.** `state_dict` + JSON sidecar — mais limpo e portável, porém mais
encanamento; anotado como refactor futuro.

---

## Impacto nos requisitos

| Decisão | Fecha/《habilita》 |
|---------|-------------------|
| D1 | FR-001 (negative sampling) |
| D2 | FR-003 (sem vazamento), mitiga Seção 1 |
| D3 | FR-005/005a/006 (ranking + baseline justo) |
| D4 | FR-011 (cold-start + segmentação) |
| D5 | FR-007 (consistência treino↔serving) |

Todas as ambiguidades da Fase 0 estão resolvidas → liberado para `data-model.md`,
`contracts/` e `tasks.md`.
