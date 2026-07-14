# Feature Specification: Fechar o gap de relevância ampla e fortalecer sinal de cold-start

**Feature Branch**: `002-multitask-coldstart`
**Created**: 2026-07-14
**Status**: Ready for planning
**Input**: Limitações documentadas em [`docs/model_card.md`](../../docs/model_card.md)
(seções "Desempenho — Ranking Top-K" e "Limitações") após a implementação da
feature `001-recommender-quality`: (1) sob relevância ampla o modelo fica
~12% abaixo do baseline de popularidade para usuários warm; (2) para
usuários cold o modelo puro é próximo do aleatório em ranking.

> Este documento descreve **o quê** e **por quê**. Arquitetura, fórmulas de
> loss e passos de implementação pertencem ao `plan.md`/`research.md` que
> acompanham esta spec — não a este arquivo.

---

## ⚡ Contexto e problema

O modelo em produção (`NCFRecommender`, spec `001-recommender-quality`) é
treinado para prever exclusivamente **interação forte**
(addtocart/transaction). Isso produz dois efeitos colaterais medidos e
documentados no `model_card.md`:

1. **Relevância ampla (inclui `view`) fica abaixo da popularidade.** O
   modelo nunca recebe sinal de treino sobre `view` — ele otimiza uma
   tarefa diferente da que a métrica de relevância ampla mede. Resultado:
   NDCG@10 warm = 0.207 (modelo) vs. 0.236 (popularidade), ~12% pior.
2. **Cold-start é quase aleatório para o modelo puro.** NDCG@10 cold sob
   relevância forte cai para 0.065 (vs. 0.307 da popularidade). O fallback
   de popularidade no serving cobre esse caso operacionalmente, mas o
   modelo em si desperdiça sinal de conteúdo disponível: uma inspeção do
   código (`src/models/mlp.py::_split_matrix`) mostra que a categoria de um
   item roteado para o índice "unknown" (item ausente do treino) é
   consultada **pelo índice já roteado**, não pelo `item_idx` real — ou
   seja, mesmo quando o item tem `categoryid` conhecido (via estágio
   `content`), esse sinal é descartado se o item não apareceu no treino.
   Além disso, usuários com pouquíssimas interações (mediana ~2 no
   dataset) têm apenas features **agregadas** (por usuário/por item), sem
   nenhum sinal específico do par (usuário, item) que ajudaria a
   diferenciar "este usuário já viu este item" de "nunca viu".

O objetivo desta feature é: (a) dar ao modelo sinal de treino explícito
para `view`, permitindo pontuar relevância ampla sem depender só da
popularidade; e (b) parar de descartar sinal de conteúdo/par já disponível
para itens e pares pouco vistos, reduzindo — não necessariamente
eliminando — o gap de cold-start.

---

## User Scenarios & Testing *(mandatory)*

### Primary User Story

Como consumidor da API de recomendação, ao solicitar recomendações para um
usuário conhecido cujo comportamento predominante é navegação (`view`) sem
necessariamente comprar, quero que a lista Top-K reflita esse padrão de
navegação **pelo menos tão bem quanto** simplesmente mostrar os itens mais
populares — hoje o modelo faz pior que isso sob esse critério.

### Acceptance Scenarios

1. **Dado** um usuário warm (com histórico de treino), **quando** o
   ranking é avaliado sob **relevância ampla** (view/addtocart/transaction),
   **então** as métricas NDCG@{10,20} do modelo deixam de ficar
   consistentemente abaixo do baseline de popularidade (gap atual: ~12%).
2. **Dado** o mesmo cenário sob **relevância forte** (addtocart/
   transaction) já resolvido pela spec `001-recommender-quality`,
   **quando** o modelo é reavaliado após esta feature, **então** o
   resultado da spec anterior (modelo supera popularidade em NDCG@{10,20})
   **não regride**.
3. **Dado** um item ausente do treino mas com `categoryid` conhecido (via
   estágio `content`), **quando** o modelo pontua esse item, **então** o
   score usa o embedding da categoria real do item, não um valor de
   categoria "unknown" genérico.
4. **Dado** um usuário com poucas interações totais mas que já viu um item
   específico anteriormente, **quando** esse item é pontuado para
   recomendação, **então** o modelo tem acesso a um sinal específico desse
   par (usuário, item) além das features agregadas por usuário/item.
5. **Dado** um usuário verdadeiramente cold (zero histórico), **quando**
   recomendações são solicitadas, **então** o sistema continua usando o
   fallback de popularidade (fora do escopo desta feature mudar isso —
   pares (usuário, item) inexistentes não têm o que capturar).

### Edge Cases

- Item com `categoryid` desconhecido (estágio `content` vazio ou item sem
  propriedade `categoryid`) → mantém o comportamento atual (`UNKNOWN_CAT_IDX`).
- Par (usuário, item) nunca visto → feature de par usa o mesmo sentinela de
  "sem histórico" já usado por `recency_days` (`NO_HISTORY_RECENCY`).
- Volume de eventos `view` (~882K) é ~13x o de positivos fortes (~66K) —
  a inclusão de `view` como sinal de treino não pode inflar o custo de
  treino nessa proporção sem controle (ver requisitos de amostragem).

---

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: O treino DEVE incluir um sinal supervisionado para `view`
  (interação ampla), distinto do rótulo de interação forte já existente,
  sem substituir esse rótulo.
- **FR-002**: A arquitetura do modelo DEVE prever ambos os sinais
  (interação forte e `view`) a partir de uma representação compartilhada
  (mesmos embeddings de usuário/item/categoria), evitando duplicar o
  modelo inteiro.
- **FR-003**: A inclusão de eventos `view` no treino DEVE ser amostrada de
  forma controlável (não 1:1 com o volume bruto de views), para não
  inflar desproporcionalmente o custo de treino nem afogar o sinal de
  interação forte.
- **FR-004**: A avaliação de ranking sob **relevância ampla** DEVE usar o
  sinal de `view` do modelo para pontuar; a avaliação sob **relevância
  forte** DEVE continuar usando o sinal de interação forte — ambas já
  reportadas separadamente desde a spec `001-recommender-quality`
  (FR-005a daquela spec).
- **FR-005**: O critério de early stopping / seleção de modelo (FR-010 da
  spec `001-recommender-quality`) DEVE continuar baseado na métrica de
  validação da interação **forte** — o sinal de `view` é auxiliar, não
  pode desviar a seleção do cenário principal.
- **FR-006**: A consulta de categoria de um item DEVE usar o `item_idx`
  real do item, não o índice pós-roteamento para "unknown" — um item
  ausente do treino mas com categoria conhecida (via estágio `content`)
  DEVE preservar esse sinal.
- **FR-007**: O contrato de features DEVE ganhar um sinal causal
  (*as-of*) específico do par (usuário, item): quantas vezes este usuário
  já viu este item antes do instante do evento.
- **FR-008**: O contrato de features DEVE ganhar um sinal causal de
  recência específica do par (usuário, item): há quanto tempo (dias) desde
  a última vez que este usuário viu este item, com sentinela para "nunca
  visto" — mesma semântica do `recency_days` já existente para o usuário.
- **FR-009**: As novas features de par (FR-007/FR-008) DEVEM ser
  calculadas com a mesma disciplina causal das features existentes — sem
  vazamento temporal — inclusive para os negativos amostrados
  (`src/data/labeling.py`), que hoje já recalculam `view_count` *as-of*
  para os pares sintéticos.
- **FR-010**: A avaliação DEVE reportar as métricas de ranking segmentadas
  por warm/cold (já existente) **antes e depois** desta feature, para
  quantificar a redução do gap de cold-start no `model_card.md`.
- **FR-011**: O pipeline DEVE permanecer reprodutível via DVC (`dvc repro`)
  e rastreável no MLflow, como hoje.

### Critérios de aceite quantitativos

- **AC-1**: Sob relevância ampla, NDCG@{10,20} do modelo para usuários
  warm **deixa de ser consistentemente inferior** ao baseline de
  popularidade — meta direcional: paridade ou vantagem (hoje: -12% em
  NDCG@10, -10% em NDCG@20).
- **AC-2**: Sob relevância forte, o resultado da spec `001-recommender-quality`
  (NDCG/Recall@{10,20} do modelo > popularidade, warm) **não regride** —
  portão de não-regressão.
- **AC-3**: ROC-AUC de teste continua **≥ 0.70** (AC-2 original da spec
  `001-recommender-quality`) — portão de não-regressão.
- **AC-4**: O gap de NDCG@10 cold vs. popularidade (relevância forte, hoje
  0.065 vs. 0.307) **reduz** após FR-006/FR-007/FR-008 — meta direcional,
  sem alvo numérico rígido (cold continua coberto operacionalmente pelo
  fallback de popularidade no serving, fora do escopo mudar isso).

### Decisões resolvidas

- **Escopo do cold-start nesta feature**: reduzir o gap para itens/pares
  com **algum** sinal disponível (categoria conhecida, par já visto antes);
  usuário verdadeiramente zero-histórico continua coberto só pelo
  fallback — não há o que aprender sem nenhuma interação. → FR-006/007/008,
  cenário 5.
- **Métrica de promoção no Registry**: permanece `val_ndcg_at_20` sob
  relevância forte (não muda para a métrica de `view`) — o sinal de `view`
  é reportado no `model_card.md`, mas não decide qual run é promovido.
  → FR-005.
- **Amostragem de `view`**: proporção configurável em `params.yaml`
  (detalhe de implementação — fica no `research.md`), não volume bruto.
  → FR-003.

### Key Entities

- **Sinal de `view`**: rótulo binário auxiliar (`view_label`) — 1 para
  qualquer evento observado (view/addtocart/transaction) de um par
  (usuário, item), 0 para pares negativos amostrados. Coexiste com o
  rótulo de interação forte já existente (`label`), sem substituí-lo.
- **Feature de par (usuário, item)**: `user_item_view_count` e
  `user_item_recency_days` — agregações causais no nível do par, não do
  usuário ou do item isoladamente.
- **Categoria do item**: já existente (`cat_idx`, estágio `content`); esta
  feature corrige onde ela é *consultada* para itens roteados a
  "unknown", não sua origem.

---

## Review & Acceptance Checklist

### Content Quality
- [x] Sem detalhes de implementação (arquitetura/loss ficam no plan.md/research.md)
- [x] Focado em valor mensurável (métricas de ranking documentadas no model_card.md)
- [x] Escrito para stakeholders, não só para devs

### Requirement Completeness
- [x] Nenhum `[A ESCLARECER]` pendente
- [x] Requisitos são testáveis e não ambíguos
- [x] Critérios de sucesso são mensuráveis (AC-1 a AC-4)
- [x] Escopo está delimitado (multi-task view + correção de roteamento de
  categoria + features de par; NÃO inclui sessão, GMF, blend de score ou
  busca de hiperparâmetro — ver Fora de escopo)

---

## Fora de escopo (nesta feature)

- Blend/ensemble de score modelo+popularidade (alternativa mais barata
  discutida, mas não escolhida para esta rodada).
- Ramo GMF (produto escalar) somado ao MLP e busca sistemática de
  hiperparâmetro — ficam como trabalho futuro independente.
- Modelagem de sessão/sequência.
- Mudar a estratégia de fallback do serving para usuário zero-histórico
  (continua sendo popularidade).
- Reescrever a API de serving além do necessário para expor o novo sinal
  de `view` na pontuação de relevância ampla (se o serving hoje só usa
  relevância forte para o `strategy: "model"`, isso é decisão de
  `plan.md`, não desta spec).

## Execution Status

- [x] Problema e contexto descritos (com achado concreto no código:
  roteamento de categoria via índice pós-unknown)
- [x] Cenários de usuário definidos
- [x] Requisitos gerados
- [x] Entidades identificadas
- [x] Ambiguidades resolvidas
- [x] Pronto para `plan.md`
