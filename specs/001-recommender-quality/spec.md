# Feature Specification: Recomendador que aprende preferências e é medido como recomendação

**Feature Branch**: `001-recommender-quality`
**Created**: 2026-07-13
**Status**: Ready for planning
**Input**: Diagnóstico consolidado em [`docs/feature_selection.md`](../../docs/feature_selection.md) — o modelo atual entrega AUC≈0.5 (não aprende) e é treinado/avaliado com uma formulação de tarefa incompatível com o uso real (recomendação Top-K).

> Este documento descreve **o quê** e **por quê**. Decisões de arquitetura, bibliotecas
> e passos de implementação (NCF, `nn.Embedding`, scaler, hyperparams) pertencem ao
> `plan.md` que acompanha esta spec — não a este arquivo.

---

## ⚡ Contexto e problema

Hoje o pipeline treina um MLP para prever `weight >= 3` sobre o evento da própria linha,
usando features constantes por usuário/item e `user_idx`/`item_idx` como número cru.
Consequências, todas documentadas em `docs/feature_selection.md`:

- Vetores de features quase idênticos recebem rótulos conflitantes → **AUC≈0.5** (Seção 1).
- Agregações calculadas antes do split → **vazamento temporal** (Seção 4).
- Sem negative sampling → **descasamento treino/serving** (Seção 5).
- Avaliação em AUC/`weight>=3`, mas o produto é **ranking Top-K** (Seção 5).

O objetivo desta feature é reformular a tarefa e a avaliação para que o modelo
efetivamente aprenda preferências e seja medido pelo que entrega ao usuário.

---

## User Scenarios & Testing *(mandatory)*

### Primary User Story

Como consumidor da API de recomendação, ao solicitar recomendações para um usuário
conhecido, quero receber uma lista Top-K de itens **relevantes e personalizados** —
melhores do que simplesmente "os itens mais populares" —, de modo que a lista reflita o
histórico e as preferências daquele usuário.

### Acceptance Scenarios

1. **Dado** um usuário com histórico de interações no treino, **quando** o modelo é
   avaliado no conjunto de teste (interações futuras), **então** as métricas de ranking
   Top-K (NDCG@K, Recall@K, Precision@K, HitRate@K) do modelo **superam** as do baseline
   de popularidade nas mesmas condições.

2. **Dado** o pipeline treinado, **quando** as métricas de classificação são calculadas,
   **então** o ROC-AUC no teste fica **materialmente acima de 0.5** (meta direcional
   0.75–0.85, ver requisitos), demonstrando que o modelo aprende sinal real.

3. **Dado** o split cronológico treino/val/teste, **quando** as features são geradas,
   **então** nenhuma feature de uma linha de treino é calculada usando eventos de
   val/teste (ausência de vazamento temporal verificável).

4. **Dado** que o modelo é servido, **quando** ele pontua itens candidatos que o usuário
   nunca viu, **então** a distribuição de entrada do serving é a mesma vista no treino
   (o modelo foi treinado também com exemplos de não-interação).

5. **Dado** um usuário sem histórico (cold-start), **quando** recomendações são
   solicitadas, **então** o sistema recorre ao fallback de popularidade sem erro.

### Edge Cases

- Usuário conhecido cujos itens candidatos já foram todos vistos → **resolvido**: recorre
  ao fallback de popularidade (mesmo comportamento do cold-start), nunca lista vazia.
- Item sem `view` (view_count=0) e itens de cauda longa continuam pontuáveis.
- Definição de "relevante" na avaliação quando o usuário só teve `view` no teste →
  coberto por FR-005a (relevância ampla inclui `view`).

---

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: A definição de rótulo de treino DEVE deixar de ser `weight >= 3` sobre o
  evento da própria linha e passar a distinguir **interação positiva** de
  **não-interação**, incluindo exemplos negativos (pares usuário-item não interagidos).
- **FR-002**: A definição de rótulo/positividade DEVE estar centralizada em um único
  ponto do código, sem duplicação entre estágios de treino e avaliação.
- **FR-003**: As features agregadas por usuário/item DEVEM ser calculadas **sem vazamento
  temporal** — apenas com informação disponível até o instante do evento e ajustadas
  somente sobre o conjunto de treino.
- **FR-004**: O modelo DEVE ser capaz de representar preferências **personalizadas** por
  usuário e por item (além de atributos agregados/populares).
- **FR-005**: A avaliação DEVE reportar métricas de **ranking Top-K** — NDCG@K, Recall@K,
  Precision@K, HitRate@K — para **K = 10 e K = 20**, além das métricas de classificação
  já existentes. As métricas DEVEM ser calculadas **por usuário** (todos os positivos de
  teste do usuário ranqueados numa mesma lista de candidatos) e depois agregadas — com um
  único positivo por lista, Recall/Precision degeneram em função do HitRate.
- **FR-005a**: As métricas de ranking DEVEM ser reportadas sob **ambas** as definições de
  relevância: **forte** (addtocart/transaction) e **ampla** (view/addtocart/transaction),
  exibindo as duas no `model_card.md`.
- **FR-006**: A avaliação DEVE comparar o modelo contra o **baseline de popularidade**
  nas mesmas métricas e no mesmo split.
- **FR-007**: A entrada de features usada no **serving** DEVE ser consistente com a usada
  no **treino** (mesma ordem, mesma escala, mesma transformação).
- **FR-008**: O pipeline DEVE permanecer reprodutível de ponta a ponta via DVC
  (`dvc repro`) e rastreável no MLflow, como hoje.
- **FR-009**: Ao final, o `docs/model_card.md` DEVE ser preenchido com dataset, métricas
  (treino/val/teste), comparação com baselines e limitações.
- **FR-010**: O critério de early stopping / seleção de modelo DEVE usar uma métrica de
  **validação** (não a loss de treino).
- **FR-011**: O sistema DEVE tratar **cold-start** explicitamente: para usuários e/ou
  itens sem histórico no treino, deve haver estratégia definida (ex.: aproveitar features
  de conteúdo/categoria e/ou fallback), e a avaliação DEVE segmentar o desempenho entre
  usuários com histórico e cold-start.

### Critérios de aceite quantitativos

- **AC-1**: NDCG@K e Recall@K do modelo **>** baseline de popularidade (FR-006), para
  K = 10 e 20, sob relevância forte e ampla.
- **AC-2**: ROC-AUC de teste **≥ 0.70** — **portão rígido de aceite** (meta direcional
  0.75–0.85).
- **AC-3**: Ausência de vazamento verificada por teste automatizado (FR-003).

### Decisões resolvidas

- **Relevância nas métricas de ranking:** reportar **ambas** — forte
  (addtocart/transaction) e ampla (view/addtocart/transaction). → FR-005a.
- **K das métricas Top-K:** 10 e 20. → FR-005.
- **Cold-start:** **dentro do escopo** desta feature. → FR-011.
- **Meta de AUC:** portão rígido ≥ 0.70. → AC-2.
- **Razão de negativos:** default 4:1 (do `model_card.md`); revisível no tuning (Fase 3
  do plano). Detalhe de implementação — fica no `plan.md`.
- **Candidatos esgotados no serving:** usuário conhecido cujos candidatos já foram todos
  vistos recebe o **fallback de popularidade** — a API nunca retorna lista vazia.
- **Protocolo das métricas de ranking:** avaliação por usuário com positivos agrupados
  numa única lista (ver FR-005); protocolo detalhado no `research.md` (D3).

### Key Entities

- **Interação**: par usuário-item com tipo de evento (view/addtocart/transaction),
  timestamp e rótulo (positivo/negativo) derivado.
- **Usuário**: `user_idx` contíguo, com features comportamentais causais (frequência,
  engajamento, recência) e representação de preferência.
- **Item**: `item_idx` contíguo, com features de popularidade e, opcionalmente, conteúdo
  (`categoryid`, `popularity_tier`).
- **Recomendação Top-K**: lista ordenada de itens pontuados para um usuário, com a
  estratégia usada (modelo vs. popularidade).

---

## Review & Acceptance Checklist

### Content Quality
- [x] Sem detalhes de implementação (arquitetura/bibliotecas ficam no plan.md)
- [x] Focado em valor para o usuário e no resultado de negócio
- [x] Escrito para stakeholders, não só para devs

### Requirement Completeness
- [x] Nenhum `[A ESCLARECER]` pendente
- [x] Requisitos são testáveis e não ambíguos
- [x] Critérios de sucesso são mensuráveis
- [x] Escopo está delimitado (reformulação de tarefa + avaliação + cold-start)

---

## Fora de escopo (nesta feature)

- Reescrita da camada de serving além de garantir consistência de features (FR-007).
- Novos algoritmos do exploration_doc além do necessário para bater o baseline
  (Two-Tower, ALS, Apriori ficam como trabalho futuro).
- Coleta de novos dados — usa-se o RetailRocket já disponível em `data/raw/`.

## Execution Status

- [x] Problema e contexto descritos
- [x] Cenários de usuário definidos
- [x] Requisitos gerados
- [x] Entidades identificadas
- [x] Ambiguidades resolvidas
- [x] Pronto para `plan.md`
