# Implementation Plan: Recomendador que aprende preferências e é medido como recomendação

**Feature Branch**: `001-recommender-quality`
**Created**: 2026-07-13
**Spec**: [`spec.md`](./spec.md)
**Status**: Draft

---

## Summary

Reformular a **tarefa de aprendizado** e a **avaliação** do recomendador para que o
modelo deixe de entregar AUC≈0.5 e passe a superar o baseline de popularidade em métricas
de ranking Top-K. Em ordem de impacto: (0) trocar o rótulo `weight>=3` por
positivo/negativo com negative sampling, tornar as agregações causais e sem leakage, e
avaliar com métricas de ranking; (1) migrar o MLP para uma arquitetura NCF com embeddings
de usuário/item; (2) escala + conteúdo (`categoryid`) + cold-start; (3) tuning.

Referência de diagnóstico: [`docs/feature_selection.md`](../../docs/feature_selection.md),
Seções 1–8.

---

## Technical Context

| Item | Valor |
|------|-------|
| **Linguagem** | Python 3.13 |
| **ML** | PyTorch (modelo), scikit-learn (baselines, scaler, métricas de classificação) |
| **Orquestração** | DVC (`dvc.yaml`, 5 estágios) + MLflow (tracking/registry) |
| **Deps/build** | Poetry, ruff, pytest |
| **Dados** | RetailRocket em `data/raw/` (~2.8M eventos, ~1.4M users, ~235K itens, esparsidade >99.99%) |
| **Serving** | FastAPI (`src/serving/`) — recomendação Top-K + fallback popularidade |
| **Project type** | Pipeline ML monolítico (single project) |
| **Scale/perf** | Negative sampling e ranking-eval precisam ser eficientes na escala acima (evitar produtos cartesianos densos) |

---

## Constitution Check *(gates)*

Não há `memory/constitution.md` formal no repo. Na ausência dele, adoto como portões os
princípios do projeto + a revisão do TC1 (registrados em memória do time):

- [ ] **Sem erros de ruff** no código novo/alterado (o TC1 reprovou por isso).
- [ ] **Testes não dependem de dados reais em disco** — usar fixtures sintéticas pequenas.
- [ ] **Sem caminhos absolutos** — tudo relativo à raiz do projeto / `params.yaml`.
- [ ] **Definição de rótulo/threshold em um único módulo** (FR-002) — nada de duplicar
  `weight>=3` entre treino e avaliação, como hoje.
- [ ] **Reprodutibilidade DVC** preservada (`dvc repro` verde) e runs no MLflow (FR-008).

> Recomendação paralela (fora do caminho crítico): rodar `specify init` para materializar
> uma `constitution.md` de verdade. Não bloqueia esta feature.

---

## Project Structure (arquivos afetados)

```
src/
  data/
    feature_engineering.py   [MOD]  agregações causais/as-of; fit só no treino
    labeling.py              [NEW]  definição ÚNICA de positivo/negativo + negative sampling
    build_features.py …      (src/features/build_features.py) [MOD] split ANTES de agregar
    dataset.py               [MOD]  separar ids (long) de contínuas (float); expor target
  models/
    mlp.py                   [MOD]  MLPRecommender → arquitetura NCF (embeddings + MLP)
    factory.py               (sem mudança de contrato; novos kwargs)
  training/
    trainer.py               [MOD]  usar labeling + dataset; early stopping por val-metric
  evaluation/
    evaluate.py              [MOD]  ranking Top-K por usuário; segmentar cold-start
    ranking.py               [NEW]  ndcg@k / recall@k / precision@k / hitrate@k
  serving/
    recommender.py           [MOD]  consistência de features com o treino (scaler/ids)
    store.py                 [MOD]  servir features causais + scaler
    model_loader.py          [MOD]  carregar scaler + metadados (n_users/n_items)
configs / raiz:
  params.yaml                [MOD]  novos blocos: labeling, model(embeddings), eval(K);
                                    registry.metric → métrica de ranking de validação
  dvc.yaml                   [MOD]  deps novas; possivelmente estágio de ETL de conteúdo
docs/
  model_card.md              [MOD]  preencher (FR-009)
specs/001-recommender-quality/
  research.md                [NEW]  Fase 0 (decisões técnicas abertas)
  data-model.md              [NEW]  Fase 1 (esquemas dos parquets/tensores)
  contracts/                 [NEW]  Fase 1 (contrato de features treino↔serving)
  quickstart.md              [NEW]  como rodar e validar os critérios de aceite
```

---

## Phase 0 — Research (`research.md`)

Decisões técnicas a fechar antes de codar (cada uma vira uma seção em `research.md` com
Decisão / Alternativas / Justificativa):

1. **Geração de negativos em escala.** Amostragem uniforme de itens vs. amostragem por
   popularidade (popularity-based negatives). Como garantir que o par (user, item)
   amostrado não seja positivo sem materializar a matriz densa (235K itens). Alvo: 4:1.
2. **Agregação as-of eficiente.** Cumulativo por usuário/item ordenado por timestamp
   (`groupby().cumcount()/cumsum().shift()`) para excluir o evento corrente, evitando
   join O(n²). Confirmar semântica de `recency_days` causal.
3. **Conjunto de candidatos na avaliação Top-K.** Ranquear contra o catálogo inteiro é
   caro; decidir estratégia (sampled metrics **por usuário**: todos os positivos de teste
   do usuário + N negativos, N~100 — 1 positivo por lista degeneraria Recall/Precision em
   HitRate) e documentar que baseline e modelo usam o **mesmo** conjunto de candidatos
   (AC-1 justa). Dimensionar o custo sob relevância ampla (views dominam o teste).
4. **Estratégia de cold-start (FR-011).** Como pontuar usuário/item novo sem embedding
   treinado — via features de conteúdo (`categoryid`) e/ou embedding "unknown" reservado
   + fallback. Definir o corte "com histórico vs. cold-start" para a segmentação.
5. **Persistência do modelo com scaler + metadados.** Hoje **não há scaler** (treino e
   serving usam features cruas); ao introduzi-lo, decidir onde persiste: `pickle` do
   `MLPRecommender` auto-contido (scaler + `n_users/n_items`) ou `state_dict` + sidecar
   JSON. Impacta `save_model`/`load_model`/`model_loader`.

---

## Phase 1 — Design & Contracts

### `data-model.md` (esquemas)

- **Dataset de treino pós-labeling**: `user_idx, item_idx, label ∈ {0,1}` + features
  contínuas causais (`frequency`, `engagement_score`, `recency_days`, `view_count`, `hour`,
  `day_of_week`) + (Fase 2) `categoryid`, `popularity_tier`. Positivos = interações fortes
  reais; negativos = pares amostrados.
- **Tensores para o NCF**: `user_idx` (long), `item_idx` (long), `x_cont` (float32
  escalado), `y` (float32). `RetailRocketDataset` retorna essa tripla.
- **Split**: cronológico, aplicado **antes** de qualquer agregação; val/test usam
  estatísticas do treino.

### `contracts/feature-contract.md` (treino ↔ serving)

Contrato explícito da ordem e transformação das features, consumido tanto pelo trainer
quanto por `recommender._build_matrix` (FR-007). Fonte única da verdade para: ordem das
colunas contínuas, colunas de id, e o scaler serializado. Um teste garante que serving e
treino produzem o mesmo vetor para o mesmo (user, item, contexto).

### Mudanças por FR

| FR | Onde | Mudança |
|----|------|---------|
| FR-001, FR-002 | `src/data/labeling.py` [NEW] | Função única `build_labeled_dataset(df_train)` → positivos + negativos amostrados; constantes de evento importadas de um só lugar |
| FR-003 | `feature_engineering.py`, `build_features.py` | Split primeiro; agregações as-of; `fit_user/item_stats(train)` reaproveitadas em val/test |
| FR-004 | `models/mlp.py` | `nn.Embedding(n_users+1,d)` + `nn.Embedding(n_items+1,d)` (último índice = unknown, D4) ⊕ `x_cont` → MLP → logit |
| FR-005/005a | `evaluation/ranking.py` [NEW], `evaluate.py` | NDCG/Recall/Precision/HitRate @{10,20}, relevância forte **e** ampla |
| FR-006 | `evaluate.py` | Baseline popularidade avaliado no mesmo split/candidatos |
| FR-007 | `serving/*`, scaler | Scaler serializado; contrato de features compartilhado |
| FR-010 | `trainer.py`, `mlp.py` | Early stopping por métrica de **validação** (val-AUC ou NDCG@20) |
| FR-011 | `evaluate.py`, cold-start strategy | Segmentar métricas; estratégia de conteúdo/fallback |
| FR-008 | `dvc.yaml`, `params.yaml` | Estágios/deps atualizados; `dvc repro` verde |
| FR-009 | `docs/model_card.md` | Preencher com dataset, métricas, baselines, limitações |

### `quickstart.md`

Passo a passo: `dvc repro` → inspecionar `metrics/eval_metrics.json` → checar AC-1
(modelo > popularidade em NDCG@{10,20} sob ambas relevâncias) e AC-2 (AUC teste ≥ 0.70).

---

## Phase 2 — Task planning approach

O `tasks.md` (gerado depois) DEVE agrupar tarefas pelas 4 fases da spec, respeitando
dependências:

- **Fase 0 — tarefa (bloqueia tudo):** labeling+negatives (FR-001/002), agregações causais
  (FR-003), ranking metrics + baseline (FR-005/005a/006). É o maior ganho isolado.
- **Fase 1 — arquitetura NCF (FR-004/010):** só depois que os dados estiverem corretos.
- **Fase 2 — escala/conteúdo/cold-start (FR-007/011 + categoryid):** depois do NCF aprender.
- **Fase 3 — tuning (embedding_dim, hidden, lr, dropout, weight_decay, razão de negativos).**
- **Transversal:** testes (fixtures sintéticas), `model_card.md`, gates de constituição.

Marcar `[P]` tarefas paralelizáveis (ex.: `ranking.py` pode ser escrito em paralelo ao
labeling; ETL de `categoryid` em paralelo ao NCF).

---

## Complexity Tracking

| Risco | Descrição | Mitigação |
|-------|-----------|-----------|
| Mudança do contrato de dados | NCF exige ids separados + candidatos por usuário na avaliação; quebra o formato plano `X/y` atual | `data-model.md` + `feature-contract.md` como fonte única; testes de contrato |
| Escala do negative sampling / ranking-eval | 235K itens × usuários pode explodir | Sampled metrics (Phase 0 item 3); amostragem vetorizada |
| Consistência treino/serving | Serving hoje monta a matriz à mão (`_build_matrix`) | FR-007 + teste de contrato compartilhado |
| Cold-start entrando no escopo | Aumenta superfície | Estratégia mínima (conteúdo + fallback) + segmentação de métricas; sem reescrever serving |

---

## Progress Tracking

- [x] Technical Context preenchido
- [x] Constitution Check definido (gates listados)
- [x] Estrutura e arquivos afetados mapeados
- [x] Phase 0 (research) descrita
- [x] Phase 1 (design/contracts) descrita
- [x] Phase 2 (abordagem de tasks) descrita
- [x] `research.md` escrito (5 decisões fechadas)
- [ ] `data-model.md` / `contracts/` / `quickstart.md` escritos (tarefas T001–T003)
- [x] `tasks.md` gerado
