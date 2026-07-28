# Tasks: Recomendador que aprende preferências e é medido como recomendação

**Feature Branch**: `001-recommender-quality`
**Spec**: [`spec.md`](./spec.md) · **Plan**: [`plan.md`](./plan.md) · **Research**: [`research.md`](./research.md)

Convenções: `[P]` = paralelizável (arquivos independentes). IDs `T0xx`. Cada tarefa cita o
FR/decisão que atende. **Ordem por dependência** — não pular fases.

---

## Fase de Design (pré-código) — completar Phase 1 do plano

- [ ] **T001** `[P]` Escrever `data-model.md`: esquema do dataset rotulado (positivos +
  negativos), tensores do NCF (ids long / contínuas float), e regra de split→agregação.
- [ ] **T002** `[P]` Escrever `contracts/feature-contract.md`: ordem/tipos das colunas de
  id e contínuas + transformação (scaler) — fonte única para treino e serving (FR-007).
- [ ] **T003** `[P]` Escrever `quickstart.md`: como rodar `dvc repro` e checar AC-1/AC-2.

---

## Fase 0 — Reformular a tarefa *(maior alavanca; bloqueia Fase 1)*

- [ ] **T010** Criar `src/data/labeling.py` com definição **única** de rótulo: positivo =
  addtocart/transaction; centralizar `EVENT_WEIGHTS`/threshold aqui (FR-002).
- [ ] **T011** Implementar negative sampling por popularidade^0.75, razão 4:1, com rejeição
  de itens já vistos por usuário e semente fixa (FR-001, D1). *(depende de T010)*
- [ ] **T012** Refatorar `src/features/build_features.py`: **split cronológico ANTES** do
  fit de scaler/estatísticas de referência (FR-003, D2). *Nota: quem elimina o vazamento
  das agregações é T013 (as-of); T012 garante que nada seja "fitado" fora do treino.*
- [ ] **T013** Refatorar `src/data/feature_engineering.py`: agregações *as-of*
  (`cumcount`/`cumsum().shift`, recência vs. evento anterior) — **é aqui que o vazamento
  da Seção 4 morre**; stats/scaler fitados só no treino (FR-003, D2). *(depende de T012)*
- [ ] **T014** `[P]` Criar `src/evaluation/ranking.py`: `ndcg@k`, `recall@k`, `precision@k`,
  `hit_rate@k` (puros, testáveis) (FR-005).
- [ ] **T015** Reescrever avaliação por usuário em `src/evaluation/evaluate.py`: **todos os
  positivos de teste do usuário** + 100 negativos numa mesma lista (nunca 1 pos/lista —
  degeneraria Recall/Precision), K∈{10,20}, relevância forte **e** ampla (ampla com teto de
  20 positivos/usuário), mesmo conjunto p/ baseline de popularidade (FR-005/005a/006, D3).
  *(depende de T014)*
- [ ] **T016** `[P]` Testes de Fase 0 com fixtures **sintéticas** (sem dados reais):
  ausência de vazamento (AC-3), balanço de negativos, métricas de ranking em caso conhecido.

**Gate F0:** `dvc repro` verde até `evaluate`; ranking metrics saindo p/ modelo atual e
baseline (ainda pode empatar — objetivo é a infra correta).

---

## Fase 1 — Arquitetura NCF *(depende de Fase 0)*

- [ ] **T020** Reescrever `src/models/mlp.py`: `nn.Embedding(n_users+1,d)` +
  `nn.Embedding(n_items+1,d)` ⊕ contínuas → MLP → logit; **último índice** (`n_users`/
  `n_items`) = "unknown" — não usar o índice 0, que já é um id real do `factorize`
  (FR-004, D4).
- [ ] **T021** Adaptar `src/data/dataset.py`: retornar `(user_idx long, item_idx long,
  x_cont float, y)` conforme `data-model.md` (FR-004). *(depende de T001)*
- [ ] **T022** Atualizar `src/training/trainer.py`: consumir labeling + dataset novo; passar
  `n_users`/`n_items`/`embedding_dim` via factory/`params.yaml`.
- [ ] **T023** Early stopping por **métrica de validação** (val-AUC ou NDCG@20), não pela
  loss de treino (FR-010).
- [ ] **T024** Atualizar `params.yaml`: blocos `labeling`, `model` (embedding_dim, hidden),
  `eval` (K, candidatos, teto de positivos p/ relevância ampla); atualizar
  `registry.metric` (hoje `val_auc`) para a métrica de validação de ranking usada na
  seleção (ex.: `val_ndcg_at_20`), mantendo a promoção coerente com FR-010; refletir deps
  em `dvc.yaml` (FR-008).
- [ ] **T025** `[P]` Testes do NCF: forward com ids conhecidos/unknown; shapes; overfit em
  amostra minúscula (sanidade de aprendizado).

**Gate F1:** modelo **supera o baseline de popularidade** em NDCG@{10,20} (AC-1) e ROC-AUC
de teste **≥ 0.70** (AC-2). *Se não bater, iterar antes da Fase 2.*

---

## Fase 2 — Escala, conteúdo e cold-start *(depende de Fase 1)*

- [ ] **T030** Serializar `scaler` + metadados dentro do `model.pkl` (auto-contido); ajustar
  `save_model`/`load_model` (FR-007, D5).
- [ ] **T031** Alinhar serving ao contrato: `recommender._build_matrix`, `store.py`,
  `model_loader.py` usam o mesmo scaler/ordem de features (FR-007). *(depende de T030, T002)*
- [ ] **T032** `[P]` ETL de conteúdo: extrair `categoryid` de `data/raw/item_properties_*`
  (e `popularity_tier` já calculado) → estágio/deps no `dvc.yaml`.
- [ ] **T033** Adicionar embedding de `categoryid` ao item tower (FR-004/011, D4).
  *(depende de T032, T020)*
- [ ] **T034** Cold-start na avaliação: segmentar métricas `warm` vs `cold`; roteamento p/
  "unknown"/fallback (FR-011, D4). *(depende de T015)*
- [ ] **T035** `[P]` Teste de contrato treino↔serving: mesmo vetor p/ mesmo (user,item,contexto).

**Gate F2:** métricas estáveis com conteúdo; segmento cold-start reportado sem erro.

---

## Fase 3 — Tuning *(depende de Fase 1/2)*

- [ ] **T040** Busca de `embedding_dim`, `hidden_dims`, `lr`, `dropout`, `weight_decay`,
  razão de negativos; runs no MLflow, seleção pela métrica de validação.
- [ ] **T041** Escolher melhor run e promover (estágio `promote`/registry já existente).

---

## Transversal / fechamento

- [ ] **T050** Preencher `docs/model_card.md`: dataset, métricas treino/val/teste (ambas
  relevâncias), comparação com baselines, limitações, cold-start (FR-009).
- [ ] **T051** `[P]` Rodar gates de constituição: `ruff` limpo, testes sem dados reais, sem
  caminhos absolutos, threshold em módulo único (ver `plan.md` → Constitution Check).
- [ ] **T052** Atualizar `docs/feature_selection.md` (Seção 8) e `docs/architecture.md` se o
  fluxo DVC mudar.

---

## Grafo de dependências (resumo)

```
T001/T002/T003 (design)
        │
T010 → T011 ┐
T012 → T013 ┼→ Gate F0 → T020 → T021/T022/T023/T024 → Gate F1
T014 → T015 ┘                         │
                                      ├→ T030 → T031
                                      ├→ T032 → T033
                                      └→ T034 → Gate F2 → T040 → T041 → T050
```

Paralelizáveis com segurança: **T001/T002/T003**, **T014**, **T016**, **T025**, **T032**,
**T035**, **T051**.
