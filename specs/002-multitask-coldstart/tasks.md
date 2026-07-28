# Tasks: Fechar o gap de relevância ampla e fortalecer sinal de cold-start

**Feature Branch**: `002-multitask-coldstart`
**Spec**: [`spec.md`](./spec.md) · **Plan**: [`plan.md`](./plan.md) · **Research**: [`research.md`](./research.md)

Convenções: `[P]` = paralelizável (arquivos independentes). IDs `T1xx`
(Frente A — cold-start), `T2xx` (Frente B — multi-task), `T3xx`
(fechamento comum). Cada tarefa cita o FR/decisão que atende.

> **Status (2026-07-14): CONCLUÍDA.** Todas as tarefas fechadas (T304 fica
> deliberadamente em aberto — trabalho futuro fora do escopo desta spec).
> 95 testes sintéticos passando, `ruff check`/`format --check` limpos,
> validado com `dvc repro` ponta a ponta em dados reais. AC-1 e AC-2
> confirmados juntos após corrigir uma regressão real encontrada no
> caminho (ver `research.md` D3, revisão empírica, e
> `docs/model_card.md` § "Multi-task de view").

---

## Frente A — Sinal de conteúdo/par (cold-start) *(pode ir primeiro; menor risco)*

- [x] **T101** Corrigir `NCFRecommender._split_matrix` em `src/models/mlp.py`:
  consultar `item_categories` pelo `item_idx` real (clipado), não pelo
  índice pós-roteamento (FR-006, D4).
- [x] **T102** `[P]` Teste de regressão: item ausente do treino mas com
  categoria conhecida (via `content`) preserva o `cat_idx` correto na
  pontuação; item sem categoria continua caindo em `UNKNOWN_CAT_IDX`
  (cenário 3 do spec.md).
- [x] **T103** Implementar `build_user_item_view_times()` em
  `src/data/labeling.py` — análogo a `build_item_view_times()`, mas
  chaveado por `(user_idx, item_idx)` (FR-009, D5).
- [x] **T104** Implementar `add_causal_pair_features()` em
  `src/data/feature_engineering.py`: `user_item_view_count` e
  `user_item_recency_days` (*as-of*, mesmo sentinela `NO_HISTORY_RECENCY`
  do `recency_days` existente) (FR-007/008).
- [x] **T105** Atualizar `src/data/feature_contract.py`: adicionar as duas
  colunas a `CONT_COLS`; `CONTRACT_VERSION` 2 → 3. *(depende de T104)*
- [x] **T106** Estender `_sample_negative_rows()` em `labeling.py` para
  calcular `user_item_view_count`/`user_item_recency_days` *as-of* dos
  pares sintéticos, usando T103 (FR-009). *(depende de T103, T105)*
- [x] **T107** `[P]` Testes de Frente A com fixtures sintéticas: ausência
  de vazamento nas features de par (mesmo padrão de teste já usado para
  `view_count`), sentinela de par nunca visto.

**Gate FA:** `dvc repro` verde até `feature_eng`/`train`; features de par
aparecem no parquet processado e no `model.pkl`; teste T102 passa.

---

## Frente B — Multi-task `view` *(depende do contrato estável — pode começar após Gate FA ou em paralelo se coordenado)*

- [x] **T201** Reescrever `_NCFNet` em `src/models/mlp.py`: trunk
  compartilhado + duas cabeças (`strong_head`, `view_head`); `forward()`
  retorna `(strong_logit, view_logit)` (FR-002, D2).
- [x] **T202** `NCFRecommender.predict_proba(X, head="strong")`: parâmetro
  `head` seleciona qual cabeça pontua; default preserva todo o código
  existente que chama sem esse argumento. *(depende de T201)*
- [x] **T203** Adicionar `view_label`/amostragem de eventos `view` em
  `build_labeled_dataset()` (`src/data/labeling.py`), taxa controlada por
  `labeling.view_sample_ratio` (FR-001/003, D1).
- [x] **T204** Atualizar `NCFRecommender.fit()`/loop de treino
  (`src/models/mlp.py`): loss combinada
  `strong_bce + λ·view_bce`, `λ = train.view_loss_weight` (FR-002, D3).
  *(depende de T201, T203)*
- [x] **T205** Confirmar que early stopping (`_validation_metric`,
  `trainer.py::val_ranking_metric`) permanece calculado só sobre a cabeça
  `strong` — **não** misturar com `view` na métrica de seleção (FR-005).
  *(depende de T204)*
- [x] **T206** Atualizar `src/evaluation/scorers.py::make_model_scorer`
  para aceitar `head="strong"|"view"`; `evaluate.py` passa a pontuar
  relevância ampla com `head="view"` e relevância forte com
  `head="strong"` (FR-004, D6). *(depende de T202)*
- [x] **T207** Atualizar `params.yaml`: `train.view_loss_weight` (default
  0.3), `labeling.view_sample_ratio` (default igual a `num_negatives`);
  refletir deps em `dvc.yaml` se necessário.
- [x] **T208** `[P]` Testes de Frente B: shapes das duas cabeças; loss
  combinada diminui em amostra pequena; `predict_proba(head=...)` retorna
  scores diferentes por cabeça; early stopping ignora `view_head`.

**Gate FB:** `dvc repro` verde ponta a ponta; AC-2/AC-3 (não-regressão do
cenário principal, spec `001-recommender-quality`) confirmados antes de
aceitar qualquer valor de λ diferente do default.

> **Nota (2026-07-14):** o Gate FB pegou uma regressão real na primeira
> tentativa (loss combinada sem máscara — ver `research.md` D3, revisão
> empírica). Corrigido mascarando as linhas de `view` amostrado da loss
> `strong`; AC-1 e AC-2 confirmados juntos após a correção, validado com
> `dvc repro` em dados reais (não só testes sintéticos).

---

## Fechamento comum

- [x] **T301** Retreinar com as duas frentes integradas; conferir AC-1
  (relevância ampla, warm, paridade/vantagem vs. popularidade) e AC-4
  (redução do gap cold vs. popularidade, relevância forte). Resultado:
  AC-1 ✅ (NDCG@10 broad/warm 0.338 vs. pop 0.236), AC-2 ✅ (NDCG@10
  strong/warm 0.379 vs. pop 0.303, melhor que a spec 001), AC-4 parcial
  (broad/cold melhorou 0.095→0.126; strong/cold ~inalterado, esperado —
  fallback de popularidade continua cobrindo zero-histórico).
- [x] **T302** Atualizar `docs/model_card.md`: tabela antes/depois
  (warm/cold × forte/ampla), nova versão de contrato (v3), seção de
  limitações revisada (FR-010), seção dedicada ao achado/correção do
  multi-task.
- [x] **T303** `[P]` Rodar gates de constituição: `ruff check`/`format
  --check` limpos nos arquivos alterados, 95 testes sintéticos passando,
  sem caminhos absolutos, `view_label` centralizado em `labeling.py`.
  Adicional (não previsto originalmente): limpeza dos runs incompatíveis
  (contrato v2) no MLflow Registry, que quase causaram uma promoção
  quebrada — ver `docs/model_card.md`.
- [ ] **T304** Se `research.md` D6 for revisitado (expor relevância ampla
  na API), abrir uma spec separada — não expandir esta feature no meio do
  fechamento.

---

## Grafo de dependências (resumo)

```
T101 → T102
T103 → T104 → T105 → T106 → T107  ┐
                                    ├→ Gate FA ┐
T201 → T202 ┐                                  │
T203        ┼→ T204 → T205 → T206 → T207 → T208 ┼→ Gate FB → T301 → T302 → T303
             ┘                                  │
                                                (T101/T105 alimentam T201+
                                                 se o contrato v3 entrar
                                                 antes da Frente B)
```

Paralelizáveis com segurança: **T102**, **T107**, **T208**; Frente A e
Frente B podem rodar em paralelo se coordenadas para não conflitar em
`mlp.py` (T101 mexe em `_split_matrix`, T201/T204 mexem em `_NCFNet`/`fit`
— arquivos iguais, métodos diferentes; mesclar com cuidado ou serializar).
