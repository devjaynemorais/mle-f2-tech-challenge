# Quickstart — validar os critérios de aceite

**Feature**: `001-recommender-quality` · **Spec**: [`spec.md`](./spec.md)

## Pré-requisitos

- Dataset RetailRocket em `data/raw/` (`events.csv`,
  `item_properties_part{1,2}.csv`).
- Ambiente Poetry instalado (`poetry install`).
- (Opcional) MLflow server em `http://localhost:5000` — sem ele, os runs
  são logados no store local e o serving usa o fallback
  `models/promoted_model.json`.

## 1. Rodar o pipeline completo

```bash
dvc repro
```

Estágios: `preprocess` → `content` (ETL de categoryid) → `feature_eng`
(features causais + split cronológico) → `train` (labeling + NCF) →
`evaluate` → `promote`.

## 2. Checar os critérios de aceite

```bash
dvc metrics show          # ou: cat metrics/eval_metrics.json
```

- **AC-1** (modelo > popularidade): em `metrics/eval_metrics.json`, para
  cada relevância (`ranking.strong` e `ranking.broad`) e K ∈ {10, 20}:
  `model.overall.ndcg_at_K > popularity.overall.ndcg_at_K` e idem para
  `recall_at_K`.
- **AC-2** (AUC ≥ 0.70): `classification.roc_auc ≥ 0.70` no teste.
- **AC-3** (sem vazamento): garantido por teste automatizado —
  `pytest tests/test_feature_engineering.py -k leakage`.

Segmentação cold-start (FR-011): `ranking.<relevância>.model.warm` vs
`.cold`, com contagens em `n_users`.

## 3. Gates de qualidade

```bash
ruff check src tests   # constituição: zero erros
pytest                 # 100% verde, fixtures sintéticas (sem dados reais)
```

## 4. Servir e testar a API

```bash
uvicorn src.serving.api:app --reload
curl "http://localhost:8000/recommend?user_id=42&top_k=10"
```

- Usuário conhecido → `strategy: "model"`.
- Usuário sem histórico ou candidatos esgotados → `strategy: "popularity"`
  (nunca lista vazia).
