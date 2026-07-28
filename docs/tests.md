# Estratégia de Testes

## Como Executar

```bash
# Todos os testes
poetry run pytest tests/ -v

# Um arquivo específico
poetry run pytest tests/test_feature_engineering.py -v

# Com cobertura
poetry run pytest tests/ --cov=src --cov-report=term-missing
```

Todos os testes são **sem dependência de disco real** — usam DataFrames sintéticos em memória (alguns usam um backend MLflow SQLite temporário em `tmp_path`, nunca um servidor no ar). Passam em qualquer clone limpo do repositório. Cobertura de linha: **100%** em todo o `src/`.

> Nota: as seções detalhadas abaixo (por teste individual) cobrem os arquivos
> originais do projeto e estão desatualizadas em relação aos arquivos novos
> adicionados depois (`test_labeling.py`, `test_ranking.py`, `test_ncf.py`,
> `test_contract.py`, os testes de cada etapa DVC, etc.) — a tabela agregada
> abaixo já reflete os 22 arquivos/200 testes atuais; o detalhamento por
> teste individual desses arquivos novos não foi escrito aqui ainda.

---

## Arquivos de Teste

| Arquivo | Escopo | Testes |
|---------|--------|--------|
| `tests/test_preprocess.py` | Estratégias de pré-processamento | 11 |
| `tests/test_preprocess_stage.py` | Etapa DVC `preprocess` (orquestração) | 2 |
| `tests/test_content_etl.py` | ETL de conteúdo (categoria por item) | 6 |
| `tests/test_feature_engineering.py` | Engenharia de features e split | 16 |
| `tests/test_build_features_stage.py` | Etapa DVC `feature_eng` (orquestração) | 9 |
| `tests/test_labeling.py` | Rótulo, negative sampling, determinismo | 18 |
| `tests/test_dataset.py` | `RetailRocketDataset` | 1 |
| `tests/test_baselines.py` | `DummyRecommender`, `LogisticRecommender` | 3 |
| `tests/test_ncf.py` | NCF: shapes, roteamento unknown, aprendizado | 14 |
| `tests/test_smoke.py` | Factory de modelos e NCF | 4 |
| `tests/test_trainer.py` | Etapa DVC `train` (orquestração + MLflow) | 15 |
| `tests/test_ranking.py` | Métricas Top-K e protocolo por usuário | 13 |
| `tests/test_scorers.py` | `HistoryFeatureLookup` e scorers | 6 |
| `tests/test_evaluate.py` | Etapa DVC `evaluate` (orquestração + MLflow) | 12 |
| `tests/test_contract.py` | Contrato de features treino ↔ serving | 3 |
| `tests/test_registry.py` | MLflow tracking + etapa DVC `promote` | 9 |
| `tests/test_serving.py` | FeatureStore, model loader, RecommendationService, API, demo | 43 |
| `tests/test_eda.py` | Helpers de EDA | 5 |
| `tests/test_plots.py` | Visualização (matplotlib/seaborn) | 6 |
| `tests/test_seed.py` | Reprodutibilidade (seeds) | 2 |
| `tests/test_logging_config.py` | Configuração central de logging | 1 |
| `tests/test_make_dataset.py` | Helper genérico de download (scaffold) | 1 |
| **Total** | | **200** |

---

## `tests/test_preprocess.py`

Testa `DefaultPreprocessor` e `RetailRocketPreprocessor` de `src/data/preprocessor.py`.

### Fixture

```python
df_eventos  # 9 linhas: 3 usuários (5, 2, 2 interações), 3 itens, timestamps ordenados
```

### DefaultPreprocessor

| Teste | O que valida |
|-------|-------------|
| `test_remove_duplicatas` | Linhas duplicadas são eliminadas |
| `test_sem_duplicatas_mantem_tamanho` | DataFrame sem duplicatas não perde linhas |
| `test_indice_resetado` | Índice do resultado é contíguo (0, 1, 2, …) |

### RetailRocketPreprocessor — filtragem

| Teste | O que valida |
|-------|-------------|
| `test_filter_remove_cold_users` | Usuários com < `min_interactions` removidos |
| `test_filter_keeps_active_users` | Usuários acima do threshold mantidos |
| `test_filter_boundary_exactly_n` | Usuário com exatamente N interações é mantido (inclusivo) |

### RetailRocketPreprocessor — encoding

| Teste | O que valida |
|-------|-------------|
| `test_encode_ids_start_from_zero` | `user_idx` e `item_idx` começam em 0 |
| `test_encode_ids_are_contiguous` | Índices são inteiros consecutivos sem lacunas |
| `test_encode_same_original_same_idx` | Mesmo `visitorid` original → mesmo `user_idx` sempre |

### RetailRocketPreprocessor — saída

| Teste | O que valida |
|-------|-------------|
| `test_output_sorted_by_timestamp` | Saída ordenada por timestamp |
| `test_output_has_required_columns` | Colunas obrigatórias presentes: `user_idx`, `item_idx`, `timestamp`, `event` |

---

## `tests/test_feature_engineering.py`

Testa todas as funções de `src/data/feature_engineering.py`.

### Fixture

```python
df_clean  # 10 linhas: 3 usuários (5, 3, 2 eventos), 3 itens, eventos variados, timestamps diários
```

### `add_event_weights`

| Teste | O que valida |
|-------|-------------|
| `test_event_weights_view` | `view` → `weight == 1` |
| `test_event_weights_addtocart` | `addtocart` → `weight == 3` |
| `test_event_weights_transaction` | `transaction` → `weight == 5` |
| `test_event_weights_column_exists` | Coluna `weight` adicionada ao DataFrame |

### `add_temporal_features`

| Teste | O que valida |
|-------|-------------|
| `test_temporal_hour_of_day` | Coluna `hour` extrai hora correta do timestamp |
| `test_temporal_day_of_week` | Coluna `day_of_week` extrai dia da semana (0=seg, 6=dom) |
| `test_temporal_does_not_drop_rows` | Número de linhas não muda |

### `compute_user_features`

| Teste | O que valida |
|-------|-------------|
| `test_compute_user_frequency` | Frequência = total de interações por `user_idx` |
| `test_compute_user_recency` | Recência = dias desde último evento até `reference_date` (normalizado por dia) |
| `test_compute_weighted_engagement` | `engagement_score` = soma dos pesos por usuário |
| `test_user_features_one_row_per_user` | Uma linha por `user_idx` no resultado |

### `compute_item_features`

| Teste | O que valida |
|-------|-------------|
| `test_item_view_count` | `view_count` = número de eventos `view` por `item_idx` |
| `test_item_popularity_tier_column_exists` | Coluna `popularity_tier` gerada |
| `test_item_popularity_tier_valid_values` | `popularity_tier` ∈ `{long_tail, mid_tier, top_tier}` |
| `test_item_features_one_row_per_item` | Uma linha por `item_idx` no resultado |

### `build_interaction_features`

| Teste | O que valida |
|-------|-------------|
| `test_build_features_has_weight` | Resultado contém coluna `weight` |
| `test_build_features_has_temporal` | Resultado contém `hour` e `day_of_week` |
| `test_build_features_has_user_features` | Resultado contém `frequency`, `recency_days`, `engagement_score` |
| `test_build_features_has_item_features` | Resultado contém `view_count` e `popularity_tier` |
| `test_build_features_preserves_row_count` | Número de linhas = número de eventos originais |

### `chronological_split`

| Teste | O que valida |
|-------|-------------|
| `test_chronological_split_order` | `max(train.timestamp) ≤ min(val.timestamp) ≤ min(test.timestamp)` |
| `test_split_proportions` | Proporções train/val/test dentro de ±1 evento do esperado |
| `test_no_future_leakage` | `min(test.timestamp) ≥ max(train.timestamp)` |

---

## `tests/test_smoke.py`

Testa o funcionamento básico da factory de modelos e do MLP.

| Teste | O que valida |
|-------|-------------|
| `test_modelos_registrados_na_factory` | `mlp`, `logistic` e `dummy` registrados na Factory |
| `test_factory_cria_mlp` | `ModelFactory.create("mlp")` retorna instância de `RecommenderBase` |
| `test_factory_nome_invalido_lanca_erro` | `ValueError` para modelo inexistente |
| `test_mlp_fit_predict_com_dados_sinteticos` | MLP treina e prediz em arrays sintéticos (100×4) |

---

## `tests/test_registry.py`

Testa `find_best_model_run`, `register_model` e `promote_model` de
`src/utils/mlflow_tracking.py`. Usa um backend `sqlite:///` em `tmp_path` com
`monkeypatch` sobre `settings.mlflow_tracking_uri` — sem servidor MLflow.

| Teste | O que valida |
|-------|-------------|
| `test_find_best_model_run_escolhe_maior_auc` | Retorna o run com maior `val_auc` dentre vários |
| `test_find_best_model_run_experimento_inexistente` | `ValueError` quando o experimento não existe |
| `test_register_e_promote_para_production` | `register_model` + `promote_model` deixam a versão em `Production` |

---

## `tests/test_serving.py`

Testa `FeatureStore`, `model_loader`, `RecommendationService` e os endpoints da
API. `TestClient(app)` é usado **sem** `with`, então o `lifespan` não dispara
(sem MLflow/disco); o serviço é injetado em `api.state["service"]`.

### FeatureStore

| Teste | O que valida |
|-------|-------------|
| `test_store_user_and_item_counts` | `n_users` / `n_items` corretos |
| `test_store_user_features_and_seen` | `has_user`, `user_features`, `seen_items` |
| `test_store_candidates_are_popularity_sorted` | candidatos, `view_counts` e `popular_items` ordenados por popularidade |

### model_loader

| Teste | O que valida |
|-------|-------------|
| `test_load_from_local_reads_record_and_unpickles` | fallback local lê `promoted_model.json` e desserializa |
| `test_registry_reachable_false_for_closed_port` | porta fechada → `False` (falha rápida) |
| `test_registry_reachable_true_for_non_http_scheme` | `file://` → `True` |
| `test_registry_reachable_true_when_listening` | socket em escuta → `True` |

### RecommendationService

| Teste | O que valida |
|-------|-------------|
| `test_recommend_ranks_and_excludes_seen` | ranqueia por score e exclui itens já vistos |
| `test_recommend_unknown_user_uses_popularity` | cold start → `strategy: "popularity"` |
| `test_recommend_scores_are_descending` | scores em ordem decrescente |
| `test_recommend_orders_multiple_candidates_descending` | ordenação com múltiplos candidatos |
| `test_recommend_known_user_seen_all_falls_back_to_popularity` | usuário que viu tudo → popularidade |

### Endpoints

| Teste | O que valida |
|-------|-------------|
| `test_health_reports_loaded` | `/health` reporta `status/model_loaded/n_users/n_items` |
| `test_recommend_endpoint_happy_path` | `/recommend` retorna 200 e recomendações do modelo |
| `test_recommend_top_k_out_of_range_returns_422` | `top_k` fora de `[1,100]` → 422 |
| `test_recommend_without_service_returns_503` | serviço não carregado → 503 |

---

## Filosofia de Testes

**Sem dependência de disco.** Todos os testes usam `pd.DataFrame` sintético construído diretamente nas fixtures. Nenhum teste lê arquivos de `data/` nem escreve nada.

**Funções puras.** As funções em `feature_engineering.py` não têm efeitos colaterais, o que torna os testes determinísticos e rápidos.

**TDD.** Os testes foram escritos antes da implementação:
1. `tests/` escritos → pytest coletava erros de importação (RED)
2. `src/data/feature_engineering.py` implementado → todos passam (GREEN)

**Cobertura.** 100% de cobertura de linha em todo o `src/` (`pytest --cov=src --cov-report=term-missing`), incluindo as etapas de orquestração do pipeline DVC (`preprocess`, `feature_eng`, `train`, `evaluate`, `promote`) e os utilitários (`eda.py`, `plots.py`, `seed.py`, `logging_config.py`, `mlflow_tracking.py`).
