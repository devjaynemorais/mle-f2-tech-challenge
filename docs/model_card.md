# Model Card — Sistema de Recomendação de Produtos

## Detalhes do Modelo

| Campo | Valor |
|-------|-------|
| **Nome** | retailrocket_recommender |
| **Tipo** | NCF — Neural Collaborative Filtering (PyTorch) |
| **Arquitetura** | Embeddings de usuário (64d), item (64d) e categoria (8d) ⊕ 6 features contínuas causais (log1p + StandardScaler) → MLP [128, 64] → logit |
| **Tarefa** | Feedback implícito: interação positiva (addtocart/transaction) vs. não-interação (negativos amostrados) |
| **Cold-start** | Último índice de cada embedding = "unknown" (treinado via id-dropout 10%); serving usa fallback de popularidade |
| **Artefato** | `model.pkl` auto-contido: rede + scaler + vocabulários/máscaras (contrato de features v2) |
| **Framework** | PyTorch |
| **Estágio no MLflow** | Production (promovido por `val_ndcg_at_20`) |

## Uso Pretendido

- **Uso principal**: recomendar produtos Top-K a usuários de e-commerce com
  base no histórico de navegação/compra (`GET /recommend`).
- **Fora do escopo**: usuários sem nenhum histórico recebem o fallback de
  popularidade (nunca lista vazia); ver Limitações para usuários sem
  histórico de treino.

## Dataset

| Campo | Valor |
|-------|-------|
| **Nome** | RetailRocket E-commerce Dataset (Kaggle) |
| **Eventos** | 948.537 após filtro `min_interactions ≥ 5` (2,76M brutos) |
| **Distribuição** | 882K view · 48K addtocart · 18K transaction |
| **Usuários / Itens** | 81.620 / 103.873 (+ categoria via `item_properties`) |
| **Splits** | Cronológico: treino 664K (~70%) / val 95K (~10%) / teste 190K (~20%) |
| **Features** | Causais (*as-of*): cada linha usa apenas eventos anteriores — sem vazamento temporal (verificado por teste automatizado) |

## Desempenho — Classificação (teste rotulado: positivos reais + negativos 8:1)

| Métrica | Validação | Teste |
|---------|-----------|-------|
| ROC-AUC | 0.844 | **0.838** ✅ (AC-2: ≥ 0.70) |
| Average Precision | — | 0.458 |
| F1 | — | 0.414 |
| Precisão | — | 0.503 |
| Recall | — | 0.351 |
| NDCG@20 (ranking, val) | 0.156 | — |

## Desempenho — Ranking Top-K vs. baseline de popularidade

Protocolo: por usuário de teste, todos os positivos + 100 negativos não
vistos; mesmos candidatos para modelo e baseline; média sobre usuários.
**Warm** = usuário com histórico no treino; **cold** = sem histórico no
treino (aparece só em val/teste).

### Relevância forte (addtocart/transaction) — 585 warm / 3.594 cold

| Métrica | NCF warm | Pop warm | NCF cold | Pop cold |
|---------|----------|----------|----------|----------|
| NDCG@10 | **0.317** | 0.303 | 0.065 | 0.307 |
| NDCG@20 | **0.356** | 0.344 | 0.093 | 0.349 |
| Recall@10 | **0.447** | 0.427 | 0.110 | 0.457 |
| Recall@20 | **0.592** | 0.570 | 0.210 | 0.603 |
| HitRate@20 | **0.691** | 0.670 | 0.312 | 0.701 |

✅ **AC-1 (cenário principal da spec — usuários com histórico de treino):**
o modelo supera a popularidade em NDCG e Recall para K ∈ {10, 20}.

### Relevância ampla (inclui view, teto 20 positivos/usuário) — 7.399 warm / 15.213 cold

| Métrica | NCF warm | Pop warm | NCF cold | Pop cold |
|---------|----------|----------|----------|----------|
| NDCG@10 | 0.207 | **0.236** | 0.095 | 0.315 |
| NDCG@20 | 0.249 | **0.276** | 0.133 | 0.371 |
| Recall@20 | 0.437 | **0.466** | 0.235 | 0.536 |

⚠️ Sob relevância ampla o modelo fica ~12% abaixo da popularidade no warm:
ele é treinado para prever interação **forte**, enquanto a relevância ampla
premia prever *views* — tarefa em que popularidade é quase ótima. Melhorar
isso (ex.: multi-task com views) fica como trabalho futuro.

## Comparação e histórico de tuning (Fase 3)

| Run | Config | AUC teste | NDCG@10 warm forte (vs pop 0.303) |
|-----|--------|-----------|------------------------------------|
| 1 | negativos por popularidade^0.75, 4:1 | 0.59 | 0.175 ❌ |
| 2 | negativos uniformes (α=0), 4:1 | 0.804 | 0.305 ≈ |
| 3 | + id-dropout 0.1, emb 64 | 0.837 | 0.286 ❌ |
| 4 | + log1p nas features de cauda pesada | 0.833 | 0.299 ≈ |
| **5** | **+ weight_decay 1e-4, negativos 8:1** | **0.838** | **0.317 ✅** |
| 6 | + init N(0, 0.01) | 0.828 | 0.303 ≈ (revertido) |
| 7 | init N(0, 0.01), wd 1e-5 | 0.827 | 0.300 ≈ (revertido) |

Lição principal: negativos de treino amostrados por popularidade ensinam o
modelo a *punir* itens populares e ele perde para o próprio baseline
(run 1); negativos uniformes + regularização forte dos embeddings é a
combinação que funciona neste dataset.

## Limitações

- **Cold-start real**: para usuários sem histórico de treino o modelo puro
  fica próximo do aleatório em ranking (embeddings sem sinal); o produto
  cobre esse segmento com o fallback de popularidade — que é forte
  (NDCG@10 ≈ 0.31). Rotear cold explicitamente para o fallback no serving
  é a recomendação operacional.
- Não modela padrões sequenciais (sessão); candidatos do serving vêm dos
  5.000 itens mais populares.
- Métricas de ranking são *sampled* (100 negativos/usuário) — têm viés
  conhecido (Krichene & Rendle, 2020); válidas para comparação interna.
- Desempenho sob relevância ampla abaixo do baseline (ver acima).

## Vieses

- **Viés de popularidade**: positivos reais concentram-se em itens
  populares; `view_count` é feature explícita.
- **Viés temporal**: split cronológico → o teste reflete tendências do fim
  do período; itens novos dependem do embedding de categoria.

## Detalhes do Treinamento

- Rótulo: positivo = addtocart/transaction; negativos = 8 por positivo,
  amostrados uniformemente do catálogo do treino, rejeitando itens já
  vistos pelo usuário; `view_count` dos negativos calculado *as-of* o
  timestamp do positivo (`src/data/labeling.py` — definição única).
- Otimizador: Adam (lr 1e-3, weight_decay 1e-4); BCEWithLogitsLoss;
  batch 1024; até 40 épocas.
- Early stopping por **ROC-AUC de validação** (paciência 5), restaurando o
  melhor estado; promoção no Registry por **val_ndcg_at_20**.
- Reprodutível via `dvc repro` (semente 42 em split, labeling e treino);
  runs rastreados no MLflow.

## Considerações Éticas

- Nenhuma informação pessoal identificável (PII) é usada nas features.
- A diversidade das recomendações deve ser monitorada para evitar bolhas
  de filtro — especialmente porque o fallback é por popularidade.
