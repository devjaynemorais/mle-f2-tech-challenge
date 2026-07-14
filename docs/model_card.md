# Model Card — Sistema de Recomendação de Produtos

> Guia de como treinar/ajustar este modelo (hiperparâmetros, arquitetura,
> fluxo de tuning): [`training.md`](training.md).

## Detalhes do Modelo

| Campo | Valor |
|-------|-------|
| **Nome** | retailrocket_recommender |
| **Tipo** | NCF — Neural Collaborative Filtering (PyTorch), multi-task |
| **Arquitetura** | Embeddings de usuário (64d), item (64d) e categoria (8d) ⊕ 8 features contínuas causais (log1p + StandardScaler) → trunk MLP [128, 64] → **duas cabeças**: `strong` (interação forte, cenário principal) e `view` (interação ampla, auxiliar) |
| **Tarefa** | Feedback implícito multi-task: interação forte (addtocart/transaction) vs. não-interação, com sinal auxiliar de `view` (spec `002-multitask-coldstart`) |
| **Cold-start** | Último índice de cada embedding = "unknown" (treinado via id-dropout 10%); categoria consultada pelo `item_idx` real (não pelo índice "unknown" — fix da spec 002); features de par `user_item_view_count`/`recency` dão sinal por (usuário, item); serving usa fallback de popularidade para usuário zero-histórico |
| **Artefato** | `model.pkl` auto-contido: rede (2 cabeças) + scaler + vocabulários/máscaras (contrato de features **v3**) |
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
| ROC-AUC | 0.971 | **0.957** ✅ (AC-2 original: ≥ 0.70) |
| Average Precision | — | 0.888 |
| F1 | — | 0.852 |
| Precisão | — | 0.914 |
| Recall | — | 0.798 |
| NDCG@20 (ranking, val) | 0.193 | — |

⚠️ **Nota de leitura**: essas métricas de classificação subiram bastante em
relação à spec `001` (AUC 0.838 → 0.957) desde que as features de par
(`user_item_view_count`/`recency`) entraram no contrato (spec `002`). Parte
do salto é **inflado por construção**: todo negativo amostrado tem, por
definição, `user_item_view_count = 0` (o par nunca foi visto), enquanto boa
parte dos positivos reais tem valor não-zero — o classificador aprende esse
atalho facilmente. Isso não é vazamento temporal (não usa o futuro), mas
também não significa que o ranking melhorou na mesma proporção — **as
métricas de ranking abaixo são a referência real** de qualidade, já que ali
todo candidato pontuado é, por construção do protocolo, um par nunca visto
(o atalho não existe nesse contexto).

## Desempenho — Ranking Top-K vs. baseline de popularidade

Protocolo: por usuário de teste, todos os positivos + 100 negativos não
vistos; mesmos candidatos para modelo e baseline; média sobre usuários.
**Warm** = usuário com histórico no treino; **cold** = sem histórico no
treino (aparece só em val/teste).

### Relevância forte (addtocart/transaction) — 585 warm / 3.594 cold

| Métrica | NCF warm | Pop warm | NCF cold | Pop cold |
|---------|----------|----------|----------|----------|
| NDCG@10 | **0.379** | 0.303 | 0.072 | 0.307 |
| NDCG@20 | **0.409** | 0.344 | 0.100 | 0.349 |
| Recall@10 | **0.470** | 0.427 | 0.114 | 0.457 |
| Recall@20 | **0.578** | 0.570 | 0.211 | 0.603 |
| HitRate@20 | **0.677** | 0.670 | 0.313 | 0.701 |

✅ **AC-1 (spec `001`, cenário principal — usuários com histórico de
treino):** o modelo supera a popularidade em NDCG e Recall para K ∈ {10, 20}
— e por uma margem maior que na versão anterior (NDCG@10 0.379 vs. 0.317).
✅ **AC-2 (spec `002`, não-regressão):** confirmado — o multi-task não só
não regrediu o cenário principal como o melhorou.

### Relevância ampla (inclui view, teto 20 positivos/usuário) — 7.399 warm / 15.213 cold

| Métrica | NCF warm | Pop warm | NCF cold | Pop cold |
|---------|----------|----------|----------|----------|
| NDCG@10 | **0.338** | 0.236 | 0.126 | 0.315 |
| NDCG@20 | **0.370** | 0.276 | 0.161 | 0.371 |
| Recall@10 | **0.400** | 0.332 | 0.152 | 0.382 |
| Recall@20 | **0.509** | 0.466 | 0.250 | 0.536 |

✅ **AC-1 (spec `002`)**: a cabeça auxiliar de `view` (multi-task) fechou o
gap — o modelo passou de **perder** para a popularidade (0.207 vs. 0.236,
spec `001`) para **ganhar com folga** (0.338 vs. 0.236). Ver
["Multi-task de `view` — spec 002"](#multi-task-de-view--spec-002) abaixo
para como isso foi alcançado (e um desvio de percurso real no meio do
caminho).

## Multi-task de `view` — spec 002

A spec [`002-multitask-coldstart`](../specs/002-multitask-coldstart/spec.md)
adicionou uma segunda cabeça de saída (`view`) compartilhando os mesmos
embeddings, treinada com sinal auxiliar de views reais amostradas
(`labeling.view_sample_ratio: 8`), com o objetivo de fechar o gap de
relevância ampla documentado na Fase 3. **Primeira tentativa regrediu o
cenário principal** — registrado aqui porque é exatamente o tipo de erro
que vale documentar:

1. **Sintoma**: com a loss combinada ingênua
   (`strong_bce + λ·view_bce`, aplicada a todas as linhas do batch), o
   cenário principal (warm, relevância forte) caiu de NDCG@10 0.317 → 0.161
   — uma regressão clara, enquanto a relevância ampla melhorou
   (0.207 → 0.332).
2. **Diagnóstico** (ablation com dados reais): desligar só o multi-task
   (`view_sample_ratio: 0`), mantendo as features de par (Frente A),
   restaurou e até melhorou o cenário principal (NDCG@10 0.341) — isolando
   a causa na Frente B (multi-task), não nas features de par.
3. **Causa raiz**: as ~528K linhas de `view` amostradas (vistas mas não
   convertidas) entravam como negativo também na loss da cabeça `strong`.
   O modelo aprendia a tratar "visto mas não comprado" como equivalente a
   "nunca visto" — uma distinção que o ranking do cenário principal pune.
4. **Correção**: mascarar essas linhas (`y_view=1 AND y=0`) da loss da
   cabeça `strong` — elas alimentam só a cabeça `view`
   (`NCFRecommender._combined_loss`). Com a correção, os dois cenários
   melhoram juntos (tabelas acima): AC-1 (spec 002) e AC-2 (não-regressão)
   ambos satisfeitos, sem trade-off entre eles.

Lição, no mesmo espírito da Fase 3 (spec `001`): um sinal auxiliar mal
delimitado — mesmo sem nenhum vazamento temporal — pode ensinar o trunk
compartilhado uma distinção errada para a tarefa principal. Isolar por
ablation qual frente causa a regressão foi o que permitiu a correção
cirúrgica em vez de reverter tudo.

## Comparação e histórico de tuning (Fase 3, spec 001)

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
combinação que funciona neste dataset. (Ver seção acima para a lição
equivalente da Fase 4 / spec 002, sobre o multi-task.)

## Limitações

- **Cold-start real** (usuário zero-histórico): o modelo puro continua
  próximo do aleatório em ranking sob relevância forte (NDCG@10 0.072 vs.
  0.307 da popularidade) — as features de par e a correção de categoria da
  spec 002 não resolvem esse caso (não há o que aprender sem nenhuma
  interação prévia). Sob relevância ampla o cold melhorou algo com o
  multi-task (0.095 → 0.126), mas segue bem abaixo da popularidade. O
  produto cobre esse segmento com o fallback de popularidade — que é forte
  (NDCG@10 ≈ 0.31–0.32). Rotear cold explicitamente para o fallback no
  serving continua sendo a recomendação operacional.
- Não modela padrões sequenciais (sessão); candidatos do serving vêm dos
  5.000 itens mais populares.
- Métricas de ranking são *sampled* (100 negativos/usuário) — têm viés
  conhecido (Krichene & Rendle, 2020); válidas para comparação interna.
- As métricas de **classificação** (ROC-AUC/AP/F1) ficaram infladas pelas
  features de par desde a spec 002 (ver nota na seção de Classificação) —
  usar as métricas de **ranking** como referência de qualidade real.

## Vieses

- **Viés de popularidade**: positivos reais concentram-se em itens
  populares; `view_count` é feature explícita.
- **Viés temporal**: split cronológico → o teste reflete tendências do fim
  do período; itens novos dependem do embedding de categoria.

## Detalhes do Treinamento

- Rótulo: positivo = addtocart/transaction; negativos = 8 por positivo,
  amostrados uniformemente do catálogo do treino, rejeitando itens já
  vistos pelo usuário; `view_count` e as features de par
  (`user_item_view_count`/`recency`) dos negativos calculadas *as-of* o
  timestamp do positivo (`src/data/labeling.py` — definição única).
- **Multi-task** (spec 002): 8 views reais amostradas por positivo forte
  como sinal auxiliar (`labeling.view_sample_ratio: 8`); loss combinada
  `strong_bce + 0.3·view_bce`, com as linhas de view amostrada
  **excluídas** da loss forte (só alimentam a cabeça `view` —
  `NCFRecommender._combined_loss`, ver seção de multi-task acima).
- Otimizador: Adam (lr 1e-3, weight_decay 1e-4); BCEWithLogitsLoss;
  batch 1024; até 40 épocas.
- Early stopping por **ROC-AUC de validação da cabeça `strong`** (paciência
  5), restaurando o melhor estado — a cabeça `view` nunca participa da
  métrica de seleção (FR-005 da spec 002); promoção no Registry por
  **val_ndcg_at_20**.
- Reprodutível via `dvc repro` (semente 42 em split, labeling e treino);
  runs rastreados no MLflow.

## Considerações Éticas

- Nenhuma informação pessoal identificável (PII) é usada nas features.
- A diversidade das recomendações deve ser monitorada para evitar bolhas
  de filtro — especialmente porque o fallback é por popularidade.
