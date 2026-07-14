# Como treinar o modelo

Guia focado só no estágio `train` (o NCF): como rodar, o que cada
hiperparâmetro faz, como ajustar e reproduzir, e como ler o resultado.
Complementa o [`README.md`](../README.md) (setup, pipeline completo, API) e
o [`model_card.md`](model_card.md) (resultados finais e histórico de
tuning já concluído). Arquitetura multi-task (cabeça `view`) e features de
par (usuário, item): [`specs/002-multitask-coldstart/`](../specs/002-multitask-coldstart/spec.md).

---

## 1. Pré-requisitos

- `data/processed/{train,val,test}.parquet` já existem (rode `preprocess` →
  `content` → `feature_eng` antes — ver [`README.md`](../README.md#pipeline-dvc)).
- MLflow Tracking Server no ar (`make mlflow`), apontado por
  `MLFLOW_TRACKING_URI` no `.env`. O treino falha com conexão recusada se
  não estiver rodando.

## 2. Rodar o treino

```bash
make train                          # poetry run python -m src.training.trainer
# ou, respeitando as dependências do DVC:
dvc repro train                     # só treina se algo relevante mudou
dvc repro -s train --force          # força retreino mesmo sem mudança
```

No Windows, os dois cuidados de sempre se aplicam (ver
[`README.md`](../README.md#windowspowershell--dois-cuidados-no-dvc-repro)):

```powershell
$env:PATH = "$PWD\.venv\Scripts;$env:PATH"; $env:PYTHONUTF8 = "1"; dvc repro train
```

Cada execução cria um novo run no MLflow (experimento
`recommendation_system`, nome do run = `mlflow.run_name` em `params.yaml`)
e salva o artefato em `models/artifacts/<run_id>/model.pkl`. Rodar de novo
**não sobrescreve** runs anteriores — o `promote` escolhe o melhor entre
todos pela métrica do registry (seção 6).

## 3. O que o estágio faz (`src/training/trainer.py`)

1. Carrega `train.parquet`/`val.parquet` (já com features causais).
2. Rotula ambos os splits com `build_labeled_dataset()`
   (`src/data/labeling.py`): mantém os positivos reais (addtocart/
   transaction) e amostra negativos — a validação usa o `train` como
   histórico para calcular `view_count` *as-of* corretamente.
3. Instancia o modelo via `ModelFactory.create(train.model_type, ...)`.
4. Treina com early stopping por **ROC-AUC de validação**
   (`NCFRecommender.fit(X_train, y_train, X_val, y_val)`).
5. Calcula `val_auc` e `val_ndcg_at_20` (ranking Top-K de validação —
   ver seção 5) e loga ambos no MLflow.
6. Salva o artefato auto-contido (rede + `StandardScaler` + vocabulários)
   e `metrics/train_metrics.json`.

## 4. Hiperparâmetros (`params.yaml` → bloco `train`)

| Chave | Default | O que faz | Quando mexer |
|-------|---------|-----------|---------------|
| `model_type` | `ncf` | Modelo da Factory (`ncf` \| `logistic` \| `dummy`) | `logistic`/`dummy` só para baseline/debug rápido — não usam embeddings |
| `epochs` | 40 | Nº máximo de épocas | Raramente — o early stopping já corta antes |
| `batch_size` | 1024 | Tamanho do mini-batch (Adam) | Subir se GPU/CPU com folga e quiser treinar mais rápido |
| `learning_rate` | 0.001 | LR do Adam | Baixe se a loss oscilar/divergir |
| `weight_decay` | 0.0001 | L2 do Adam — regulariza os embeddings | **Parâmetro mais sensível do tuning** (ver model_card §Histórico): embeddings sem sinal precisam encolher para ~0, senão viram ruído no ranking |
| `embedding_dim` | 64 | Dimensão dos embeddings de usuário/item | Maior = mais capacidade, mais risco de overfit com poucos dados |
| `cat_embedding_dim` | 8 | Dimensão do embedding de categoria | Categoria é sinal auxiliar (cold-start de conteúdo); manter pequeno |
| `hidden_dims` | `[128, 64]` | Camadas ocultas do MLP após a concatenação | Lista de qualquer tamanho — `[256, 128, 64]` etc. |
| `dropout` | 0.2 | Dropout entre as camadas do MLP | Subir se overfit (val_auc cai enquanto train loss continua caindo) |
| `unknown_dropout` | 0.1 | Fração de ids de treino roteada p/ o índice "unknown" a cada época | **Não zere** — é o que treina o embedding de cold-start; sem isso usuários/itens novos recebem vetor aleatório no serving |
| `view_loss_weight` | 0.3 | Peso (λ) da loss auxiliar de `view` somada à loss principal (multi-task — spec `002-multitask-coldstart`) | Só tem efeito se `labeling.view_sample_ratio > 0`. **Mantenha baixo** — a mesma lição do `popularity_alpha` (sinal auxiliar mal calibrado pode piorar o cenário principal); ver `specs/002-multitask-coldstart/research.md` D3 |
| `early_stopping_patience` | 5 | Épocas sem melhora no val-AUC antes de parar | Subir se a curva de val-AUC for ruidosa/demorar a convergir |
| `random_state` | 42 | Semente (split de shuffle do DataLoader + init da rede) | Fixo para reprodutibilidade; mude só para checar variância entre seeds |

Bloco `labeling` (também consumido pelo `train`, controla como os exemplos
são construídos antes de chegar no modelo):

| Chave | Default | O que faz |
|-------|---------|-----------|
| `num_negatives` | 8 | Negativos amostrados por positivo (proporção 8:1) |
| `popularity_alpha` | 0.0 | Peso de popularidade na amostragem de negativos. `0` = uniforme. **Não volte para 0.75** sem reler `research.md` D1 / `model_card.md`: negativos amostrados por popularidade ensinam o modelo a *punir* itens populares — o run 1 do tuning (AUC 0.59) perdeu feio para o baseline de popularidade por causa disso |
| `seed` | 42 | Semente da amostragem de negativos |
| `view_sample_ratio` | 8 | Eventos `view` reais amostrados por positivo forte, como sinal auxiliar de multi-task (spec `002-multitask-coldstart`) | `0` desliga o multi-task por completo (nenhuma coluna `view_label` é criada) — comportamento idêntico ao anterior à essa spec |

Bloco `eval` (usado pelo `train` só para calcular `val_ndcg_at_20`; o
detalhe completo da avaliação fica no estágio `evaluate` — ver
[`README.md`](../README.md#métricas-de-avaliação)):

| Chave | Default | O que faz |
|-------|---------|-----------|
| `k_values` | `[10, 20]` | Valores de K para as métricas de ranking |
| `num_candidate_negatives` | 100 | Negativos por usuário no ranking sampled |
| `seed` | 42 | Semente da amostragem de candidatos |

## 5. Arquitetura (`src/models/mlp.py` — `NCFRecommender`)

```
user_idx ──► Embedding(n_users+1, 64) ──┐
item_idx ──► Embedding(n_items+1, 64) ──┤                     ┌──► strong_head ──► logit ──► sigmoid
cat_idx  ──► Embedding(n_cats+1,   8) ──┼──► concat ──► trunk ┤
features contínuas (8) ──► log1p (cauda pesada) ──► StandardScaler ──┘  └──► view_head   ──► logit ──► sigmoid
```

- **Duas cabeças de saída** (multi-task, spec `002-multitask-coldstart`):
  o trunk compartilhado (embeddings + MLP `hidden_dims`) alimenta duas
  `nn.Linear` independentes — `strong_head` (interação forte, cenário
  principal) e `view_head` (interação ampla, auxiliar). Treinadas juntas
  quando `labeling.view_sample_ratio > 0` (loss combinada — seção 6);
  `predict_proba(X, head="strong"|"view")` escolhe qual cabeça consultar.
  Com `view_sample_ratio = 0` (ou chamando `fit()` sem `y_view`), a
  `view_head` fica com a inicialização aleatória, sem receber gradiente —
  comportamento idêntico ao de uma única cabeça.
- **Índice "unknown"**: o último índice de cada tabela de embedding
  (`n_users`, `n_items`, `n_categories` — não `0`, que já é um id real do
  `pd.factorize`) é reservado para ids fora do vocabulário de treino.
  `_route_ids()` faz esse roteamento tanto no treino (ids de validação
  desconhecidos) quanto no serving (usuários/itens novos). A **categoria**
  de um item roteado para "unknown" é consultada pelo `item_idx` **real**
  (clipado), não pelo índice já roteado — um item ausente do treino mas
  com categoria conhecida (estágio `content`) não perde esse sinal só
  porque seu embedding de item é o "unknown" compartilhado.
- **`unknown_dropout`**: sem treinar esse índice explicitamente ele nunca
  recebe gradiente e fica com o vetor de inicialização aleatória. A cada
  batch, uma fração dos ids *conhecidos* do treino é redirecionada para
  "unknown" de propósito, para o embedding aprender a representar o
  "usuário/item médio".
- **Features de par (usuário, item)**: `user_item_view_count` e
  `user_item_recency_days` (contrato v3) dão sinal específico do par —
  "este usuário já viu este item antes" — além das agregações por
  usuário/item isoladamente. Úteis principalmente para usuários com pouco
  histórico agregado mas que já interagiram com o item em questão; não
  ajudam usuário verdadeiramente zero-histórico (esse continua coberto
  pelo fallback de popularidade no serving).
- **`log1p` nas features de cauda pesada** (`frequency`,
  `engagement_score`, `view_count`): sem isso a distribuição power-law do
  e-commerce esmaga o sinal útil numa faixa estreita de z-score.
- **Artefato auto-contido**: o `StandardScaler` é ajustado dentro de
  `fit()` e serializado junto no `model.pkl` — avaliação e serving nunca
  reimplementam a normalização, eliminando risco de train/serving skew.

## 6. Early stopping e métrica de promoção

- **Early stopping** (dentro do `fit`): a cada época, calcula ROC-AUC na
  validação **da cabeça `strong`** — o sinal auxiliar de `view` nunca
  entra na métrica de seleção, mesmo em multi-task; se não melhorar por
  `patience` épocas seguidas, para e restaura os pesos da melhor época
  (`_run_early_stopping` guarda um `deepcopy` do `state_dict`).
- **Métrica de promoção** (`registry.metric` em `params.yaml`):
  `val_ndcg_at_20`, calculada *depois* do fit, com o ranking Top-K
  sampled sobre os usuários de validação (`val_ranking_metric()`). É
  diferente da métrica do early stopping de propósito — early stopping
  usa AUC (barata, calculada a cada época); promoção usa NDCG@20 porque é
  mais próxima da métrica de negócio (ranking, não classificação).
- O `promote` (próximo estágio) não promove automaticamente o run que
  você acabou de treinar — ele varre **todos os runs** do experimento e
  promove o melhor histórico por `val_ndcg_at_20`. Para forçar a promoção
  do run mais recente mesmo que não seja o melhor, troque
  `mlflow.experiment_name` (começa um experimento limpo) ou apague os
  runs antigos na UI do MLflow.

## 7. Onde olhar o resultado

- **Console**: última linha do log —
  `Run MLflow: <run_id> | val_auc=0.XXXX | val_ndcg@20=0.XXXX`.
- **`metrics/train_metrics.json`**: mesmas duas métricas + `model_type`,
  em disco (é o que o `dvc metrics show` também lê).
- **MLflow UI** (`http://localhost:5000`): aba **Experiments** →
  `recommendation_system` → o run mais recente tem todos os
  hiperparâmetros logados (`mlflow.log_params`) e o artefato
  `model/model.pkl` anexado — útil para comparar runs lado a lado.

## 8. Fluxo de tuning

1. Edite `params.yaml` (bloco `train`, `labeling` ou `eval`).
2. `dvc repro` — o DVC detecta que só os params do `train` mudaram (via
   `dvc.yaml` → `train.params`) e reexecuta **só esse estágio**
   (`evaluate`/`promote` rodam em seguida porque dependem do `train`).
3. Compare `val_ndcg_at_20` entre runs na UI do MLflow, ou
   `metrics/train_metrics.json` antes/depois.
4. Depois de fechar a config, rode `evaluate` (métricas no teste, warm/cold,
   forte/ampla — ver [`README.md`](../README.md#métricas-de-avaliação))
   antes de decidir promover.

O histórico completo de 7 rodadas de tuning já feitas neste projeto
(inclusive duas tentativas revertidas) está documentado em
[`model_card.md`](model_card.md#comparação-e-histórico-de-tuning-fase-3) —
vale ler antes de repetir um experimento parecido, principalmente:
negativos por popularidade (perdeu para o baseline), e inicialização
`N(0, 0.01)` dos embeddings (piorou o AUC neste dataset — revertida).

## 9. Troubleshooting específico do treino

| Sintoma | Causa provável | Correção |
|---------|------------------|----------|
| `train` falha com conexão recusada | MLflow não está de pé | `make mlflow` antes |
| `val_auc` ≈ 0.5 (não aprende) | `popularity_alpha` voltou a ser > 0, ou dados de entrada mudaram | Confirme `labeling.popularity_alpha: 0.0`; confira `data/processed/train.parquet` |
| `val_ndcg_at_20` cai mas `val_auc` sobe | Modelo melhorando em classificação mas não em ranking (candidatos diferentes) | Normal em algum grau — a métrica de promoção é o NDCG, confie nela para decidir |
| Treino muito lento na CPU | `batch_size` pequeno ou `epochs` alto sem early stopping ajudando | Suba `batch_size`; confirme que `early_stopping_patience` não está artificialmente alto |
| `ModuleNotFoundError: pydantic_settings` | `python` do PATH ≠ venv (Windows) | Ver seção 2 / [`README.md`](../README.md#windowspowershell--dois-cuidados-no-dvc-repro) |
| Erro de dimensão no forward do NCF | `hidden_dims` vazio ou tipo errado no YAML | `hidden_dims` precisa ser uma lista, ex. `[128, 64]` |
