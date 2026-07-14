# Implementation Plan: Fechar o gap de relevância ampla e fortalecer sinal de cold-start

**Feature Branch**: `002-multitask-coldstart`
**Created**: 2026-07-14
**Spec**: [`spec.md`](./spec.md)
**Status**: Draft

---

## Summary

Duas frentes independentes, priorizadas pelo diagnóstico do `model_card.md`:

1. **Multi-task loss** (FR-001/002/003/004/005): adicionar uma segunda
   cabeça de saída ao `_NCFNet` que prevê `view` (interação ampla),
   treinada junto com a cabeça de interação forte já existente, via loss
   combinada `strong_bce + λ·view_bce`. A avaliação de relevância ampla
   passa a pontuar com a cabeça de `view`, não com a de interação forte.
2. **Sinal de conteúdo/par para cold-start** (FR-006/007/008/009): corrigir
   um bug de roteamento (categoria consultada pelo índice pós-"unknown"
   em vez do `item_idx` real) e adicionar duas features causais de par
   (usuário, item): contagem e recência de views desse par específico.

As duas frentes compartilham arquivos (`mlp.py`, `feature_engineering.py`,
`labeling.py`) mas são logicamente independentes — podem ser feitas em
qualquer ordem ou em paralelo por pessoas diferentes. Este plano sequencia
a Frente 2 (cold-start) primeiro por ser mais cirúrgica e de menor risco
(não muda a função de loss nem a early-stopping), servindo de aquecimento
antes da mudança de arquitetura da Frente 1.

Diagnóstico de origem: [`docs/model_card.md`](../../docs/model_card.md)
(seções "Desempenho — Ranking Top-K" e "Limitações").

---

## Technical Context

| Item | Valor |
|------|-------|
| **Linguagem** | Python 3.13 |
| **ML** | PyTorch (`_NCFNet`/`NCFRecommender` em `src/models/mlp.py`) |
| **Orquestração** | DVC (`dvc.yaml`, 6 estágios) + MLflow (tracking/registry) |
| **Deps/build** | Poetry, ruff, pytest |
| **Dados** | RetailRocket em `data/raw/` — `train_df`/`val_df` já contêm TODOS os eventos (view incluso), só `build_labeled_dataset()` hoje filtra para positivos fortes |
| **Serving** | FastAPI (`src/serving/`) — hoje só expõe pontuação de interação forte (`strategy: "model"`) |
| **Contrato de features** | `src/data/feature_contract.py`, `CONTRACT_VERSION: int = 2` → sobe para **3** com as features de par (FR-007/008) |
| **Project type** | Pipeline ML monolítico (mesmo projeto da spec `001-recommender-quality`) |
| **Compatibilidade** | `model.pkl` treinado antes desta feature **não é compatível** (nova saída de rede, novo contrato) — todo `models/artifacts/` existente fica obsoleto após o merge; não é objetivo desta feature manter compatibilidade retroativa de artefato |

---

## Constitution Check *(gates)*

Mesmos portões adotados na spec `001-recommender-quality` (sem
`memory/constitution.md` formal no repo):

- [ ] **Sem erros de ruff** no código novo/alterado.
- [ ] **Testes não dependem de dados reais em disco** — fixtures sintéticas.
- [ ] **Sem caminhos absolutos**.
- [ ] **Definição de rótulo em um único módulo** (`src/data/labeling.py`)
  — o novo `view_label` entra ali, não duplicado em `trainer.py`/`evaluate.py`.
- [ ] **Reprodutibilidade DVC preservada** (`dvc repro` verde) e runs no
  MLflow.
- [ ] **Não-regressão dos ACs da spec anterior** (AC-2/AC-3 desta spec) —
  gate específico desta feature, verificado no Gate F1 (ver `tasks.md`).

---

## Project Structure (arquivos afetados)

```
src/
  data/
    labeling.py               [MOD]  view_label; sampling de eventos view;
                                      as-of view_count/recency do PAR p/ negativos
    feature_engineering.py    [MOD]  add_causal_pair_features() (FR-007/008)
    feature_contract.py       [MOD]  CONT_COLS ganha user_item_view_count,
                                      user_item_recency_days; CONTRACT_VERSION 2→3
  features/
    build_features.py         (sem mudança de contrato de chamada)
  models/
    mlp.py                    [MOD]  _NCFNet: 2 cabeças (strong, view);
                                      fix de roteamento de categoria (FR-006);
                                      NCFRecommender.predict_proba(head=...)
  training/
    trainer.py                [MOD]  loss combinada; early stopping continua
                                      em métrica de validação FORTE (FR-005)
  evaluation/
    scorers.py                 [MOD]  make_model_scorer(..., head="strong"|"view")
    evaluate.py                [MOD]  relevância ampla pontua com head="view"
  serving/
    recommender.py            [MOD, opcional]  se expor relevância ampla no
                                      serving (decisão em research.md D5)
configs / raiz:
  params.yaml                 [MOD]  train.view_loss_weight; labeling.view_sample_ratio
docs/
  model_card.md                [MOD]  métricas antes/depois (FR-010), nova versão
                                       de contrato, limitação de cold-start atualizada
specs/002-multitask-coldstart/
  research.md                  [NEW]  decisões técnicas (D1–D6)
  tasks.md                     [NEW]  tarefas por frente
```

> Sem `data-model.md`/`contracts/`/`quickstart.md` separados desta vez —
> o escopo é aditivo sobre um contrato já documentado
> (`specs/001-recommender-quality/contracts/feature-contract.md`); as
> mudanças de esquema estão descritas inline na tabela "Mudanças por FR"
> abaixo e a validação de aceite é a mesma checklist do `spec.md`
> (AC-1 a AC-4), sem necessidade de um quickstart dedicado.

---

## Phase 0 — Research (`research.md`)

Decisões a fechar antes de codar:

1. **Como incluir `view` no treino sem inflar o dataset 13x.** Amostrar
   uma fração dos eventos `view` reais (já causal-featurizados em
   `train_df`/`val_df`) em vez de replicar todo o volume bruto.
2. **Arquitetura das duas cabeças.** Onde o trunk compartilhado termina e
   as duas `nn.Linear` de saída começam; como isso afeta
   `predict_proba`/serialização do `model.pkl`.
3. **Peso da loss auxiliar (λ).** Ponto de partida e por que não deixar
   `view` dominar o gradiente (repetindo o erro já visto no tuning da
   spec anterior, onde um sinal mal calibrado piorou o cenário principal).
4. **Correção do roteamento de categoria.** Por que a categoria deve ser
   consultada pelo `item_idx` real, e como fazer isso sem reintroduzir
   vazamento (item de teste/serving nunca visto no treino ainda pode ter
   categoria conhecida via `content`, mas não pode "aprender" um
   embedding de item que nunca treinou).
5. **Features de par (usuário, item) para os negativos amostrados.**
   Como estender `build_item_view_times`/`_as_of_view_counts` (já
   existentes) para o nível de par sem custo quadrático.
6. **Onde a relevância ampla é servida.** Se o `GET /recommend` deve
   opcionalmente expor pontuação por `view` (fora do escopo mínimo) ou se
   o ganho desta feature fica só na avaliação offline por enquanto.

---

## Mudanças por FR

| FR | Onde | Mudança |
|----|------|---------|
| FR-001/003 | `src/data/labeling.py` | `view_label` amostrado de `events` (não só `history`); taxa controlada por `labeling.view_sample_ratio` |
| FR-002 | `src/models/mlp.py` (`_NCFNet`) | Trunk compartilhado → duas `nn.Linear` de saída (`strong_head`, `view_head`) |
| FR-004 | `src/evaluation/scorers.py`, `evaluate.py` | `make_model_scorer(model, lookup, ctx, head="view")` para relevância ampla |
| FR-005 | `src/training/trainer.py` | `_validation_metric`/early stopping continuam só com a cabeça `strong` |
| FR-006 | `src/models/mlp.py` (`_split_matrix`) | `cats = self.item_categories[<item_idx real, clipado>]`, não `self.item_categories[items roteado]` |
| FR-007/008 | `src/data/feature_engineering.py` (`add_causal_pair_features`), `feature_contract.py` | Novas colunas em `CONT_COLS`; `CONTRACT_VERSION = 3` |
| FR-009 | `src/data/labeling.py` (`_sample_negative_rows`) | Equivalente a `_as_of_view_counts` no nível de par |
| FR-010 | `docs/model_card.md` | Tabela antes/depois (warm/cold, forte/ampla) |
| FR-011 | `dvc.yaml`, `params.yaml` | Deps/params atualizados; `dvc repro` verde |

---

## Phase 2 — Task planning approach

`tasks.md` agrupa por **Frente A (cold-start)** e **Frente B (multi-task)**,
cada uma com seu próprio gate de não-regressão, mais uma fase de fechamento
comum:

- **Frente A — sinal de conteúdo/par (FR-006/007/008/009):** mais
  cirúrgica, sem tocar na loss; pode ir primeiro.
- **Frente B — multi-task view (FR-001/002/003/004/005):** muda a rede e
  o loop de treino; depende só do contrato de features estar estável
  (não depende estritamente da Frente A, mas evita retrabalho de mesclar
  mudanças no `mlp.py`/`trainer.py` em paralelo).
- **Fechamento comum:** retreinar, reavaliar AC-1 a AC-4, atualizar
  `model_card.md`, gates de constituição.

Marcar `[P]` tarefas paralelizáveis (ex.: features de par podem ser
escritas em paralelo à correção de roteamento de categoria — arquivos
diferentes dentro da Frente A).

---

## Complexity Tracking

| Risco | Descrição | Mitigação |
|-------|-----------|-----------|
| Multi-task dilui o sinal do cenário principal (AC-2) | Compartilhar embeddings entre duas tarefas pode piorar a que já funciona | λ pequeno por padrão (ver D3 do research.md); early stopping continua na cabeça forte (FR-005); Gate F1 exige AC-2 sem regressão antes de prosseguir |
| Amostragem de `view` desbalanceia o batch | 882K views vs. 66K positivos fortes, mesmo amostrados | `view_sample_ratio` parametrizado e tunável (Fase de fechamento) |
| Contrato de features muda (v2→v3) | Quebra compatibilidade com `model.pkl` antigo e com o `feature-contract.md` da spec anterior | Documentar a versão nova; nenhum artefato antigo precisa continuar servindo durante o desenvolvimento (ambiente de treino, não produção crítica) |
| Fix de roteamento de categoria muda scores para itens já "conhecidos" indiretamente | Risco baixo — só afeta itens hoje roteados a "unknown"; itens conhecidos usam o mesmo caminho de sempre | Teste de regressão comparando categoria consultada antes/depois para itens warm |

---

## Progress Tracking

- [x] Technical Context preenchido
- [x] Constitution Check definido (gates listados)
- [x] Estrutura e arquivos afetados mapeados
- [x] Phase 0 (research) descrita
- [x] Mudanças por FR mapeadas
- [x] Phase 2 (abordagem de tasks) descrita
- [ ] `research.md` escrito (6 decisões)
- [ ] `tasks.md` gerado
