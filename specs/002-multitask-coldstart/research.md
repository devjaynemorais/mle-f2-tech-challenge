# Research: decisões técnicas (Fase 0)

**Feature**: `002-multitask-coldstart`
**Plan**: [`plan.md`](./plan.md)

Fecha as 6 decisões abertas na Fase 0 do plano. Formato por decisão:
**Decisão / Justificativa / Alternativas consideradas**.

---

## D1 — Como incluir `view` no treino sem inflar o dataset 13x

**Decisão.** `train_df`/`val_df` (saída do `feature_eng`) já contêm **todos**
os eventos, incluindo `view` — hoje `build_labeled_dataset()` só extrai os
positivos fortes (`is_positive`). Adicionar uma segunda extração: amostrar
uma fração `labeling.view_sample_ratio` (default proposto: **igual ao
`num_negatives`**, isto é, ~8 linhas de `view` por positivo forte, para
manter a mesma ordem de grandeza dos dois sinais auxiliares no batch) dos
eventos `view` do split, com semente fixa. Cada linha amostrada recebe
`view_label = 1`, `label = 0` (a menos que o mesmo par também seja um
positivo forte em outra linha — não há conflito, são linhas/eventos
diferentes). Os negativos sintéticos já existentes (`_sample_negative_rows`)
recebem `view_label = 0`; os positivos fortes recebem `view_label = 1`
(interação forte implica engajamento amplo).

**Justificativa.** Reaproveita eventos causais já materializados (nenhum
recálculo de agregação); a razão configurável evita que 882K views afoguem
os 66K positivos fortes num único epoch, mantendo o custo de treino na
mesma ordem de grandeza de hoje (que já é 8 negativos : 1 positivo).

**Alternativas.** (a) Usar 100% dos eventos `view` — descartada, infla o
dataset em ~13x e desequilibra o gradiente na direção do sinal auxiliar.
(b) Não usar eventos reais de `view`, e sim redefinir os negativos
amostrados como "positivos fracos" — descartada, perde o sinal real de
que aquele usuário efetivamente visitou aquele item.

---

## D2 — Arquitetura das duas cabeças de saída

**Decisão.** `_NCFNet` para de terminar em `nn.Linear(prev, 1)` e passa a
expor um **trunk compartilhado** (embeddings + camadas ocultas do
`hidden_dims`) seguido de **duas cabeças lineares independentes**:
`self.strong_head = nn.Linear(hidden_dims[-1], 1)` e
`self.view_head = nn.Linear(hidden_dims[-1], 1)`. O `forward()` retorna
`(strong_logit, view_logit)`. `NCFRecommender.predict_proba(X, head="strong")`
ganha o parâmetro `head` (`"strong"` continua sendo o default, preservando
compatibilidade com todo o código existente que já chama
`predict_proba(X)` sem argumento extra).

**Justificativa.** Duas cabeças sobre um trunk compartilhado é o desenho
padrão de multi-task learning quando as tarefas usam a mesma
representação latente (aqui, "o quanto esse usuário se interessa por esse
item") — mais barato que dois modelos separados, e força os embeddings a
aprender uma representação útil para ambas as granularidades de
interesse (visita vs. conversão).

**Alternativas.** (a) Modelo totalmente separado para `view` — descartada,
dobra custo de treino/serving e embeddings não compartilham sinal (perde
justamente o benefício de multi-task). (b) Cabeça única com 2 logits via
`nn.Linear(prev, 2)` — equivalente em capacidade a duas `nn.Linear(prev,1)`,
mas menos legível/nomeável no código; sem vantagem prática, mantido como
duas camadas nomeadas.

---

## D3 — Peso da loss auxiliar (λ)

**Decisão.** `loss = BCEWithLogitsLoss(strong_logit, label) + λ · BCEWithLogitsLoss(view_logit, view_label)`,
com `λ = train.view_loss_weight`, **default proposto 0.3** (sinal auxiliar
presente mas subordinado). Ajustável em `params.yaml`; validado no Gate F1
(tasks.md) exigindo que AC-2/AC-3 (não-regressão do cenário principal) não
quebrem antes de aceitar qualquer valor de λ.

**Justificativa.** A spec anterior já documentou um caso concreto de sinal
auxiliar mal calibrado destruindo o resultado principal (negativos por
popularidade^0.75 fizeram o AUC cair de ~0.84 para 0.59 — ver D1 de
`specs/001-recommender-quality/research.md`). λ pequeno por padrão é a
postura conservadora coerente com essa lição: introduzir o sinal de `view`
sem deixá-lo dominar o gradiente compartilhado.

**Alternativas.** (a) λ=1 (pesos iguais) — mais arriscado, adiado para
tuning se λ=0.3 não fechar o AC-1 desta spec. (b) Scheduling de λ
(começar baixo, subir ao longo do treino) — complexidade desnecessária
para a primeira iteração.

> **Revisão empírica (Gate FB).** λ=0.3 sozinho **não bastou** — a primeira
> implementação (loss combinada aplicada a TODAS as linhas do batch,
> inclusive as de `view` amostrado) regrediu o cenário principal (NDCG@10
> warm forte 0.317 → 0.161), mesmo com λ baixo. Ablation em dados reais
> (desligar `view_sample_ratio` mantendo as features de par) isolou a causa
> na loss, não no valor de λ: linhas de `view` amostrado (`y_view=1,
> label=0` — visto mas não convertido) estavam entrando como negativo
> também da loss `strong`, ensinando o modelo a tratar "visto mas não
> comprado" como "nunca visto". **Decisão revista**: essas linhas são
> mascaradas da loss `strong` (só alimentam `view_bce`) —
> `NCFRecommender._combined_loss`. Com a máscara, λ=0.3 funciona como
> previsto: AC-1 (broad) e AC-2 (não-regressão do warm/forte) passam
> juntos. Ver `docs/model_card.md` § "Multi-task de view" para os números.

---

## D4 — Correção do roteamento de categoria (FR-006)

**Decisão.** Em `NCFRecommender._split_matrix`, a consulta de categoria
passa a usar o **`item_idx` real** (antes do roteamento para "unknown"),
clipado ao intervalo válido (`[0, n_items)`) — não mais o `items` já
roteado. Concretamente: `cats = self.item_categories[np.clip(X[:, 1].astype("int64"), 0, self.n_items)]`,
em vez de `cats = self.item_categories[items]` (onde `items` já é o
índice pós-`_route_ids`, que colapsa qualquer item desconhecido para
`n_items`, cuja entrada em `item_categories` é sempre a categoria
"unknown" por construção).

**Por que isso é seguro (não reintroduz vazamento).** O **embedding do
item** continua roteado para "unknown" normalmente — o modelo não passa a
"conhecer" um item que nunca treinou. Só a **categoria** (sinal de
conteúdo, independente de ter visto aquele item específico) deixa de ser
descartada. `item_categories` já é construído a partir de
`train_df ∪ val_df` (`_vocab_kwargs` em `trainer.py`) mais o mapeamento do
estágio `content`, então um item ausente do treino mas presente no
`content` (ou em `val_df`) já tem sua categoria correta na tabela — o bug
é só a consulta usar o índice errado.

**Justificativa.** É a causa raiz mais concreta encontrada para o
desperdício de sinal de conteúdo em itens cold: o dado já existe e já é
carregado no modelo, só não é lido corretamente.

**Alternativas.** (a) Centróide de embedding por categoria (ideia inicial
discutida antes da inspeção do código) — mais complexo e ataca um sintoma
diferente (inicialização do embedding de item, não da leitura da
categoria); fica como possível trabalho futuro se D4 sozinho não for
suficiente para mover o AC-4.

---

## D5 — Features de par (usuário, item) para os negativos amostrados

**Decisão.** Estender o padrão já usado para `view_count` (função
`build_item_view_times`/`_as_of_view_counts` em `labeling.py`) para o
nível de par: `build_user_item_view_times(history)` mapeia
`(user_idx, item_idx) → array de timestamps de view`, e
`_as_of_pair_features(users, items, timestamps, pair_times)` calcula, por
`searchsorted`, tanto a contagem quanto a recência causal do par para os
negativos sintéticos — mesma técnica O(log n) por consulta já usada para
`view_count`, sem custo quadrático.

**Justificativa.** Reaproveita o padrão de implementação existente
(mesma função `np.searchsorted` sobre arrays ordenados por chave), então
o custo assintótico e o risco de bug são baixos — é extensão de um código
já testado, não uma técnica nova.

**Alternativas.** Calcular via `groupby` denso `(user_idx, item_idx)` —
mais caro em memória para pares únicos na escala do dataset (esparsidade
>99,99% significa poucos pares repetidos, mas a estrutura de dicionário
por chave composta é direta o suficiente para não precisar de otimização
adicional agora).

---

## D6 — Onde a relevância ampla é servida

**Decisão.** Nesta feature, o sinal de `view` (cabeça auxiliar) é
consumido **apenas na avaliação offline** (`evaluate.py`, relevância
ampla). O endpoint `GET /recommend` continua usando a cabeça `strong`
(comportamento inalterado do serving) — expor a cabeça de `view` como uma
segunda estratégia de recomendação (ex.: parâmetro `relevance=broad`) fica
registrado como trabalho futuro, não nesta spec (ver "Fora de escopo" no
`spec.md`).

**Justificativa.** Mantém o escopo da feature focado em fechar o gap
*medido* (offline, no `model_card.md`); mudar a API pública é uma decisão
de produto separada (qual estratégia mostrar ao usuário final) que não
precisa bloquear a correção do sinal de treino/avaliação.

**Alternativas.** Expor os dois scores na resposta da API desde já —
descartado por aumentar o escopo sem necessidade comprovada de produto.

---

## Impacto nos requisitos

| Decisão | Fecha/habilita |
|---------|-------------------|
| D1 | FR-001, FR-003 |
| D2 | FR-002 |
| D3 | FR-002, protege AC-2/AC-3 |
| D4 | FR-006, habilita AC-4 |
| D5 | FR-007, FR-008, FR-009 |
| D6 | FR-004 (escopo: só avaliação offline por ora) |

Todas as ambiguidades da Fase 0 estão resolvidas → liberado para `tasks.md`.
