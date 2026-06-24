# Análise Exploratória dos Dados (EDA)

## Notebook

`notebooks/01_exploratory_data_analysis.ipynb`

Executado e outputs salvos. Para re-executar, abrir no VS Code com kernel `.venv` (Python 3.13) e rodar todas as células.

---

## Estrutura do Notebook

| Seção | O que analisa |
|-------|--------------|
| Entendimento do Negócio | Contexto RetailRocket, tipos de eventos e pesos implícitos |
| Sanity Check | Shape, dtypes, nulos e duplicatas dos 3 arquivos |
| Análise Temporal | Volume diário, hora do dia, dia da semana |
| Comportamento do Usuário | Distribuição de interações, power-law, threshold de filtro |
| Popularidade de Itens | Distribuição, log-log, concentração top-N |
| Funil de Conversão | Taxas view → addtocart → transaction |
| Propriedades dos Itens | Top propriedades, cobertura de `categoryid` |
| Árvore de Categorias | Profundidade da hierarquia, categorias raiz |
| Esparsidade | Densidade da matriz usuário × item |
| Análise de Associação | Correlação entre métricas de engajamento, pairplot log1p |
| Árvore de Decisão | Feature importance para popularidade de itens |
| Sugestões de FE | Insumos para a etapa de feature engineering |

---

## Funções Utilitárias

### `src/utils/eda.py`

| Função | Descrição |
|--------|-----------|
| `sanity_check(df, name)` | Shape, dtypes, nulos e duplicatas |
| `freq_table(df, col)` | Tabela de frequência absoluta, relativa e acumulada |
| `taxa_conversao_evento(events)` | Taxas de conversão entre tipos de eventos |
| `interaction_summary(events, id_col)` | Estatísticas descritivas de interações por entidade |

### `src/utils/plots.py`

| Função | Descrição |
|--------|-----------|
| `plot_univariate(df, col)` | Histograma + KDE + Boxplot + QQ Plot |
| `plot_event_timeline(events)` | Série temporal de volume de eventos |
| `plot_power_law(series, title)` | Histograma clipped + log-log |
| `plot_conversion_funnel(events)` | Funil horizontal de conversão |
| `boxplots_por_evento(events, cols)` | Boxplots por tipo de evento |

---

## Principais Achados

### Dataset
- **~2,8M eventos** em 4,5 meses (~Mai–Out 2015)
- **~1,4M usuários** únicos · **~235K itens** únicos
- Esparsidade da matriz usuário × item: **> 99,99%**
- Pico de tráfego entre **10h–16h** (fuso UTC+3, Moscou)

### Comportamento
- Distribuição de interações segue **power-law** em usuários e itens
- Mediana de interações por usuário: **~2** (maioria de passagem)
- Usuários com ≥ 5 interações: ~20–25% dos usuários, ~70%+ dos eventos

### Funil de Conversão
| Transição | Taxa |
|-----------|------|
| view → addtocart | ~4–5% |
| addtocart → transaction | ~35–45% |

### Feature Importance (Árvore Exploratória)
1. `view` — driver principal de popularidade
2. `addtocart` — sinal de intenção complementar
3. `transaction` — menor volume, maior qualidade de sinal
4. `has_category` — impacto moderado na popularidade

---

## Decisões para o Pré-Processamento

| Decisão | Justificativa |
|---------|--------------|
| Filtrar usuários com < 5 interações | Elimina ruído sem perder volume significativo de eventos |
| Encoding `visitorid` → `user_idx` (int sequencial) | Necessário para embeddings no modelo |
| Encoding `itemid` → `item_idx` (int sequencial) | Necessário para embeddings no modelo |
| Split cronológico (últimas interações = teste) | Evita data leakage em séries temporais |
| Peso implícito: view=1, addtocart=3, transaction=5 | Reflete hierarquia de intenção de compra |
| `categoryid` como feature de conteúdo | Único metadado estruturado confiável disponível |
