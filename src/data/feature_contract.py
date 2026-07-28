"""Contrato de features treino ↔ serving (FR-007).

Fonte única da verdade para a ordem das colunas que formam a matriz de
entrada dos modelos. Trainer, avaliação e serving importam daqui — nunca
redeclarar listas de colunas em outro módulo.

O escalonamento das colunas contínuas acontece DENTRO do modelo (o scaler
é ajustado no fit e serializado junto no model.pkl — decisão D5), portanto
quem monta a matriz entrega valores crus nesta ordem.

Documentação: specs/001-recommender-quality/contracts/feature-contract.md
"""

from __future__ import annotations

# Colunas de identidade (consumidas como índices de embedding pelo NCF).
ID_COLS: list[str] = ["user_idx", "item_idx"]

# Colunas contínuas causais (valores conhecidos ATÉ o instante do evento).
CONT_COLS: list[str] = [
    "hour",
    "day_of_week",
    "frequency",
    "engagement_score",
    "recency_days",
    "view_count",
    "user_item_view_count",
    "user_item_recency_days",
]

# Ordem completa da matriz de entrada: ids primeiro, contínuas depois.
FEATURE_COLS: list[str] = ID_COLS + CONT_COLS

# Coluna alvo do dataset rotulado (positivo=1 / negativo amostrado=0).
TARGET_COL: str = "label"

# Coluna alvo auxiliar (multi-task, spec 002): 1 = par (user, item) com
# QUALQUER interação observada (view/addtocart/transaction), 0 = negativo
# amostrado. Coexiste com TARGET_COL, não o substitui.
VIEW_TARGET_COL: str = "view_label"

# Sentinela de recency_days/user_item_recency_days para "sem evento anterior".
NO_HISTORY_RECENCY: float = -1.0

# Versão do contrato — gravada nos metadados do modelo serializado.
# v3 (spec 002): + user_item_view_count, user_item_recency_days (par causal).
CONTRACT_VERSION: int = 3
