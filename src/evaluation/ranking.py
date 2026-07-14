"""Métricas de ranking Top-K e protocolo de avaliação por usuário (FR-005, D3).

Métricas puras (arrays → float), testáveis sem dados reais, e o protocolo
*sampled ranking*: todos os positivos de teste do usuário + N negativos não
vistos numa mesma lista de candidatos, ranqueada pelo scorer avaliado.
O MESMO conjunto de candidatos (mesma semente) deve ser usado por modelo e
baseline (AC-1).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

# Assinatura de um scorer de pares: (users, items) → scores.
PairScorer = Callable[[np.ndarray, np.ndarray], np.ndarray]

_MAX_RESAMPLE_ROUNDS = 50


def hit_rate_at_k(ranked_labels: np.ndarray, k: int) -> float:
    """HitRate@K: 1.0 se há ao menos um positivo no top-K.

    Args:
        ranked_labels: Rótulos binários na ordem do ranking (melhor primeiro).
        k: Tamanho do corte.

    Returns:
        0.0 ou 1.0.
    """
    return float(np.any(ranked_labels[:k] > 0))


def precision_at_k(ranked_labels: np.ndarray, k: int) -> float:
    """Precision@K: fração de positivos no top-K.

    Args:
        ranked_labels: Rótulos binários na ordem do ranking.
        k: Tamanho do corte.

    Returns:
        Valor em [0, 1].
    """
    return float(np.sum(ranked_labels[:k] > 0) / k)


def recall_at_k(ranked_labels: np.ndarray, k: int, n_pos: int) -> float:
    """Recall@K: fração dos positivos do usuário recuperada no top-K.

    Args:
        ranked_labels: Rótulos binários na ordem do ranking.
        k: Tamanho do corte.
        n_pos: Total de positivos do usuário na lista.

    Returns:
        Valor em [0, 1] (0.0 se não há positivos).
    """
    if n_pos == 0:
        return 0.0
    return float(np.sum(ranked_labels[:k] > 0) / n_pos)


def ndcg_at_k(ranked_labels: np.ndarray, k: int, n_pos: int) -> float:
    """NDCG@K com ganhos binários.

    Args:
        ranked_labels: Rótulos binários na ordem do ranking.
        k: Tamanho do corte.
        n_pos: Total de positivos do usuário na lista.

    Returns:
        Valor em [0, 1] (0.0 se não há positivos).
    """
    if n_pos == 0:
        return 0.0
    gains = ranked_labels[:k].astype("float64")
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg = float(np.sum(gains * discounts))
    ideal = min(n_pos, k)
    idcg = float(np.sum(discounts[:ideal]))
    return dcg / idcg


def rank_metrics(
    scores: np.ndarray, labels: np.ndarray, ks: Sequence[int]
) -> dict[str, float]:
    """Calcula as 4 métricas de ranking para cada K, dada uma lista pontuada.

    Args:
        scores: Score de cada candidato.
        labels: Rótulo binário de cada candidato (alinhado a scores).
        ks: Valores de K a reportar.

    Returns:
        Dict ``{metric}_at_{k}`` → valor.
    """
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order]
    n_pos = int(np.sum(labels > 0))
    out: dict[str, float] = {}
    for k in ks:
        out[f"ndcg_at_{k}"] = ndcg_at_k(ranked, k, n_pos)
        out[f"recall_at_{k}"] = recall_at_k(ranked, k, n_pos)
        out[f"precision_at_{k}"] = precision_at_k(ranked, k)
        out[f"hit_rate_at_{k}"] = hit_rate_at_k(ranked, k)
    return out


def sample_unseen_items(
    catalog: np.ndarray,
    excluded: set[int],
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Amostra ``size`` itens do catálogo fora do conjunto excluído.

    Amostragem uniforme com rejeição e sem reposição (protocolo padrão de
    sampled metrics — He et al., NCF).

    Args:
        catalog: Ids de item elegíveis.
        excluded: Itens a excluir (vistos + positivos do usuário).
        size: Quantidade de negativos desejada.
        rng: Gerador de números aleatórios.

    Returns:
        Array de item_idx únicos, não excluídos (pode ser menor que ``size``
        se o catálogo não tiver itens suficientes).
    """
    available = len(catalog) - len(excluded)
    size = min(size, max(available, 0))
    chosen: set[int] = set()
    for _ in range(_MAX_RESAMPLE_ROUNDS):
        if len(chosen) >= size:
            break
        # replace=True + dedupe: O(size) por rodada, independente do catálogo.
        draw = rng.integers(0, len(catalog), size=2 * (size - len(chosen)))
        for item in catalog[draw]:
            if len(chosen) >= size:
                break
            if int(item) not in excluded:
                chosen.add(int(item))
    return np.fromiter(chosen, dtype="int64", count=len(chosen))


def _candidate_lists(
    positives_by_user: dict[int, np.ndarray],
    seen_by_user: dict[int, set[int]],
    catalog: np.ndarray,
    num_negatives: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Monta as listas de candidatos de todos os usuários num único batch."""
    users_arr: list[np.ndarray] = []
    items_arr: list[np.ndarray] = []
    labels_arr: list[np.ndarray] = []
    slices: list[tuple[int, int]] = []
    offset = 0
    for user in sorted(positives_by_user):
        pos = positives_by_user[user]
        excluded = set(seen_by_user.get(user, set())) | {int(i) for i in pos}
        negs = sample_unseen_items(catalog, excluded, num_negatives, rng)
        items = np.concatenate([pos.astype("int64"), negs])
        labels = np.concatenate(
            [np.ones(len(pos)), np.zeros(len(negs))]
        )
        # Permutação para não beneficiar posições fixas em caso de empate
        # de score (a popularidade empata com frequência).
        perm = rng.permutation(len(items))
        users_arr.append(np.full(len(items), user, dtype="int64"))
        items_arr.append(items[perm])
        labels_arr.append(labels[perm])
        slices.append((offset, offset + len(items)))
        offset += len(items)
    return (
        np.concatenate(users_arr),
        np.concatenate(items_arr),
        np.concatenate(labels_arr),
        slices,
    )


def evaluate_ranking_per_user(
    scorers: dict[str, PairScorer],
    positives_by_user: dict[int, np.ndarray],
    seen_by_user: dict[int, set[int]],
    catalog: np.ndarray,
    ks: Sequence[int],
    num_negatives: int = 100,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Avalia um ou mais scorers no MESMO conjunto de candidatos por usuário.

    Para cada usuário: candidatos = todos os seus positivos + ``num_negatives``
    itens não vistos; métricas por usuário e média sobre usuários (D3).

    Args:
        scorers: Nome → scorer de pares (ex.: {"model": ..., "popularity": ...}).
        positives_by_user: user_idx → array de item_idx positivos no teste.
        seen_by_user: user_idx → itens já vistos (histórico completo).
        catalog: Itens elegíveis como negativos (catálogo do treino).
        ks: Valores de K (ex.: [10, 20]).
        num_negatives: Negativos por usuário (default 100).
        seed: Semente — candidatos idênticos entre scorers (AC-1).

    Returns:
        Dict nome do scorer → dict de métricas médias ``{metric}_at_{k}``.
    """
    if not positives_by_user:
        return {name: {} for name in scorers}
    rng = np.random.default_rng(seed)
    users, items, labels, slices = _candidate_lists(
        positives_by_user, seen_by_user, catalog, num_negatives, rng
    )
    results: dict[str, dict[str, float]] = {}
    for name, scorer in scorers.items():
        scores = np.asarray(scorer(users, items), dtype="float64")
        per_user = [
            rank_metrics(scores[a:b], labels[a:b], ks) for a, b in slices
        ]
        results[name] = {
            key: float(np.mean([m[key] for m in per_user])) for key in per_user[0]
        }
    return results
