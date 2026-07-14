"""Testes das métricas de ranking Top-K e do protocolo por usuário (FR-005, D3)."""

import numpy as np
import pytest

from src.evaluation.ranking import (
    evaluate_ranking_per_user,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    rank_metrics,
    recall_at_k,
    sample_unseen_items,
)

# ---------------------------------------------------------------------------
# Métricas puras — casos conhecidos calculados à mão
# ---------------------------------------------------------------------------


def test_hit_rate_at_k():
    assert hit_rate_at_k(np.array([0, 0, 1, 0]), k=3) == 1.0
    assert hit_rate_at_k(np.array([0, 0, 1, 0]), k=2) == 0.0


def test_precision_at_k():
    assert precision_at_k(np.array([1, 0, 1, 0]), k=4) == pytest.approx(0.5)
    assert precision_at_k(np.array([1, 0, 1, 0]), k=1) == pytest.approx(1.0)


def test_recall_at_k():
    assert recall_at_k(np.array([1, 0, 1, 0]), k=2, n_pos=2) == pytest.approx(0.5)
    assert recall_at_k(np.array([1, 0, 1, 0]), k=4, n_pos=2) == pytest.approx(1.0)
    assert recall_at_k(np.array([0, 0]), k=2, n_pos=0) == 0.0


def test_ndcg_at_k_known_case():
    # ranking [1, 0, 1], n_pos=2: DCG = 1 + 0.5 = 1.5; IDCG = 1 + 1/log2(3)
    ranked = np.array([1, 0, 1])
    expected = (1.0 + 1.0 / np.log2(4)) / (1.0 + 1.0 / np.log2(3))
    assert ndcg_at_k(ranked, k=3, n_pos=2) == pytest.approx(expected)


def test_ndcg_perfect_ranking_is_1():
    assert ndcg_at_k(np.array([1, 1, 0, 0]), k=4, n_pos=2) == pytest.approx(1.0)


def test_rank_metrics_orders_by_score_desc():
    scores = np.array([0.1, 0.9, 0.5])
    labels = np.array([0.0, 1.0, 0.0])  # melhor score é o positivo
    m = rank_metrics(scores, labels, ks=[1])
    assert m["hit_rate_at_1"] == 1.0
    assert m["ndcg_at_1"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Amostragem de negativos de avaliação
# ---------------------------------------------------------------------------


def test_sample_unseen_items_excludes_seen():
    catalog = np.arange(50, dtype="int64")
    excluded = set(range(40))
    rng = np.random.default_rng(0)
    sampled = sample_unseen_items(catalog, excluded, size=10, rng=rng)
    assert set(sampled.tolist()) == set(range(40, 50))


def test_sample_unseen_items_caps_at_available():
    catalog = np.arange(5, dtype="int64")
    rng = np.random.default_rng(0)
    sampled = sample_unseen_items(catalog, {0, 1}, size=10, rng=rng)
    assert sorted(sampled.tolist()) == [2, 3, 4]


# ---------------------------------------------------------------------------
# Protocolo por usuário — mesmos candidatos p/ todos os scorers (AC-1)
# ---------------------------------------------------------------------------


def _positives():
    return {1: np.array([100]), 2: np.array([101, 102])}


def test_perfect_scorer_gets_ndcg_1():
    def perfect(users, items):
        return np.isin(items, [100, 101, 102]).astype(float)

    results = evaluate_ranking_per_user(
        {"perfect": perfect},
        _positives(),
        seen_by_user={},
        # catálogo disjunto dos positivos: negativos nunca pontuam alto
        catalog=np.arange(200, 400, dtype="int64"),
        ks=[10],
        num_negatives=20,
        seed=0,
    )
    assert results["perfect"]["ndcg_at_10"] == pytest.approx(1.0)
    assert results["perfect"]["hit_rate_at_10"] == 1.0
    assert results["perfect"]["recall_at_10"] == pytest.approx(1.0)


def test_worst_scorer_gets_low_metrics():
    def worst(users, items):
        return -np.isin(items, [100, 101, 102]).astype(float)

    results = evaluate_ranking_per_user(
        {"worst": worst},
        _positives(),
        seen_by_user={},
        catalog=np.arange(200, 400, dtype="int64"),
        ks=[10],
        num_negatives=100,
        seed=0,
    )
    assert results["worst"]["ndcg_at_10"] == 0.0
    assert results["worst"]["hit_rate_at_10"] == 0.0


def test_scorers_receive_identical_candidates():
    captured: dict[str, np.ndarray] = {}

    def make_capture(name):
        def scorer(users, items):
            captured[name] = items.copy()
            return np.zeros(len(items))

        return scorer

    evaluate_ranking_per_user(
        {"a": make_capture("a"), "b": make_capture("b")},
        _positives(),
        seen_by_user={},
        catalog=np.arange(200, dtype="int64"),
        ks=[10],
        num_negatives=50,
        seed=7,
    )
    assert np.array_equal(captured["a"], captured["b"])


def test_empty_positives_returns_empty_metrics():
    results = evaluate_ranking_per_user(
        {"m": lambda u, i: np.zeros(len(i))},
        {},
        seen_by_user={},
        catalog=np.arange(10, dtype="int64"),
        ks=[10],
    )
    assert results == {"m": {}}
