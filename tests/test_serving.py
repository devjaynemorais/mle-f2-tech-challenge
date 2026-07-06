"""Testes da API de serving (store, recommender, loader, endpoints)."""

import numpy as np
import pandas as pd
import pytest

from src.serving.store import FeatureStore


def _sample_interactions() -> pd.DataFrame:
    # user 1 viewed items 10, 11; user 2 viewed item 11. view_counts: 11>10>12.
    return pd.DataFrame(
        {
            "user_idx": [1, 1, 2],
            "item_idx": [10, 11, 11],
            "frequency": [2, 2, 1],
            "engagement_score": [4.0, 4.0, 1.0],
            "recency_days": [3, 3, 5],
            "view_count": [5, 9, 9],
        }
    )


def _store_with_item12() -> FeatureStore:
    df = _sample_interactions()
    extra = pd.DataFrame(
        {
            "user_idx": [3],
            "item_idx": [12],
            "frequency": [1],
            "engagement_score": [1.0],
            "recency_days": [1],
            "view_count": [1],
        }
    )
    return FeatureStore(pd.concat([df, extra], ignore_index=True))


def test_store_user_and_item_counts():
    store = _store_with_item12()
    assert store.n_users == 3
    assert store.n_items == 3


def test_store_user_features_and_seen():
    store = _store_with_item12()
    assert store.has_user(1) is True
    assert store.has_user(99) is False
    assert store.user_features(1) == (2.0, 4.0, 3.0)
    assert store.seen_items(1) == {10, 11}


def test_store_candidates_are_popularity_sorted():
    store = _store_with_item12()
    candidates = store.candidate_items(max_candidates=10)
    assert list(candidates) == [11, 10, 12]
    assert list(store.view_counts(np.array([11, 10, 12]))) == [9, 5, 1]
    assert store.popular_items(2) == [(11, 9), (10, 5)]
