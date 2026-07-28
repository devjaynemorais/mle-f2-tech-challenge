"""Testes dos modelos baseline (DummyRecommender, LogisticRecommender)."""

import numpy as np

from src.models.baselines import DummyRecommender, LogisticRecommender


def _toy_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.random((20, 4)).astype("float32")
    y = (rng.random(20) > 0.5).astype("float32")
    return X, y


def test_dummy_recommender_fit_predict_proba_and_predict():
    X, y = _toy_data()
    model = DummyRecommender().fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (20,)
    # DummyClassifier(most_frequent) prediz a mesma classe pra tudo.
    assert len(set(proba.tolist())) == 1
    preds = model.predict(X)
    assert preds.shape == (20,)
    assert set(preds.tolist()) <= {0, 1}


def test_logistic_recommender_fit_predict_proba_and_predict():
    X, y = _toy_data()
    model = LogisticRecommender(random_state=0).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (20,)
    assert ((proba >= 0.0) & (proba <= 1.0)).all()
    preds = model.predict(X, threshold=0.5)
    assert preds.shape == (20,)
    assert set(preds.tolist()) <= {0, 1}


def test_logistic_recommender_predict_respects_threshold():
    X, y = _toy_data()
    model = LogisticRecommender(random_state=0).fit(X, y)
    proba = model.predict_proba(X)
    # threshold=0.0 classifica tudo como positivo (proba sempre > 0).
    assert model.predict(X, threshold=0.0).sum() == len(proba)
    # threshold=1.01 classifica tudo como negativo (proba sempre < 1.01).
    assert model.predict(X, threshold=1.01).sum() == 0
