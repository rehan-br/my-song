"""Tests for the M2 contrastive taste model."""

import numpy as np
import pytest

from taste_model.contrastive import ContrastiveModel


def test_fit_separates_positives_from_negatives() -> None:
    # positives cluster on one axis, negatives on another — after InfoNCE
    # training the model should score positives well above negatives.
    rng = np.random.default_rng(0)
    positives = np.array([1.0, 0.0, 0.0]) + 0.05 * rng.standard_normal((30, 3))
    negatives = np.array([0.0, 1.0, 0.0]) + 0.05 * rng.standard_normal((30, 3))

    model = ContrastiveModel().fit(positives, negatives, np.zeros(3), epochs=200)

    assert model.fitted
    assert model.score(positives).mean() > model.score(negatives).mean()
    assert model.final_loss < 1.0


def test_score_before_fit_raises() -> None:
    with pytest.raises(RuntimeError):
        ContrastiveModel().score(np.ones((1, 3)))


def test_fit_requires_positives_and_negatives() -> None:
    with pytest.raises(ValueError):
        ContrastiveModel().fit(np.empty((0, 3)), np.ones((2, 3)))
    with pytest.raises(ValueError):
        ContrastiveModel().fit(np.ones((2, 3)), np.empty((0, 3)))


def test_save_load_roundtrip(tmp_path) -> None:
    rng = np.random.default_rng(1)
    positives = rng.standard_normal((10, 4))
    negatives = rng.standard_normal((10, 4))
    model = ContrastiveModel().fit(positives, negatives, np.zeros(4), epochs=50)

    path = tmp_path / "m2.npz"
    model.save(path)
    loaded = ContrastiveModel.load(path)

    probe = rng.standard_normal((5, 4))
    np.testing.assert_allclose(model.score(probe), loaded.score(probe))
