"""Tests for the M3 manifold (VAE) model."""

import numpy as np
import pytest

from taste_model.manifold import ManifoldModel


def test_on_manifold_scores_higher_than_off() -> None:
    # liked tracks lie on a 2-d manifold inside an 8-d space; the VAE should
    # give on-manifold points a better ELBO than off-manifold noise.
    rng = np.random.default_rng(0)
    projection = rng.standard_normal((2, 8))
    liked = rng.standard_normal((40, 2)) @ projection + 0.01 * rng.standard_normal((40, 8))

    model = ManifoldModel().fit(liked, epochs=300, latent_dim=4, hidden=32)

    on_manifold = rng.standard_normal((5, 2)) @ projection
    off_manifold = 3.0 * rng.standard_normal((5, 8))
    assert model.score(on_manifold).mean() > model.score(off_manifold).mean()


def test_sample_returns_embeddings() -> None:
    rng = np.random.default_rng(1)
    model = ManifoldModel().fit(rng.standard_normal((20, 8)), epochs=40, latent_dim=4, hidden=16)
    assert model.sample(7).shape == (7, 8)


def test_score_before_fit_raises() -> None:
    with pytest.raises(RuntimeError):
        ManifoldModel().score(np.ones((1, 8)))


def test_save_load_roundtrip(tmp_path) -> None:
    rng = np.random.default_rng(2)
    model = ManifoldModel().fit(rng.standard_normal((20, 8)), epochs=40, latent_dim=4, hidden=16)
    path = tmp_path / "m3.pt"
    model.save(path)
    loaded = ManifoldModel.load(path)

    probe = rng.standard_normal((5, 8))
    np.testing.assert_allclose(model.score(probe), loaded.score(probe), rtol=1e-4)
