"""Tests for the M1 centroid taste model."""

import numpy as np
import pytest

from taste_model.centroid import CentroidModel


def test_aligned_candidate_scores_highest() -> None:
    # liked tracks all point ~[1, 0, 0]; a candidate in that direction wins
    liked = np.array([[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [1.0, 0.1, 0.1]])
    model = CentroidModel().fit(liked, space_mean=np.zeros(3))

    scores = model.score(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]))
    assert np.argmax(scores) == 0  # aligned candidate
    assert np.argmin(scores) == 2  # opposite candidate


def test_weights_pull_the_centroid() -> None:
    liked = np.array([[1.0, 0.0], [0.0, 1.0]])
    heavy_x = CentroidModel().fit(liked, weights=np.array([10.0, 1.0]), space_mean=np.zeros(2))
    heavy_y = CentroidModel().fit(liked, weights=np.array([1.0, 10.0]), space_mean=np.zeros(2))

    probe = np.array([[1.0, 0.0]])
    assert heavy_x.score(probe)[0] > heavy_y.score(probe)[0]


def test_zero_weight_track_is_ignored() -> None:
    liked = np.array([[1.0, 0.0], [0.0, 1.0]])
    model = CentroidModel().fit(liked, weights=np.array([1.0, 0.0]), space_mean=np.zeros(2))
    # the second track has weight 0 -> centroid is purely the first
    assert model.score(np.array([[1.0, 0.0]]))[0] > 0.99


def test_centring_uses_the_space_mean() -> None:
    # with a non-zero space mean, scoring is relative to the centred space
    liked = np.array([[2.0, 0.0], [2.0, 0.2]])
    model = CentroidModel().fit(liked, space_mean=np.array([1.0, 0.1]))
    # a candidate at the space mean has zero centred vector -> score 0
    assert abs(model.score(np.array([[1.0, 0.1]]))[0]) < 1e-9


def test_fit_rejects_empty_and_score_needs_fit() -> None:
    with pytest.raises(ValueError):
        CentroidModel().fit(np.empty((0, 4)))
    with pytest.raises(RuntimeError):
        CentroidModel().score(np.ones((1, 4)))
