"""M1 — centroid taste model.

The simplest baseline (CLAUDE.md): a user's taste is the mean of their liked
tracks' embeddings; candidates are ranked by cosine similarity to it.

MERT embeddings are heavily anisotropic — all vectors sit in a narrow ~0.93–0.99
cosine cone (see ``notebooks/01_embedding_sanity``). So the embedding space is
**mean-centred** first: ``fit`` and ``score`` both subtract a shared space mean,
which makes cosine genuinely discriminative.
"""

import numpy as np


class CentroidModel:
    """Mean-of-liked-embeddings taste model with space mean-centring."""

    def __init__(self) -> None:
        self._space_mean: np.ndarray | None = None
        self._centroid: np.ndarray | None = None  # unit vector in centred space

    @property
    def fitted(self) -> bool:
        return self._centroid is not None

    def fit(
        self,
        liked: np.ndarray,
        weights: np.ndarray | None = None,
        space_mean: np.ndarray | None = None,
    ) -> "CentroidModel":
        """Fit on liked-track embeddings.

        Args:
            liked: ``(n, d)`` liked-track embeddings.
            weights: optional per-track weights (e.g. ``taste_weight``); a
                weight of 0 drops a track from the centroid entirely.
            space_mean: the embedding space's mean, used for centring. Defaults
                to the liked tracks' own mean when not supplied.
        """
        liked = np.asarray(liked, dtype=np.float64)
        if liked.ndim != 2 or len(liked) == 0:
            raise ValueError("fit() needs a non-empty (n, d) array of embeddings")

        self._space_mean = (
            liked.mean(axis=0) if space_mean is None else np.asarray(space_mean, np.float64)
        )
        centred = liked - self._space_mean

        if weights is None:
            centroid = centred.mean(axis=0)
        else:
            w = np.asarray(weights, dtype=np.float64)
            if w.shape != (len(liked),):
                raise ValueError("weights must be one per liked track")
            total = w.sum()
            if total <= 0:
                raise ValueError("weights sum to zero — no liked tracks to fit on")
            centroid = (centred * w[:, None]).sum(axis=0) / total

        norm = float(np.linalg.norm(centroid))
        self._centroid = centroid / norm if norm > 0 else centroid
        return self

    def score(self, candidates: np.ndarray) -> np.ndarray:
        """Cosine similarity of each candidate to the taste centroid.

        Returns a 1-D array of scores in ``[-1, 1]``; higher is more on-taste.
        """
        if self._centroid is None or self._space_mean is None:
            raise RuntimeError("CentroidModel.score() called before fit()")
        cand = np.asarray(candidates, dtype=np.float64) - self._space_mean
        norms = np.linalg.norm(cand, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (cand / norms) @ self._centroid
