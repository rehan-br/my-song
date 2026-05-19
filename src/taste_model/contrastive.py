"""M2 — aspect-weighted contrastive taste model.

Learns a per-dimension weight over the embedding space with an InfoNCE
objective: liked tracks (positives) should score higher, against the taste
centroid, than crawled / skipped tracks (negatives). Because a *uniform* weight
makes the weighted cosine identical to M1's plain cosine, M2 starts as M1 and
the training only departs from it where that helps separate the user's tracks.

The learned weight is interpretable — high-weight dimensions are the ones that
carry this user's taste.
"""

from pathlib import Path

import numpy as np


def _weighted_cosine(z: np.ndarray, centroid: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Cosine similarity of each row of ``z`` to ``centroid`` in the scaled space."""
    a = z * scale
    b = centroid * scale
    denom = np.linalg.norm(a, axis=1) * float(np.linalg.norm(b))
    denom[denom == 0] = 1.0
    return (a @ b) / denom


class ContrastiveModel:
    """Per-dimension weighted-cosine taste model, trained with InfoNCE."""

    def __init__(self) -> None:
        self.space_mean: np.ndarray | None = None
        self.centroid: np.ndarray | None = None  # unweighted, in centred space
        self.scale: np.ndarray | None = None  # learned per-dim weight (>= 0)
        self.final_loss: float = float("nan")

    @property
    def fitted(self) -> bool:
        return self.scale is not None

    def fit(
        self,
        positives: np.ndarray,
        negatives: np.ndarray,
        space_mean: np.ndarray | None = None,
        *,
        positive_weights: np.ndarray | None = None,
        epochs: int = 300,
        lr: float = 0.05,
        tau: float = 0.1,
        seed: int = 42,
    ) -> "ContrastiveModel":
        """Train the per-dimension weight by InfoNCE on positives vs negatives.

        ``positive_weights`` (one per positive — e.g. engagement-derived taste
        weights) tilt the taste centroid toward the tracks the user actually
        engages with. Uniform or omitted weights reduce to the plain mean.
        """
        import torch
        import torch.nn.functional as torch_fn

        positives = np.asarray(positives, dtype=np.float64)
        negatives = np.asarray(negatives, dtype=np.float64)
        if positives.ndim != 2 or len(positives) == 0:
            raise ValueError("fit() needs a non-empty (n, d) array of positives")
        if negatives.ndim != 2 or len(negatives) == 0:
            raise ValueError("contrastive training needs negative embeddings")

        self.space_mean = (
            positives.mean(axis=0)
            if space_mean is None
            else np.asarray(space_mean, dtype=np.float64)
        )
        zp = torch.tensor(positives - self.space_mean)
        zn = torch.tensor(negatives - self.space_mean)
        # taste direction — fixed during training; engagement-weighted if asked.
        if positive_weights is None:
            centroid = zp.mean(dim=0)
        else:
            weights = np.asarray(positive_weights, dtype=np.float64)
            if weights.shape != (len(positives),):
                raise ValueError("positive_weights must give one weight per positive")
            total = weights.sum()
            weights = weights / total if total > 0 else np.full(len(weights), 1.0 / len(weights))
            centroid = (zp * torch.tensor(weights)[:, None]).sum(dim=0)

        torch.manual_seed(seed)
        raw = torch.zeros(positives.shape[1], dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.Adam([raw], lr=lr)

        def weighted_cos(z: "torch.Tensor", scale: "torch.Tensor") -> "torch.Tensor":
            a, b = z * scale, centroid * scale
            return (a @ b) / (a.norm(dim=1) * b.norm() + 1e-12)

        loss = torch.tensor(0.0)
        for _ in range(epochs):
            optimizer.zero_grad()
            scale = torch_fn.softplus(raw)
            pos = weighted_cos(zp, scale)  # (n_pos,)
            neg = weighted_cos(zn, scale)  # (n_neg,)
            # InfoNCE: each positive should outrank every negative.
            logits = torch.cat([pos[:, None], neg[None, :].expand(len(pos), -1)], dim=1) / tau
            loss = torch_fn.cross_entropy(logits, torch.zeros(len(pos), dtype=torch.long))
            loss.backward()
            optimizer.step()

        self.scale = torch_fn.softplus(raw).detach().numpy()
        self.centroid = centroid.numpy()
        self.final_loss = float(loss.detach())
        return self

    def score(self, candidates: np.ndarray) -> np.ndarray:
        """Weighted-cosine similarity of each candidate to the taste centroid."""
        if self.scale is None or self.centroid is None or self.space_mean is None:
            raise RuntimeError("ContrastiveModel.score() called before fit()/load()")
        z = np.asarray(candidates, dtype=np.float64) - self.space_mean
        return _weighted_cosine(z, self.centroid, self.scale)

    def save(self, path: str | Path) -> None:
        """Persist the learned weight + centroid to a ``.npz`` checkpoint."""
        if self.scale is None:
            raise RuntimeError("nothing to save — model is not fitted")
        np.savez(path, space_mean=self.space_mean, centroid=self.centroid, scale=self.scale)

    @classmethod
    def load(cls, path: str | Path) -> "ContrastiveModel":
        """Load a checkpoint written by :meth:`save`."""
        data = np.load(path)
        model = cls()
        model.space_mean = data["space_mean"]
        model.centroid = data["centroid"]
        model.scale = data["scale"]
        return model
