"""M3 — manifold taste model (variational autoencoder).

Models the user's liked-track embeddings as a *distribution*. A small VAE is
trained on the liked (centred) MERT embeddings; a candidate is scored by its
ELBO — how plausibly it sits on the learned taste manifold — and the decoder
can *sample* imagined tracks in the user's taste.

``essence_siblings``, when present, add a term pulling sibling pairs together
in latent space (CLAUDE.md: siblings should share higher joint likelihood).

Caveat: a liked set of ~100 tracks is small for a 1024-d VAE — M3 is as much a
research probe as a finished model. The eval gate decides if it earns promotion.
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as torch_fn


class _VAE(nn.Module):
    """A minimal MLP variational autoencoder."""

    def __init__(self, dim: int, hidden: int, latent: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU())
        self.to_mu = nn.Linear(hidden, latent)
        self.to_logvar = nn.Linear(hidden, latent)
        self.decoder = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, dim))

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(x)
        return self.to_mu(hidden), self.to_logvar(hidden)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.decoder(z), mu, logvar


class ManifoldModel:
    """VAE over the liked-track embedding distribution; scores by ELBO."""

    def __init__(self) -> None:
        self.space_mean: np.ndarray | None = None
        self._vae: _VAE | None = None
        self._dims: tuple[int, int, int] = (0, 0, 0)  # (dim, hidden, latent)
        self.final_loss: float = float("nan")

    @property
    def fitted(self) -> bool:
        return self._vae is not None

    def fit(
        self,
        liked: np.ndarray,
        space_mean: np.ndarray | None = None,
        *,
        latent_dim: int = 16,
        hidden: int = 256,
        epochs: int = 400,
        lr: float = 1e-3,
        beta: float = 1e-3,
        sibling_pairs: list[tuple[int, int]] | None = None,
        seed: int = 42,
    ) -> "ManifoldModel":
        """Train the VAE on liked embeddings (optionally with sibling pairs)."""
        liked = np.asarray(liked, dtype=np.float64)
        if liked.ndim != 2 or len(liked) == 0:
            raise ValueError("fit() needs a non-empty (n, d) array of embeddings")

        self.space_mean = (
            liked.mean(axis=0) if space_mean is None else np.asarray(space_mean, dtype=np.float64)
        )
        dim = liked.shape[1]
        self._dims = (dim, hidden, latent_dim)

        torch.manual_seed(seed)
        x = torch.tensor(liked - self.space_mean, dtype=torch.float32)
        vae = _VAE(dim, hidden, latent_dim)
        optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
        pairs = sibling_pairs or []

        loss = torch.tensor(0.0)
        for _ in range(epochs):
            optimizer.zero_grad()
            recon, mu, logvar = vae(x)
            recon_loss = torch_fn.mse_loss(recon, x)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + beta * kl
            if pairs:
                left = torch.tensor([a for a, _ in pairs])
                right = torch.tensor([b for _, b in pairs])
                code, _ = vae.encode(x)
                loss = loss + (code[left] - code[right]).pow(2).mean()
            loss.backward()
            optimizer.step()

        self._vae = vae.eval()
        self.final_loss = float(loss.detach())
        return self

    @torch.no_grad()
    def score(self, candidates: np.ndarray) -> np.ndarray:
        """Return each candidate's ELBO — higher = more on the taste manifold."""
        if self._vae is None or self.space_mean is None:
            raise RuntimeError("ManifoldModel.score() called before fit()/load()")
        x = torch.tensor(
            np.asarray(candidates, dtype=np.float64) - self.space_mean,
            dtype=torch.float32,
        )
        mu, logvar = self._vae.encode(x)
        recon = self._vae.decoder(mu)  # deterministic — use the latent mean
        recon_err = (recon - x).pow(2).mean(dim=1)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean(dim=1)
        return (-(recon_err + kl)).numpy().astype(np.float64)

    @torch.no_grad()
    def sample(self, n: int, seed: int = 0) -> np.ndarray:
        """Decode ``n`` random latents — imagined tracks in the user's taste."""
        if self._vae is None or self.space_mean is None:
            raise RuntimeError("ManifoldModel.sample() called before fit()/load()")
        generator = torch.Generator().manual_seed(seed)
        z = torch.randn(n, self._dims[2], generator=generator)
        return self._vae.decoder(z).numpy().astype(np.float64) + self.space_mean

    def save(self, path: str | Path) -> None:
        """Persist the trained VAE to a ``.pt`` checkpoint."""
        if self._vae is None:
            raise RuntimeError("nothing to save — model is not fitted")
        torch.save(
            {
                "state": self._vae.state_dict(),
                "space_mean": self.space_mean,
                "dims": self._dims,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ManifoldModel":
        """Load a checkpoint written by :meth:`save`."""
        checkpoint = torch.load(path, weights_only=False)
        model = cls()
        model.space_mean = checkpoint["space_mean"]
        model._dims = tuple(checkpoint["dims"])  # type: ignore[assignment]
        vae = _VAE(*model._dims)
        vae.load_state_dict(checkpoint["state"])
        model._vae = vae.eval()
        return model
