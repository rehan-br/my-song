"""MERT audio embeddings (``m-a-p/MERT-v1-330M``).

Full-song embedding: the song is chunked into ~5s windows (MERT's training
clip length), each window is embedded, and the per-window vectors are
mean-pooled. A window's vector is the mean over time of the mean over all 25
hidden-state layers — a stable, layer-agnostic baseline. Per-section embeddings
and learned layer weighting come in later phases.
"""

from collections.abc import Iterator

import numpy as np
import torch
from transformers import AutoModel, Wav2Vec2FeatureExtractor

from core.logging import get_logger
from extraction.audio import chunk

log = get_logger("mert")


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _batched(items: list[np.ndarray], size: int) -> Iterator[list[np.ndarray]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


class MertEmbedder:
    """Loads MERT once, then embeds full songs into a single pooled vector."""

    def __init__(
        self,
        repo: str,
        *,
        sample_rate: int = 24000,
        device: str = "auto",
        fp16: bool = True,
        chunk_seconds: int = 5,
        batch_size: int = 8,
        seed: int = 42,
    ) -> None:
        self.repo = repo
        self.sample_rate = sample_rate
        self.device = _resolve_device(device)
        self.fp16 = fp16 and self.device == "cuda"
        self.batch_size = batch_size
        self.chunk_samples = chunk_seconds * sample_rate

        # Determinism (invariant 7): inference runs in eval()/no_grad, so a
        # fixed seed + disabling cuDNN autotuning gives reproducible embeddings.
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        log.info("mert.loading", repo=repo, device=self.device, fp16=self.fp16)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(repo, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(repo, trust_remote_code=True).to(self.device).eval()

    @torch.no_grad()
    def embed_song(self, waveform: np.ndarray) -> np.ndarray:
        """Return one pooled MERT embedding (float32) for a mono waveform."""
        windows = chunk(waveform, self.chunk_samples)
        pooled: list[torch.Tensor] = []
        for batch in _batched(windows, self.batch_size):
            inputs = self.processor(batch, sampling_rate=self.sample_rate, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16):
                out = self.model(**inputs, output_hidden_states=True)
            # hidden_states: tuple(L) of [B, T, H]  ->  [L, B, T, H]
            hidden = torch.stack(out.hidden_states, dim=0)
            # mean over time, then over layers  ->  [B, H]
            pooled.append(hidden.mean(dim=2).mean(dim=0).float().cpu())
        song = torch.cat(pooled, dim=0).mean(dim=0)  # mean over windows -> [H]
        return song.numpy().astype(np.float32)
