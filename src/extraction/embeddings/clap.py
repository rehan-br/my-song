"""CLAP audio embeddings (``laion/clap-htsat-fused``).

CLAP shares a joint audio/text embedding space, so these vectors can later be
queried with natural-language prompts. Phase 1 uses the audio side: the song is
chunked into ~10s windows, each window embedded, and the per-window vectors
mean-pooled. CLAP expects 48kHz audio (invariant 8).
"""

from collections.abc import Iterator

import numpy as np
import torch
from transformers import ClapModel, ClapProcessor

from core.logging import get_logger
from extraction.audio import chunk

log = get_logger("clap")


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _batched(items: list[np.ndarray], size: int) -> Iterator[list[np.ndarray]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


class ClapEmbedder:
    """Loads CLAP once, then embeds full songs into a single pooled vector."""

    def __init__(
        self,
        repo: str,
        *,
        sample_rate: int = 48000,
        device: str = "auto",
        fp16: bool = True,
        chunk_seconds: int = 10,
        batch_size: int = 16,
        seed: int = 42,
    ) -> None:
        self.repo = repo
        self.sample_rate = sample_rate
        self.device = _resolve_device(device)
        self.fp16 = fp16 and self.device == "cuda"
        self.batch_size = batch_size
        self.chunk_samples = chunk_seconds * sample_rate

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        log.info("clap.loading", repo=repo, device=self.device, fp16=self.fp16)
        self.processor = ClapProcessor.from_pretrained(repo)
        self.model = ClapModel.from_pretrained(repo).to(self.device).eval()

    @torch.no_grad()
    def embed_song(self, waveform: np.ndarray) -> np.ndarray:
        """Return one pooled CLAP audio embedding (float32) for a mono waveform."""
        windows = chunk(waveform, self.chunk_samples)
        pooled: list[torch.Tensor] = []
        for batch in _batched(windows, self.batch_size):
            inputs = self.processor(
                audios=batch, sampling_rate=self.sample_rate, return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16):
                feats = self.model.get_audio_features(**inputs)  # [B, D]
            pooled.append(feats.float().cpu())
        song = torch.cat(pooled, dim=0).mean(dim=0)  # mean over windows -> [D]
        return song.numpy().astype(np.float32)
