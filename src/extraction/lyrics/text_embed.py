"""Lyric text embeddings — ``intfloat/e5-large-v2`` via transformers.

e5 expects a task prefix; lyrics are embedded as a ``passage:``. The token
embeddings are attention-masked mean-pooled and L2-normalised — the standard
e5 recipe — giving a 1024-d vector comparable across tracks.
"""

from typing import Any

import numpy as np

from core.logging import get_logger

log = get_logger("text_embed")

_cache: dict[str, tuple[Any, Any]] = {}


def embed_text(text: str, repo: str = "intfloat/e5-large-v2") -> np.ndarray:
    """Embed a passage of text into a normalised e5 vector."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    if repo not in _cache:
        log.info("text_embed.loading", repo=repo)
        tokenizer = AutoTokenizer.from_pretrained(repo)
        model = AutoModel.from_pretrained(repo).eval()
        _cache[repo] = (tokenizer, model)
    tokenizer, model = _cache[repo]

    inputs = tokenizer(
        f"passage: {text}",
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )
    with torch.no_grad():
        hidden = model(**inputs).last_hidden_state
    mask = inputs["attention_mask"].unsqueeze(-1)
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    vector = torch.nn.functional.normalize(pooled, dim=1)[0]
    return vector.numpy().astype(np.float32)
