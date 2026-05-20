"""Composite scoring — blends the MERT taste score with a CLAP fit score.

MERT (the taste model, M1/M2) is the primary signal; CLAP — a joint audio/text
space — adds a second, complementary view. Per candidate, the composite score
is a config-weighted blend of the **z-normalised** MERT and CLAP scores
(z-scoring puts the two onto a comparable scale before mixing). A candidate
with no CLAP embedding keeps only its MERT term.
"""

import numpy as np


def _zscore(values: np.ndarray) -> np.ndarray:
    std = float(values.std())
    return (values - values.mean()) / std if std > 0 else np.zeros_like(values)


def blend(
    mert_scores: np.ndarray,
    clap_scores: np.ndarray,
    has_clap: np.ndarray,
    mert_weight: float = 0.6,
    clap_weight: float = 0.4,
) -> np.ndarray:
    """Blend per-candidate MERT and CLAP scores (z-normalised, then weighted).

    ``clap_scores`` need only be meaningful where ``has_clap`` is True; those
    entries are z-normalised among themselves. Candidates without a CLAP
    embedding contribute only their MERT term.
    """
    z_mert = _zscore(np.asarray(mert_scores, dtype=np.float64))
    has = np.asarray(has_clap, dtype=bool)
    z_clap = np.zeros_like(z_mert)
    if has.any():
        present = np.asarray(clap_scores, dtype=np.float64)[has]
        std = float(present.std())
        if std > 0:
            z_clap[has] = (present - present.mean()) / std
    return mert_weight * z_mert + clap_weight * z_clap


def clap_fit_scores(
    cfg: object, liked_ids: list[str], candidate_ids: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Per-candidate cosine fit to the CLAP taste centroid.

    Returns ``(scores, has_clap)`` — ``scores[i]`` is meaningful only where
    ``has_clap[i]`` is True (the candidate has a CLAP embedding).
    """
    from storage import vectors
    from taste_model.centroid import CentroidModel

    store = vectors.read_embeddings(vectors.song_embedding_path(cfg, "clap_song"))  # type: ignore[arg-type]
    scores = np.zeros(len(candidate_ids))
    has = np.zeros(len(candidate_ids), dtype=bool)

    liked_with_clap = [t for t in liked_ids if t in store]
    if not store or not liked_with_clap:
        return scores, has

    space_mean = np.stack([store[t].embedding for t in store]).mean(axis=0)
    model = CentroidModel().fit(
        np.stack([store[t].embedding for t in liked_with_clap]), space_mean=space_mean
    )
    for i, track_id in enumerate(candidate_ids):
        if track_id in store:
            scores[i] = float(model.score(store[track_id].embedding[None, :])[0])
            has[i] = True
    return scores, has
