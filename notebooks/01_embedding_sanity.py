# %% [markdown]
# # 01 — Embedding sanity check
#
# Phase 1 validation gate. Do the learned audio embeddings (MERT, CLAP) place
# tracks in a space where *similar music sits together*? We UMAP-project the
# song embeddings to 2-D and colour by genre.
#
# Three panels: **MERT (raw)**, **MERT (mean-centred)** — testing whether
# removing the anisotropic common component helps — and **CLAP**.
#
# Genre labels come from **Last.fm artist tags** (Spotify's catalogue endpoints
# are 403 for a Development-mode app since the 2026 platform-security changes).

# %%
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from dotenv import load_dotenv
from sqlmodel import select

from core import paths
from core.config import load_config
from storage import db, vectors
from storage.schema import Track

load_dotenv(paths.PROJECT_ROOT / ".env")
cfg = load_config()

# %% [markdown]
# ## Load embeddings + track metadata

# %%
mert = vectors.read_embeddings(vectors.song_embedding_path(cfg, "mert_song"))
clap = vectors.read_embeddings(vectors.song_embedding_path(cfg, "clap_song"))
track_ids = sorted(set(mert) & set(clap))
print(f"{len(track_ids)} tracks with both MERT + CLAP embeddings")

with db.session_scope(cfg) as session:
    # Pull plain values inside the session — ORM objects detach once it closes.
    meta = {
        t.id: {"artist": t.artist, "title": t.title}
        for t in session.exec(select(Track).where(Track.id.in_(track_ids))).all()
    }

df = pd.DataFrame([{"track_id": t, **meta[t]} for t in track_ids])
mert_mat = np.stack([mert[t].embedding for t in track_ids])
clap_mat = np.stack([clap[t].embedding for t in track_ids])
print(f"MERT matrix {mert_mat.shape}, CLAP matrix {clap_mat.shape}")

# %% [markdown]
# ## Cosine-similarity spread
#
# A wide spread means the space discriminates tracks; a narrow one (anisotropy)
# means raw cosine barely separates anything.

# %%
def cosine_spread(matrix: np.ndarray) -> str:
    unit = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    sim = (unit @ unit.T)[np.triu_indices(len(matrix), 1)]
    return f"min={sim.min():.3f}  mean={sim.mean():.3f}  max={sim.max():.3f}"


mert_centred = mert_mat - mert_mat.mean(axis=0, keepdims=True)
print("MERT raw     ", cosine_spread(mert_mat))
print("MERT centred ", cosine_spread(mert_centred))
print("CLAP         ", cosine_spread(clap_mat))

# %% [markdown]
# ## Genre labels — Last.fm artist tags (cached to disk)

# %%
df["primary_artist"] = df["artist"].str.split(",").str[0].str.strip()

tag_cache = paths.resolve(cfg.paths.features) / "lastfm_tags.json"
artist_tags: dict[str, list[str]] = {}
if tag_cache.exists():
    artist_tags = json.loads(tag_cache.read_text())

missing = [a for a in sorted(df["primary_artist"].unique()) if a not in artist_tags]
if missing:
    from acquisition.lastfm import LastfmClient

    client = LastfmClient()
    for artist in missing:
        artist_tags[artist] = client.artist_tags(artist, limit=5)
    tag_cache.parent.mkdir(parents=True, exist_ok=True)
    tag_cache.write_text(json.dumps(artist_tags, indent=0))

df["genre"] = df["primary_artist"].map(
    lambda a: (artist_tags.get(a) or ["unknown"])[0]
)
print(f"genre tag present for {(df['genre'] != 'unknown').sum()}/{len(df)} tracks")

# Colour by the 8 most common genres; everything else collapses to "other".
top = df["genre"].value_counts().head(8).index
df["label"] = df["genre"].where(df["genre"].isin(top), "other")
print(df["label"].value_counts())

# %% [markdown]
# ## UMAP projections

# %%
def project(matrix: np.ndarray, seed: int = 42) -> np.ndarray:
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(15, len(matrix) - 1),
        min_dist=0.1,
        metric="cosine",
        random_state=seed,
    )
    return reducer.fit_transform(matrix)


projections = [
    ("MERT (raw)", project(mert_mat)),
    ("MERT (mean-centred)", project(mert_centred)),
    ("CLAP", project(clap_mat)),
]

# %% [markdown]
# ## Plot — UMAP coloured by genre

# %%
categories = sorted(df["label"].unique())
cmap = plt.get_cmap("tab10")
colours = {c: cmap(i % 10) for i, c in enumerate(categories)}

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, (name, xy) in zip(axes, projections, strict=True):
    for category in categories:
        mask = (df["label"] == category).to_numpy()
        ax.scatter(
            xy[mask, 0], xy[mask, 1], s=45, color=colours[category],
            label=category, alpha=0.85, edgecolor="white", linewidth=0.4,
        )
    ax.set_title(name)
    ax.set_xticks([])
    ax.set_yticks([])
axes[-1].legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=8)
fig.suptitle(f"Embedding sanity check — {len(df)} tracks · UMAP(cosine)", fontsize=13)
fig.tight_layout()
fig.savefig(
    paths.PROJECT_ROOT / "notebooks" / "01_embedding_sanity.png",
    dpi=110,
    bbox_inches="tight",
)
fig  # last expression — embeds the plot when run as a notebook cell

# %% [markdown]
# ## Nearest-neighbour probe
#
# Genre-colour clustering is a weak test on a genre-homogeneous library. A more
# direct "essence" check — and exactly what M1 will do — is to look at each
# track's nearest neighbours by embedding cosine. Do they *make sense*?

# %%
labels = (df["artist"] + " — " + df["title"]).tolist()


def neighbours(matrix: np.ndarray, idx: int, k: int = 5) -> list[tuple[int, float]]:
    unit = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    sims = unit @ unit[idx]
    order = np.argsort(-sims)
    return [(int(j), float(sims[j])) for j in order if j != idx][:k]


rng = np.random.default_rng(0)
for idx in rng.choice(len(df), size=5, replace=False):
    print(f"\nQUERY  {labels[idx]}")
    for space, matrix in (("centred-MERT", mert_centred), ("CLAP", clap_mat)):
        print(f"  {space}:")
        for j, sim in neighbours(matrix, int(idx)):
            print(f"    {sim:.3f}  {labels[j]}")

# %% [markdown]
# ## Observations
#
# - **Anisotropy is real and fixable.** Raw MERT cosine sits in a narrow
#   ~0.93–0.99 cone; mean-centring spreads it to roughly −0.74…0.75. M1 must
#   centre MERT embeddings before cosine ranking.
# - **Genre-colour clusters are faint** — but this library is genre-homogeneous
#   (rnb / hip-hop / soul / trap dominate; ~half the tracks are "other"/
#   "unknown"), so colour separation is not a fair test here.
# - The **nearest-neighbour probe** is the signal to read: if a track's top
#   cosine neighbours are plausibly "similar music", the embeddings capture
#   real structure — the Phase 1 hypothesis survives its first check.
# - Re-run on a deliberately genre-diverse track sample for a stronger
#   colour-cluster test.
