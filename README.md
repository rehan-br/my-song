# Music Taste Engine

A **personal** music recommendation system that tries to capture the *felt essence*
of a listener's taste — not surface metadata.

> **Personal-use posture (invariant 5).** This is a personal research tool. No audio
> is redistributed. Audio is cached locally in `data/` (gitignored) purely for
> feature extraction. yt-dlp is used for personal/research feature analysis only.
> Do not commit or share `data/`. Any productization branch must replace the audio
> acquisition layer with a licensed source.

## Status

**Phase 0 — Foundations.** Repo scaffold, SQLite schema, Spotify auth, yt-dlp
acquisition, and a smoke-test harness. ML/extraction phases are scaffolded but not
yet implemented.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12 (uv will fetch it).

```bash
uv sync                       # core deps (Phase 0)
uv run pre-commit install     # lint/format on commit
cp .env.example .env          # then fill in Spotify credentials
```

Optional heavier stacks, installed per phase:

```bash
uv sync --extra extraction    # Phase 1: MERT, CLAP, librosa, FAISS
uv sync --extra deep          # Phase 4: Demucs, Whisper
uv sync --extra ui            # Streamlit dashboard
```

**Essentia** is not a pip dependency — it is fiddly to install. Use the
conda-forge build (`conda install -c conda-forge essentia`) or compile from
source, and record the exact path that worked here once Phase 1 begins.

**ffmpeg** must be on `PATH` for `music download` (yt-dlp post-processing).

## Spotify credentials

1. Create an app at <https://developer.spotify.com/dashboard>.
2. Add redirect URI `http://127.0.0.1:8080/callback` (must match
   `config/default.yaml`).
3. Put the Client ID / Secret in `.env` as `SPOTIFY_CLIENT_ID` /
   `SPOTIFY_CLIENT_SECRET`.

## CLI

```bash
uv run music auth                       # Spotify OAuth flow
uv run music sync                       # pull saved library + playlists
uv run music add "Radiohead - Weird Fishes"
uv run music download                   # resolve + download queued tracks
```

Every command accepts `--config <path>` for YAML overrides.

## Phase 0 smoke test

After `music auth` succeeds:

```bash
uv run music sync                       # tracks land as status=queued
uv run music download --limit 10        # audio cached under data/audio/
```

## Layout

See `CLAUDE.md` for the authoritative module map. Source lives under `src/` with
each subdirectory a top-level package (`acquisition`, `storage`, `core`, ...).
