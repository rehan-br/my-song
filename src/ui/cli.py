"""Typer CLI — entry point for ``uv run music ...``.

Phase 0 implements ``auth``, ``sync``, ``add`` and ``download``. Later phases
fill in the remaining commands; they currently stop with a clear message.
"""

from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv

from acquisition.base import TrackRef
from core import paths
from core.config import load_config
from core.logging import configure_logging, get_logger

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Music Taste Engine — personal music recommendation research tool.",
)
log = get_logger("cli")

# Populated by the callback before any command runs.
_state: dict[str, object] = {}


def _cfg() -> object:
    return _state["cfg"]


@app.callback()
def _bootstrap(
    config: Annotated[
        Path | None, typer.Option("--config", help="Path to a YAML override file.")
    ] = None,
) -> None:
    """Load .env, compose config, configure logging, ensure the DB exists."""
    load_dotenv(paths.PROJECT_ROOT / ".env")
    cfg = load_config(extra=config)
    configure_logging(
        level=str(cfg.logging.level),
        json_file=bool(cfg.logging.json_file),
        log_dir=paths.resolve(cfg.paths.logs),
    )
    from storage import db

    db.init_db(cfg)
    _state["cfg"] = cfg


@app.command()
def auth() -> None:
    """Run the Spotify OAuth flow and cache the access token."""
    from acquisition.spotify import SpotifyClient

    profile = SpotifyClient(_cfg()).authenticate()  # type: ignore[arg-type]
    name = profile.get("display_name") or profile.get("id")
    typer.secho(f"Authenticated with Spotify as {name}.", fg=typer.colors.GREEN)


@app.command()
def sync() -> None:
    """Pull the Spotify library, playlists and listening signals.

    Saved/playlist/top tracks are ingested with provenance, then each track's
    Spotify listening signals (top-track tier, recency) are recorded. These are
    observed inputs for the taste model — `sync` does not assign taste weights.
    """
    from sqlmodel import select

    from acquisition import resolver
    from acquisition.base import Provenance
    from acquisition.spotify import SpotifyClient
    from storage import db
    from storage.schema import Track
    from taste_model import engagement

    cfg = _cfg()
    client = SpotifyClient(cfg)  # type: ignore[arg-type]
    created = 0

    with db.session_scope(cfg) as session:  # type: ignore[arg-type]
        # Pass 1 — ingest saved + playlist tracks with provenance.
        for ref, source in client.iter_library():
            _, is_new = resolver.upsert_track(session, ref, source)
            created += int(is_new)

        # Pass 1b — ingest Spotify top tracks; build the affinity map.
        top_map: dict[str, tuple[str, int]] = {}
        for ref, term, rank in client.iter_top_tracks():
            _, is_new = resolver.upsert_track(session, ref, Provenance("top"))
            created += int(is_new)
            if ref.spotify_id:
                top_map[ref.spotify_id] = (term, rank)  # short->long; long wins

        recent_map = client.recently_played()
        session.flush()

        # Pass 2 — record listening signals (observed data, not a weighting).
        signalled = engagement.refresh_listening_signals(session, top_map, recent_map)
        total = len(session.exec(select(Track)).all())

    log.info("sync.done", total=total, created=created, signalled=signalled)
    typer.secho(
        f"Synced — {total} tracks ({created} new). Recorded listening signals for {signalled}.",
        fg=typer.colors.GREEN,
    )


@app.command(name="sync-history")
def sync_history() -> None:
    """Sync silent listening signals from Spotify recently-played.

    Pulls the recent-plays window, infers skip/complete from inter-play gaps,
    and appends new rows to ``listening_events``. Run regularly — the window is
    shallow (~50 plays), so behavioural history accumulates across calls.
    """
    from acquisition.events import infer_events, ingest_events
    from acquisition.spotify import SpotifyClient
    from storage import db
    from storage.schema import EventType
    from taste_model import engagement

    cfg = _cfg()
    client = SpotifyClient(cfg)  # type: ignore[arg-type]
    plays = client.iter_recent_plays()
    events = infer_events(plays)
    with db.session_scope(cfg) as session:  # type: ignore[arg-type]
        written = ingest_events(session, events, source="spotify")
        reweighted = engagement.refresh_engagement_weights(session, cfg)  # type: ignore[arg-type]

    skips = sum(e.event_type == EventType.skip for e in events)
    completes = sum(e.event_type == EventType.complete for e in events)
    log.info(
        "sync_history.done",
        plays=len(plays),
        new=written,
        skips=skips,
        completes=completes,
        reweighted=reweighted,
    )
    typer.secho(
        f"Synced {len(plays)} recent plays — {written} new events "
        f"({completes} completed, {skips} skipped). "
        f"Engagement weight updated on {reweighted} track(s).",
        fg=typer.colors.GREEN,
    )


@app.command()
def add(query: Annotated[str, typer.Argument(help='"<artist> - <title>"')]) -> None:
    """Add a track manually — an explicit taste signal.

    A manual add is the user saying "make this count", so the track is marked
    as a pinned taste signal (taste_weight_auto = false): once the taste model
    assigns weights, it will leave this track alone.
    """
    from acquisition import manual, resolver
    from acquisition.base import Provenance
    from storage import db

    cfg = _cfg()
    ref = manual.parse_manual_entry(query)
    with db.session_scope(cfg) as session:  # type: ignore[arg-type]
        track, is_new = resolver.upsert_track(session, ref, Provenance("manual"))
        track.taste_weight = 1.0  # explicit full influence
        track.taste_weight_auto = False  # pinned — the taste model won't lower it
        session.add(track)
    verb = "Added" if is_new else "Updated"
    typer.secho(
        f"{verb} (pinned as an explicit taste signal): {ref.artist} — {ref.title}",
        fg=typer.colors.GREEN,
    )


@app.command()
def download(
    limit: Annotated[int, typer.Option(help="Max queued tracks to process.")] = 100,
    workers: Annotated[int, typer.Option(help="Parallel workers (0 = config default).")] = 0,
    source: Annotated[
        str | None, typer.Option(help="Only tracks of this provenance, e.g. 'crawl'.")
    ] = None,
) -> None:
    """Resolve and download audio for queued tracks — in parallel via yt-dlp."""
    from sqlmodel import select

    from acquisition import downloader
    from acquisition.youtube import YouTubeSource
    from storage import db
    from storage.schema import SourceType, Track, TrackSource, TrackStatus

    cfg = _cfg()
    audio_source = YouTubeSource(cfg)  # type: ignore[arg-type]
    audio_dir = paths.resolve(cfg.paths.audio)  # type: ignore[attr-defined]
    tolerance = float(cfg.acquisition.duration_tolerance)  # type: ignore[attr-defined]
    n_workers = workers or int(cfg.acquisition.download_workers)  # type: ignore[attr-defined]

    with db.session_scope(cfg) as session:  # type: ignore[arg-type]
        stmt = select(Track).where(Track.status == TrackStatus.queued)
        if source:
            stmt = stmt.join(TrackSource).where(TrackSource.source_type == SourceType(source))
        queued = session.exec(stmt.limit(limit)).all()
        if not queued:
            typer.echo("Nothing queued to download.")
            return

        by_id = {t.id: t for t in queued}
        jobs = [
            (t.id, TrackRef(title=t.title, artist=t.artist, duration_ms=t.duration_ms))
            for t in queued
        ]
        for track in queued:
            track.status = TrackStatus.downloading
            session.add(track)
        session.commit()

        typer.echo(f"Downloading {len(jobs)} track(s) with {n_workers} workers…")
        ok = failed = 0
        for result in downloader.download_tracks(
            audio_source, jobs, audio_dir, tolerance, n_workers
        ):
            track = by_id[result.track_id]
            if result.ok:
                track.youtube_id = result.youtube_id
                track.audio_path = result.audio_path
                track.status = TrackStatus.downloaded
            else:
                track.status = TrackStatus.failed
                log.warning("download.failed", track_id=track.id, error=result.error)
            try:
                session.add(track)
                session.commit()
            except Exception as exc:
                # e.g. two tracks resolved to the same youtube_id — drop this one
                session.rollback()
                track.youtube_id = None
                track.audio_path = None
                track.status = TrackStatus.failed
                session.add(track)
                session.commit()
                log.warning("download.persist_failed", track_id=track.id, error=str(exc))
            ok += int(track.status == TrackStatus.downloaded)
            failed += int(track.status == TrackStatus.failed)

    color = typer.colors.GREEN if failed == 0 else typer.colors.YELLOW
    typer.secho(f"Downloaded {ok}, failed {failed}.", fg=color)


def _phase_stub(name: str, phase: str) -> None:
    typer.secho(f"`{name}` is not implemented yet — planned for {phase}.", fg=typer.colors.YELLOW)
    raise typer.Exit(code=1)


@app.command()
def extract(
    track: Annotated[
        str | None, typer.Option(help="Extract one track by id (default: all downloaded).")
    ] = None,
    limit: Annotated[int, typer.Option(help="Max tracks to process (0 = no limit).")] = 0,
    force: Annotated[
        bool, typer.Option(help="Re-extract even if already done / config changed.")
    ] = False,
    fast: Annotated[
        bool,
        typer.Option(help="MERT only — skips CLAP + Librosa. Enough for `recommend`."),
    ] = False,
) -> None:
    """Run the feature-extraction pipeline (MERT + CLAP + Librosa, or MERT-only)."""
    from sqlmodel import select

    from extraction import pipeline
    from storage import db
    from storage.schema import Track, TrackStatus

    cfg = _cfg()
    with db.session_scope(cfg) as session:  # type: ignore[arg-type]
        if track:
            stmt = select(Track).where(Track.id == track)
        else:
            stmt = select(Track).where(Track.audio_path.is_not(None))
            if not force:
                stmt = stmt.where(Track.status == TrackStatus.downloaded)
            if limit:
                stmt = stmt.limit(limit)
        tracks = list(session.exec(stmt).all())
        if not tracks:
            typer.echo("No tracks to extract (need status=downloaded with audio).")
            return
        models = "MERT" if fast else "MERT + CLAP"
        typer.echo(
            f"Extracting {len(tracks)} track(s)"
            f"{' (fast: MERT only)' if fast else ''} — first run downloads {models}."
        )
        result = pipeline.run_extraction(cfg, session, tracks, force=force, fast=fast)

    typer.secho(
        f"Extracted {result['ok']} track(s), {result['failed']} failed.",
        fg=typer.colors.GREEN if result["failed"] == 0 else typer.colors.YELLOW,
    )


@app.command()
def analyze(
    track_id: Annotated[str, typer.Argument(help="Track id to deep-analyze.")],
    deep: Annotated[bool, typer.Option(help="Confirm the heavy Demucs + Whisper run.")] = False,
) -> None:
    """Deep analysis of one track — Demucs stems + Whisper lyrics (on-demand)."""
    if not deep:
        typer.echo("Deep analysis is heavy (Demucs + Whisper) — pass --deep to confirm.")
        raise typer.Exit(code=1)

    from extraction.analyze import analyze_track
    from storage import db
    from storage.schema import Track

    cfg = _cfg()
    with db.session_scope(cfg) as session:  # type: ignore[arg-type]
        track = session.get(Track, track_id)
        if track is None:
            typer.secho(f"No track with id {track_id}.", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        result = analyze_track(cfg, session, track)
    typer.secho(
        f"Deep-analyzed — {result['stems']} stems separated, "
        f"{result['lyric_chars']} lyric characters transcribed.",
        fg=typer.colors.GREEN,
    )


@app.command()
def crawl(
    target: Annotated[int, typer.Option(help="Target number of candidate tracks.")] = 500,
    depth: Annotated[int, typer.Option(help="Artist-graph BFS depth.")] = 2,
    seeds: Annotated[int, typer.Option(help="Number of seed artists from the library.")] = 40,
) -> None:
    """Crawl Last.fm for candidate tracks — 80% artist-graph, 20% tag-graph.

    Seeds from your most-common library artists: an artist-similarity BFS finds
    music *near* your taste, a tag walk adds a serendipity stream, and the two
    are mixed ~80/20. Run `download` then `extract` on the queued tracks; then
    `recommend` ranks them.
    """
    from collections import Counter

    from sqlmodel import select

    from acquisition import resolver
    from acquisition.base import Provenance
    from acquisition.lastfm import LastfmClient
    from recommend.crawler.artist_graph import crawl_artist_graph
    from recommend.crawler.sampler import mix
    from recommend.crawler.tag_graph import crawl_tag_graph
    from storage import db
    from storage.schema import Track

    cfg = _cfg()
    with db.session_scope(cfg) as session:  # type: ignore[arg-type]
        library = session.exec(select(Track)).all()
        primary = [t.artist.split(",")[0].strip() for t in library if t.artist]
        known = {a.lower() for a in primary}
        seed_artists = [artist for artist, _ in Counter(primary).most_common(seeds)]
        if not seed_artists:
            typer.echo("No library artists to seed from — run `music sync` first.")
            return

        client = LastfmClient()
        artist_pool = crawl_artist_graph(
            client, seed_artists, depth=depth, known_artists=known, target=target
        )
        reachable = {ref.artist.lower() for ref in artist_pool}
        tag_pool = crawl_tag_graph(
            client,
            seed_artists,
            known_artists=known,
            reachable_artists=reachable,
            target=round(target * 0.3),
        )
        candidates = mix(artist_pool, tag_pool, target, artist_frac=0.8)

        existing = {(t.artist.lower(), t.title.lower()) for t in library}
        queued = 0
        for ref in candidates:
            if (ref.artist.lower(), ref.title.lower()) in existing:
                continue
            resolver.upsert_track(session, ref, Provenance("crawl"))
            queued += 1

    typer.secho(
        f"Crawled {len(candidates)} candidates "
        f"({len(artist_pool)} artist-graph, {len(tag_pool)} tag-graph) "
        f"— {queued} new tracks queued for download.",
        fg=typer.colors.GREEN,
    )


@app.command()
def train(
    model: Annotated[
        str, typer.Option(help="centroid (M1) | contrastive (M2) | manifold (M3).")
    ] = "contrastive",
) -> None:
    """Train a taste model. M2 (contrastive) learns per-dimension taste weights."""
    cfg = _cfg()
    if model in ("centroid", "m1"):
        typer.echo("M1 (centroid) needs no training — use `recommend` / `eval`.")
        return
    if model in ("manifold", "m3"):
        from storage import db
        from taste_model.trainer import train_manifold

        with db.session_scope(cfg) as session:  # type: ignore[arg-type]
            metrics = train_manifold(cfg, session)
        typer.secho(
            f"Trained M3 — VAE on {int(metrics['n_liked'])} liked tracks "
            f"({int(metrics['n_sibling_pairs'])} sibling pairs) · "
            f"final loss {metrics['final_loss']:.4f}.",
            fg=typer.colors.GREEN,
        )
        return
    if model not in ("contrastive", "m2"):
        typer.secho(f"Unknown model: {model}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    from storage import db
    from taste_model.trainer import train_contrastive

    with db.session_scope(cfg) as session:  # type: ignore[arg-type]
        metrics = train_contrastive(cfg, session)
    typer.secho(
        f"Trained M2 — {int(metrics['n_positives'])} positives vs "
        f"{int(metrics['n_negatives'])} negatives · final loss "
        f"{metrics['final_loss']:.4f} · weight spread {metrics['scale_std']:.3f}.",
        fg=typer.colors.GREEN,
    )


@app.command()
def recommend(
    top: Annotated[int, typer.Option(help="Number of recommendations to show.")] = 20,
    model: Annotated[str, typer.Option(help="auto | centroid (M1) | contrastive (M2).")] = "auto",
    composite: Annotated[
        bool, typer.Option(help="Blend the MERT score with a CLAP fit score.")
    ] = False,
) -> None:
    """Rank tracks by taste-model fit. `auto` uses M2 if trained, else M1."""
    from recommend.rank import recommend as rank_tracks
    from storage import db

    cfg = _cfg()
    with db.session_scope(cfg) as session:  # type: ignore[arg-type]
        recs = rank_tracks(cfg, session, top_k=top, model=model, composite=composite)

    for rec in recs:
        typer.echo(f"  {rec.rank:>2}. {rec.score:+.3f}  {rec.artist} — {rec.title}")
    typer.secho(f"Top {len(recs)} recommendations.", fg=typer.colors.GREEN)


def _prompt_score(label: str) -> int:
    """Prompt for a 1-5 rubric score, re-asking until valid."""
    while True:
        value = typer.prompt(f"  {label} [1-5]", type=int)
        if 1 <= value <= 5:
            return value
        typer.echo("  please enter a number from 1 to 5")


@app.command()
def rate(
    count: Annotated[int, typer.Option(help="Number of tracks to rate.")] = 15,
) -> None:
    """Blind listening session — rate tracks on the vibe / replay / skip rubric.

    Each track's audio opens in your default player; rate it without seeing the
    artist or title, so the rating reflects the sound, not the name.
    """
    import os

    from eval import listening
    from storage import db

    cfg = _cfg()
    audio_dir = paths.resolve(cfg.paths.audio)  # type: ignore[attr-defined]
    rated = 0
    with db.session_scope(cfg) as session:  # type: ignore[arg-type]
        tracks = listening.pick_session_tracks(session, count)
        if not tracks:
            typer.echo("No unrated extracted tracks — run `music extract` first.")
            return
        typer.echo(
            f"Blind listening session — {len(tracks)} track(s). Rate 1-5; Ctrl-C to stop early."
        )
        for position, track in enumerate(tracks, start=1):
            typer.secho(f"\n[{position}/{len(tracks)}] ▶ playing…", fg=typer.colors.CYAN)
            audio = audio_dir / str(track.audio_path)
            if hasattr(os, "startfile") and audio.exists():
                try:
                    os.startfile(audio)  # type: ignore[attr-defined]
                except OSError as exc:
                    log.warning("rate.play_failed", track_id=track.id, error=str(exc))
            vibe = _prompt_score("vibe   (does it feel right?)")
            replay = _prompt_score("replay (would you play it again?)")
            skip = _prompt_score("skip   (urge to skip it?)")
            notes = typer.prompt("  notes", default="", show_default=False).strip()
            listening.record_rating(session, track.id, vibe, replay, skip, notes or None)
            session.commit()
            rated += 1
    typer.secho(f"Recorded {rated} rating(s).", fg=typer.colors.GREEN)


@app.command(name="eval")
def run_eval(
    holdout: Annotated[float, typer.Option(help="Fraction of liked tracks held out.")] = 0.2,
    k: Annotated[int, typer.Option(help="Cutoff for recall@k.")] = 20,
    splits: Annotated[int, typer.Option(help="Random hold-out splits to average.")] = 5,
    model: Annotated[str, typer.Option(help="auto | centroid | contrastive | manifold.")] = "auto",
) -> None:
    """Hold-out evaluation of a taste model — recall@k and MAP."""
    from core.config import config_hash
    from eval.holdout import evaluate_holdout
    from recommend.rank import split_pool
    from storage import db, vectors
    from storage.schema import TasteModelRun
    from taste_model.trainer import checkpoint_path

    cfg = _cfg()
    store = vectors.read_embeddings(vectors.song_embedding_path(cfg, "mert_song"))
    if not store:
        typer.echo("No MERT embeddings — run `music extract` first.")
        return

    fit_fn = None
    if model in ("manifold", "m3"):
        from taste_model.manifold import ManifoldModel

        m3 = cfg.taste.m3  # type: ignore[attr-defined]
        label = "m3-manifold"

        def _fit_m3(positives, negatives, space_mean):  # type: ignore[no-untyped-def]  # noqa: ARG001
            return ManifoldModel().fit(
                positives,
                space_mean,
                latent_dim=int(m3.latent_dim),
                hidden=int(m3.hidden),
                epochs=int(m3.epochs),
                lr=float(m3.lr),
                beta=float(m3.beta),
            )

        fit_fn = _fit_m3
    elif model in ("contrastive", "m2") or (model == "auto" and checkpoint_path(cfg).exists()):
        from taste_model.contrastive import ContrastiveModel

        m2 = cfg.taste.m2  # type: ignore[attr-defined]
        label = "m2-contrastive"

        def _fit_m2(positives, negatives, space_mean):  # type: ignore[no-untyped-def]
            return ContrastiveModel().fit(
                positives,
                negatives,
                space_mean,
                epochs=int(m2.epochs),
                lr=float(m2.lr),
                tau=float(m2.temperature),
            )

        fit_fn = _fit_m2
    else:
        label = "m1-centroid"

    with db.session_scope(cfg) as session:  # type: ignore[arg-type]
        liked_ids, candidate_ids = split_pool(session, sorted(store))
        liked = {t: store[t].embedding for t in liked_ids}
        candidates = {t: store[t].embedding for t in candidate_ids}
        metrics = evaluate_holdout(
            liked, candidates, fit_fn=fit_fn, holdout_frac=holdout, k=k, n_splits=splits
        )
        session.add(
            TasteModelRun(
                version=f"{label}-eval",
                config_hash=config_hash(cfg),
                metrics_json=metrics,
            )
        )

    typer.secho(
        f"{label} hold-out eval — {splits} splits, holdout {holdout:.0%}",
        fg=typer.colors.GREEN,
    )
    typer.echo(f"  recall@{k} : {metrics['recall_at_k']:.3f}")
    typer.echo(f"  MAP        : {metrics['map']:.3f}")
    typer.echo(
        f"  pool        : {int(metrics['n_liked'])} liked, "
        f"{int(metrics['n_candidates'])} candidates"
    )


@app.command(name="ui")
def launch_ui() -> None:
    """Launch the Streamlit research dashboard."""
    import subprocess
    import sys

    dashboard = paths.PROJECT_ROOT / "src" / "ui" / "dashboard.py"
    typer.secho(f"Launching dashboard — {dashboard}", fg=typer.colors.GREEN)
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard)], check=False)


def main() -> None:
    """Console-script entry point (see ``[project.scripts]`` in pyproject.toml)."""
    app()


if __name__ == "__main__":
    main()
