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
            stmt = stmt.join(TrackSource).where(
                TrackSource.source_type == SourceType(source)
            )
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
def analyze() -> None:
    """[Phase 4] Deep analysis: Demucs stems + Whisper lyrics."""
    _phase_stub("analyze", "Phase 4")


@app.command()
def crawl(
    target: Annotated[int, typer.Option(help="Target number of candidate tracks.")] = 500,
    depth: Annotated[int, typer.Option(help="Artist-graph BFS depth.")] = 2,
    seeds: Annotated[int, typer.Option(help="Number of seed artists from the library.")] = 40,
) -> None:
    """Crawl the Last.fm artist graph for candidate tracks (Phase 2).

    Seeds from your most-common library artists, walks the similar-artist
    graph, and queues newly discovered tracks. Run `download` then `extract`
    on them, then `recommend` ranks them against your taste centroid.
    """
    from collections import Counter

    from sqlmodel import select

    from acquisition import resolver
    from acquisition.base import Provenance
    from acquisition.lastfm import LastfmClient
    from recommend.crawler.artist_graph import crawl_artist_graph
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

        existing = {(t.artist.lower(), t.title.lower()) for t in library}
        candidates = crawl_artist_graph(
            LastfmClient(), seed_artists, depth=depth, known_artists=known, target=target
        )
        queued = 0
        for ref in candidates:
            if (ref.artist.lower(), ref.title.lower()) in existing:
                continue
            resolver.upsert_track(session, ref, Provenance("crawl"))
            queued += 1

    typer.secho(
        f"Crawled {len(candidates)} candidates from {len(seed_artists)} seed artists "
        f"— {queued} new tracks queued for download.",
        fg=typer.colors.GREEN,
    )


@app.command()
def train() -> None:
    """[Phase 2+] Train a taste model (centroid / contrastive / manifold)."""
    _phase_stub("train", "Phase 2")


@app.command()
def recommend(
    top: Annotated[int, typer.Option(help="Number of recommendations to show.")] = 20,
) -> None:
    """Rank tracks by fit to your taste centroid (M1 baseline).

    The candidate pool is currently the extracted library itself — so this
    surfaces your most on-taste tracks. External candidates arrive with the
    crawler.
    """
    from recommend.rank import recommend as rank_tracks
    from storage import db

    cfg = _cfg()
    with db.session_scope(cfg) as session:  # type: ignore[arg-type]
        recs = rank_tracks(cfg, session, top_k=top)

    for rec in recs:
        typer.echo(f"  {rec.rank:>2}. {rec.score:+.3f}  {rec.artist} — {rec.title}")
    typer.secho(f"Top {len(recs)} by M1 centroid fit.", fg=typer.colors.GREEN)


@app.command()
def rate() -> None:
    """[Phase 3] Interactive rating session."""
    _phase_stub("rate", "Phase 3")


@app.command(name="eval")
def run_eval() -> None:
    """[Phase 3] Hold-out evaluation: recall@k, MAP."""
    _phase_stub("eval", "Phase 3")


@app.command(name="ui")
def launch_ui() -> None:
    """[Phase 2] Launch the Streamlit research dashboard."""
    _phase_stub("ui", "Phase 2")


def main() -> None:
    """Console-script entry point (see ``[project.scripts]`` in pyproject.toml)."""
    app()


if __name__ == "__main__":
    main()
