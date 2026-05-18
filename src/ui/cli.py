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
    limit: Annotated[int, typer.Option(help="Max queued tracks to process.")] = 20,
) -> None:
    """Resolve and download audio for queued tracks via yt-dlp."""
    from sqlmodel import select

    from acquisition import resolver
    from acquisition.youtube import YouTubeSource
    from storage import db
    from storage.schema import Track, TrackStatus

    cfg = _cfg()
    source = YouTubeSource(cfg)  # type: ignore[arg-type]
    audio_dir = paths.resolve(cfg.paths.audio)  # type: ignore[attr-defined]
    tolerance = float(cfg.acquisition.duration_tolerance)  # type: ignore[attr-defined]
    ok = failed = 0

    with db.session_scope(cfg) as session:  # type: ignore[arg-type]
        queued = session.exec(
            select(Track).where(Track.status == TrackStatus.queued).limit(limit)
        ).all()
        if not queued:
            typer.echo("Nothing queued to download.")
            return

        for track in queued:
            ref = TrackRef(title=track.title, artist=track.artist, duration_ms=track.duration_ms)
            try:
                track.status = TrackStatus.downloading
                session.add(track)
                session.commit()

                best = resolver.pick_best_candidate(ref, source.search(ref), tolerance)
                if best is None:
                    raise RuntimeError("no candidate within duration tolerance")

                path = source.fetch(best, audio_dir)
                track.youtube_id = best.source_id
                track.audio_path = str(path.relative_to(audio_dir))
                track.status = TrackStatus.downloaded
                ok += 1
                log.info("download.ok", track_id=track.id, youtube_id=best.source_id)
            except Exception as exc:
                track.status = TrackStatus.failed
                failed += 1
                log.warning("download.failed", track_id=track.id, error=str(exc))
            session.add(track)
            session.commit()

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
) -> None:
    """Run the feature-extraction pipeline (Phase 1: MERT full-song embeddings)."""
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
        typer.echo(
            f"Extracting {len(tracks)} track(s) — first run downloads MERT + CLAP models (~2GB)."
        )
        result = pipeline.run_extraction(cfg, session, tracks, force=force)

    typer.secho(
        f"Extracted {result['ok']} track(s), {result['failed']} failed.",
        fg=typer.colors.GREEN if result["failed"] == 0 else typer.colors.YELLOW,
    )


@app.command()
def analyze() -> None:
    """[Phase 4] Deep analysis: Demucs stems + Whisper lyrics."""
    _phase_stub("analyze", "Phase 4")


@app.command()
def crawl() -> None:
    """[Phase 2] Build the candidate pool via the artist/tag crawler."""
    _phase_stub("crawl", "Phase 2")


@app.command()
def train() -> None:
    """[Phase 2+] Train a taste model (centroid / contrastive / manifold)."""
    _phase_stub("train", "Phase 2")


@app.command()
def recommend() -> None:
    """[Phase 2] Produce top-K recommendations."""
    _phase_stub("recommend", "Phase 2")


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
