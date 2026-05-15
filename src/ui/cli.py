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
    """Pull the Spotify saved library + playlists into the local database."""
    from acquisition import resolver
    from acquisition.spotify import SpotifyClient
    from storage import db

    cfg = _cfg()
    client = SpotifyClient(cfg)  # type: ignore[arg-type]
    seen = created = 0
    with db.session_scope(cfg) as session:  # type: ignore[arg-type]
        for ref in client.iter_library():
            seen += 1
            _, is_new = resolver.upsert_track(session, ref)
            created += int(is_new)
    log.info("sync.done", seen=seen, created=created)
    typer.secho(
        f"Synced {seen} tracks — {created} new, {seen - created} already known.",
        fg=typer.colors.GREEN,
    )


@app.command()
def add(query: Annotated[str, typer.Argument(help='"<artist> - <title>"')]) -> None:
    """Add a track manually and queue it for download."""
    from acquisition import manual, resolver
    from storage import db

    cfg = _cfg()
    ref = manual.parse_manual_entry(query)
    with db.session_scope(cfg) as session:  # type: ignore[arg-type]
        _, is_new = resolver.upsert_track(session, ref)
    verb = "Queued" if is_new else "Already known"
    typer.secho(f"{verb}: {ref.artist} — {ref.title}", fg=typer.colors.GREEN)


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
def extract() -> None:
    """[Phase 1] Run the feature-extraction pipeline."""
    _phase_stub("extract", "Phase 1")


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
