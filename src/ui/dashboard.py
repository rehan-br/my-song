"""Streamlit research dashboard.

Three pages:
- **Recommendations** — runs the taste model (M1 or M2) and shows the ranked list.
- **Audition** — plays a track via its YouTube embed; how long it holds your
  attention is captured silently as a listening event, with an optional 👍/👎.
- **Essence siblings** — tag two tracks that *feel* like siblings (M3 supervision).

Launch with ``uv run music ui``.
"""

import time

import streamlit as st
import streamlit.components.v1 as components
from sqlmodel import func, select

from core.config import load_config
from eval import listening
from recommend.rank import recommend
from storage import db
from storage.schema import (
    EssenceSibling,
    SourceType,
    Track,
    TrackSource,
    TrackStatus,
)

st.set_page_config(page_title="Music Taste Engine", page_icon="🎧", layout="wide")


@st.cache_resource
def _config() -> object:
    return load_config()


@st.cache_resource(show_spinner=False)
def _spotify_client() -> object | None:
    """A cached SpotifyClient, or None if credentials are unavailable."""
    from acquisition.spotify import SpotifyAuthError, SpotifyClient

    try:
        return SpotifyClient(_config())  # type: ignore[arg-type]
    except SpotifyAuthError:
        return None


@st.cache_data(show_spinner=False, ttl=1800)
def _spotify_premium_token() -> str | None:
    """A valid access token iff the account is Spotify Premium, else None.

    Premium gates the Web Playback SDK. A missing or stale-scoped token (e.g.
    before re-running ``music auth``) returns None — the page then degrades to
    the Spotify deep link.
    """
    client = _spotify_client()
    if client is None:
        return None
    try:
        return client.access_token() if client.is_premium() else None
    except Exception:  # noqa: BLE001 — best-effort; any failure → deep-link fallback
        return None


@st.cache_data(show_spinner=False)
def _resolve_track_uri(
    spotify_id: str | None, artist: str, title: str, duration_ms: int
) -> str | None:
    """The Spotify URI for a track — direct for library tracks, searched for
    crawled ones (which carry no Spotify id)."""
    if spotify_id:
        return f"spotify:track:{spotify_id}"
    client = _spotify_client()
    if client is None:
        return None
    try:
        return client.find_track_uri(artist, title, duration_ms)
    except Exception:  # noqa: BLE001 — best-effort resolution; None → deep-link
        return None


def _spotify_player_html(token: str, track_uri: str) -> str:
    """A self-contained Spotify Web Playback SDK player (one Play button)."""
    template = """
<div style="font-family:sans-serif;color:#ddd;">
  <button id="play" style="padding:8px 20px;font-size:15px;border-radius:20px;
    border:none;background:#1db954;color:#fff;cursor:pointer;">&#9654; Play full song</button>
  <span id="status" style="margin-left:12px;color:#999;">loading&hellip;</span>
</div>
<script>
window.onSpotifyWebPlaybackSDKReady = () => {
  const token = "__TOKEN__", uri = "__URI__";
  const status = document.getElementById('status');
  let deviceId = null;
  const player = new Spotify.Player({
    name: 'Music Taste Engine', getOAuthToken: cb => cb(token), volume: 0.6});
  player.addListener('ready', e => { deviceId = e.device_id; status.textContent = 'ready'; });
  player.addListener('not_ready', () => { status.textContent = 'device offline'; });
  player.addListener('account_error', () => { status.textContent = 'Spotify Premium required'; });
  player.addListener('authentication_error', () => { status.textContent = 're-run music auth'; });
  player.addListener('initialization_error', e => { status.textContent = e.message; });
  player.connect();
  document.getElementById('play').onclick = () => {
    if (!deviceId) { status.textContent = 'still connecting\\u2026'; return; }
    fetch('https://api.spotify.com/v1/me/player/play?device_id=' + deviceId, {
      method: 'PUT', body: JSON.stringify({uris: [uri]}),
      headers: {'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'}
    }).then(r => {
      status.textContent = r.ok ? '\\u266a playing' : 'play failed (' + r.status + ')';
    });
  };
};
</script>
<script src="https://sdk.scdn.co/spotify-player.js"></script>
"""
    return template.replace("__TOKEN__", token).replace("__URI__", track_uri)


# --- Recommendations -----------------------------------------------------
def recommendations_page() -> None:
    cfg = _config()
    st.title("🎧 Music Taste Engine")
    st.caption("Your taste, learned from MERT audio embeddings.")

    with db.session_scope(cfg) as session:  # type: ignore[arg-type]
        total = session.exec(select(func.count()).select_from(Track)).one()
        extracted = session.exec(
            select(func.count()).select_from(Track).where(Track.status == TrackStatus.extracted)
        ).one()
        crawled = session.exec(
            select(func.count())
            .select_from(Track)
            .join(TrackSource)
            .where(TrackSource.source_type == SourceType.crawl)
        ).one()

    col1, col2, col3 = st.columns(3)
    col1.metric("Tracks", total)
    col2.metric("Extracted", extracted)
    col3.metric("Crawled candidates", crawled)
    st.divider()

    left, right = st.columns([3, 1])
    top_k = left.slider("How many recommendations?", 5, 50, 20)
    model = right.selectbox("Model", ["auto", "centroid", "contrastive"])

    if st.button("Recommend", type="primary") or "recs" not in st.session_state:
        with db.session_scope(cfg) as session:  # type: ignore[arg-type]
            st.session_state["recs"] = recommend(cfg, session, top_k=top_k, model=model)

    recs = st.session_state.get("recs", [])
    if recs:
        st.subheader(f"Top {len(recs)} for you")
        st.dataframe(
            [
                {"#": r.rank, "fit": round(r.score, 3), "artist": r.artist, "title": r.title}
                for r in recs
            ],
            hide_index=True,
            width="stretch",
        )
    else:
        st.info("No recommendations yet — extract some tracks, then hit Recommend.")


# --- Audition ------------------------------------------------------------
def rating_page() -> None:
    cfg = _config()
    state = st.session_state
    st.header("🎚️ Audition")
    st.caption(
        "Listen, then move on. How long a track holds you is the signal — "
        "thumbs are optional."
    )

    if "audit_tracks" not in state:
        with db.session_scope(cfg) as session:  # type: ignore[arg-type]
            picked = listening.pick_session_tracks(session, count=20)
            # read attributes while the session is open — Track objects detach
            # once the scope commits and closes.
            state["audit_tracks"] = [
                {
                    "id": t.id,
                    "spotify_id": t.spotify_id,
                    "duration_ms": t.duration_ms,
                    "title": t.title,
                    "artist": t.artist,
                }
                for t in picked
            ]
        state["audit_pos"] = 0

    tracks = state["audit_tracks"]
    pos = state["audit_pos"]
    if not tracks:
        st.info("No un-auditioned extracted tracks — extract or crawl more first.")
        return
    if pos >= len(tracks):
        st.success(f"Session complete — auditioned {len(tracks)} track(s).")
        st.caption("Run `music train` to fold this feedback into the taste model.")
        if st.button("Start a new session"):
            for key in ("audit_tracks", "audit_pos"):
                state.pop(key, None)
            st.rerun()
        return

    track = tracks[pos]
    st.progress(pos / len(tracks), text=f"Track {pos + 1} of {len(tracks)}")

    # Stamp when this track was first shown — dwell time is the silent signal.
    shown_key = f"shown_at_{pos}"
    state.setdefault(shown_key, time.time())

    st.subheader(f"{track['artist']} — {track['title']}")

    # Premium users get full-song playback via the Web Playback SDK; everyone
    # else opens the track in Spotify. Either way, playing it there is a real
    # Spotify play, so `sync-history` later captures the true skip/completion.
    duration_ms = track["duration_ms"] or 0
    token = _spotify_premium_token()
    uri = _resolve_track_uri(track["spotify_id"], track["artist"], track["title"], duration_ms)

    if token and uri:
        components.html(_spotify_player_html(token, uri), height=72)
    elif not token:
        st.info(
            "🎧 **Spotify Premium?** Re-run `music auth` to grant playback and "
            "full songs play right here. Otherwise, open the track in Spotify below."
        )
    else:  # Premium, but no confident Spotify match for this track
        st.caption("Couldn't match this track on Spotify — use the link below.")

    if track["spotify_id"]:
        link = f"https://open.spotify.com/track/{track['spotify_id']}"
    elif uri:
        link = f"https://open.spotify.com/track/{uri.rsplit(':', 1)[-1]}"
    else:
        query = f"{track['artist']} {track['title']}".replace(" ", "%20")
        link = f"https://open.spotify.com/search/{query}"
    st.link_button("Open in Spotify ↗", link)

    st.divider()
    col_up, col_down, col_next = st.columns(3)
    thumb: str | None = None
    if col_up.button("👍 Fits", use_container_width=True):
        thumb = "up"
    if col_down.button("👎 Not for me", use_container_width=True):
        thumb = "down"
    advance = col_next.button("Next ▶", type="primary", use_container_width=True)

    if thumb is not None or advance:
        dwell_s = time.time() - state.get(shown_key, time.time())
        event_type, completion = listening.classify_audition(
            dwell_s, track["duration_ms"] or 0, thumb
        )
        with db.session_scope(cfg) as session:  # type: ignore[arg-type]
            listening.record_audition(session, track["id"], event_type, completion)
        state["audit_pos"] = pos + 1
        state.pop(shown_key, None)
        st.rerun()


# --- Essence siblings ----------------------------------------------------
def siblings_page() -> None:
    cfg = _config()
    st.header("🔗 Essence siblings")
    st.caption("Tag two tracks that *feel* like siblings — extra supervision for M3.")

    with db.session_scope(cfg) as session:  # type: ignore[arg-type]
        rows = session.exec(select(Track).where(Track.status == TrackStatus.extracted)).all()
        options = {f"{t.artist} — {t.title}": t.id for t in rows}

    if len(options) < 2:
        st.info("Need at least 2 extracted tracks.")
        return

    labels = sorted(options)
    track_a = st.selectbox("Track A", labels, key="sibling_a")
    track_b = st.selectbox("Track B", labels, index=1, key="sibling_b")
    strength = st.slider("Sibling strength", 0.0, 1.0, 0.8)

    if st.button("Save sibling pair", type="primary"):
        if options[track_a] == options[track_b]:
            st.error("Pick two different tracks.")
        else:
            with db.session_scope(cfg) as session:  # type: ignore[arg-type]
                session.add(EssenceSibling.create(options[track_a], options[track_b], strength))
            st.success(f"Saved sibling pair: {track_a}  ↔  {track_b}")


st.navigation(
    [
        st.Page(recommendations_page, title="Recommendations", icon="🎯", default=True),
        st.Page(rating_page, title="Rate", icon="🎚️"),
        st.Page(siblings_page, title="Essence siblings", icon="🔗"),
    ]
).run()
