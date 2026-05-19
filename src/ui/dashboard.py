"""Streamlit research dashboard.

Three pages:
- **Recommendations** — runs the taste model (M1 or M2) and shows the ranked list.
- **Rate** — blind listening session: plays a track, captures the vibe/replay/
  skip rubric, no artist/title shown.
- **Essence siblings** — tag two tracks that *feel* like siblings (M3 supervision).

Launch with ``uv run music ui``.
"""

import streamlit as st
from sqlmodel import func, select

from core import paths
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


# --- Blind rating --------------------------------------------------------
def rating_page() -> None:
    cfg = _config()
    state = st.session_state
    st.header("🎚️ Blind rating")
    st.caption("Rate the sound, not the name — artist and title stay hidden.")

    if "rate_tracks" not in state:
        with db.session_scope(cfg) as session:  # type: ignore[arg-type]
            picked = listening.pick_session_tracks(session, count=20)
            # read attributes while the session is still open — the Track
            # objects detach once the scope commits and closes.
            state["rate_tracks"] = [(t.id, t.audio_path) for t in picked]
        state["rate_pos"] = 0

    tracks = state["rate_tracks"]
    pos = state["rate_pos"]
    if not tracks:
        st.info("No unrated extracted tracks — extract some first.")
        return
    if pos >= len(tracks):
        st.success(f"Session complete — rated {len(tracks)} track(s).")
        if st.button("Start a new session"):
            del state["rate_tracks"]
            st.rerun()
        return

    track_id, audio_path = tracks[pos]
    st.progress(pos / len(tracks), text=f"Track {pos + 1} of {len(tracks)}")
    audio_file = paths.resolve(cfg.paths.audio) / str(audio_path)  # type: ignore[attr-defined]
    if audio_file.exists():
        st.audio(str(audio_file))
    else:
        st.warning("Audio file missing for this track.")

    vibe = st.slider("Vibe — does it feel right?", 1, 5, 3, key=f"vibe{pos}")
    replay = st.slider("Replay — would you play it again?", 1, 5, 3, key=f"replay{pos}")
    skip = st.slider("Skip — urge to skip it?", 1, 5, 3, key=f"skip{pos}")
    notes = st.text_input("Notes (optional)", key=f"notes{pos}")

    if st.button("Submit & next", type="primary"):
        with db.session_scope(cfg) as session:  # type: ignore[arg-type]
            listening.record_rating(session, track_id, vibe, replay, skip, notes or None)
        state["rate_pos"] = pos + 1
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
