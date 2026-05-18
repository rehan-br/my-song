"""Streamlit research dashboard.

Phase 2 — the "top 20 for me" view: runs the M1 centroid recommender and shows
the ranked list, plus library stats. Launch with ``uv run music ui``.

Streamlit re-runs this script top-to-bottom on every interaction, so the
recommender is only invoked behind a button (results held in session state).
"""

import streamlit as st
from sqlmodel import func, select

from core.config import load_config
from recommend.rank import recommend
from storage import db
from storage.schema import SourceType, Track, TrackSource, TrackStatus

st.set_page_config(page_title="Music Taste Engine", page_icon="🎧", layout="wide")


@st.cache_resource
def _config() -> object:
    return load_config()


cfg = _config()

st.title("🎧 Music Taste Engine")
st.caption("M1 centroid recommender — your taste, learned from MERT audio embeddings.")

# --- Library stats -------------------------------------------------------
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

# --- Recommendations -----------------------------------------------------
top_k = st.slider("How many recommendations?", min_value=5, max_value=50, value=20)

if st.button("Recommend", type="primary") or "recs" not in st.session_state:
    with db.session_scope(cfg) as session:  # type: ignore[arg-type]
        st.session_state["recs"] = recommend(cfg, session, top_k=top_k)

recs = st.session_state.get("recs", [])
if recs:
    st.subheader(f"Top {len(recs)} for you")
    st.dataframe(
        [
            {
                "#": rec.rank,
                "fit": round(rec.score, 3),
                "artist": rec.artist,
                "title": rec.title,
            }
            for rec in recs
        ],
        hide_index=True,
        width="stretch",
    )
else:
    st.info("No recommendations yet — extract some tracks, then hit Recommend.")
