"""Manual track entry — turns a free-text "<artist> - <title>" into a TrackRef.

The CLI/UI route manual entries here; the resulting TrackRef then flows to
youtube.py for download, like any other source.
"""

import re

from acquisition.base import TrackRef

# artist / title split on a hyphen or en/em dash flanked by whitespace
_SEPARATOR = re.compile(r"\s+[-–—]\s+")


def parse_manual_entry(text: str) -> TrackRef:
    """Parse ``"<artist> - <title>"`` into a :class:`TrackRef`.

    Raises:
        ValueError: if the string is not in the expected shape.
    """
    parts = _SEPARATOR.split(text.strip(), maxsplit=1)
    if len(parts) != 2:
        raise ValueError(f'Expected "<artist> - <title>", got: {text!r}')
    artist, title = parts[0].strip(), parts[1].strip()
    if not artist or not title:
        raise ValueError(f"Empty artist or title in: {text!r}")
    return TrackRef(title=title, artist=artist)
