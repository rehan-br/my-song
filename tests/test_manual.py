"""Tests for manual track-entry parsing."""

import pytest

from acquisition.manual import parse_manual_entry


def test_parse_basic_hyphen() -> None:
    ref = parse_manual_entry("Radiohead - Weird Fishes")
    assert ref.artist == "Radiohead"
    assert ref.title == "Weird Fishes"


def test_parse_en_dash() -> None:
    ref = parse_manual_entry("Boards of Canada – Roygbiv")
    assert ref.artist == "Boards of Canada"
    assert ref.title == "Roygbiv"


def test_parse_splits_only_on_first_separator() -> None:
    ref = parse_manual_entry("Simon - Garfunkel - The Boxer")
    assert ref.artist == "Simon"
    assert ref.title == "Garfunkel - The Boxer"


@pytest.mark.parametrize("bad", ["no separator here", "- missing artist", "Artist -"])
def test_parse_rejects_malformed(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_manual_entry(bad)
