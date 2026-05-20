"""Tests for configuration loading."""

from core.config import config_hash, load_config


def test_load_config_defaults() -> None:
    cfg = load_config()
    assert cfg.project.name == "music-taste-engine"
    assert "user-library-read" in cfg.spotify.scopes
    assert cfg.acquisition.duration_tolerance == 0.10


def test_load_config_composes_model_pins() -> None:
    cfg = load_config()
    assert cfg.models.mert.repo == "m-a-p/MERT-v1-330M"
    assert cfg.models.clap.sample_rate == 48000


def test_load_config_dotlist_override() -> None:
    cfg = load_config(overrides=["logging.level=DEBUG"])
    assert cfg.logging.level == "DEBUG"


def test_config_hash_is_stable() -> None:
    h1 = config_hash(load_config())
    h2 = config_hash(load_config())
    assert h1 == h2
    assert len(h1) == 12


def test_config_hash_changes_with_extraction_config() -> None:
    base = config_hash(load_config())
    changed = config_hash(load_config(overrides=["extraction.target_sr_mert=16000"]))
    assert changed != base
