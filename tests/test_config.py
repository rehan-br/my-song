"""Tests for configuration loading."""

from core.config import load_config


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
