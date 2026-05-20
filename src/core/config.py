"""Configuration loading.

CLAUDE.md specifies Hydra (omegaconf-backed) for nested overrides. Phase 0 uses
OmegaConf directly — it is the same backing layer, has no global-state quirks,
and is trivially testable. The composition done here (default + model pins +
optional user override) mirrors Hydra's ``defaults`` list and can migrate to
full Hydra when the taste-model trainer needs multirun sweeps.
"""

import hashlib
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from core import paths


def load_config(
    extra: str | Path | None = None,
    overrides: list[str] | None = None,
) -> DictConfig:
    """Load and compose the project configuration.

    Args:
        extra: optional path to a YAML file merged on top of the defaults.
        overrides: optional dotlist overrides, e.g. ``["logging.level=DEBUG"]``.
    """
    cfg = OmegaConf.load(paths.CONFIG_DIR / "default.yaml")
    models = OmegaConf.load(paths.CONFIG_DIR / "models" / "embeddings.yaml")
    cfg = OmegaConf.merge(cfg, {"models": models})

    if extra is not None:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(Path(extra)))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    assert isinstance(cfg, DictConfig)
    return cfg


def config_hash(cfg: DictConfig) -> str:
    """Stable short hash of the extraction-relevant config (invariant 2).

    Covers ``models`` (foundation-model pins) and ``extraction`` (sample rates,
    pooling params). Every features row stores this so re-running extraction
    with a changed config is detectable rather than silently overwriting.
    """
    relevant = OmegaConf.create({"models": cfg.models, "extraction": cfg.extraction})
    text = OmegaConf.to_yaml(relevant, sort_keys=True)
    return hashlib.sha1(text.encode()).hexdigest()[:12]
