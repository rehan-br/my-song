"""Project path resolution.

The project root is the directory containing ``config/`` and ``src/``. Config
files store paths relative to that root; :func:`resolve` turns them absolute.
"""

from pathlib import Path

# core/paths.py -> core -> src -> <project root>
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


def resolve(path: str | Path) -> Path:
    """Resolve ``path`` against the project root if it is relative."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p
