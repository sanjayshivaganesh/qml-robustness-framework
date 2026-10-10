"""Resolve result directories from the repository root, not the cwd."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def experiment_dir(name: str) -> Path:
    """``results/<name>/``, created if missing."""
    path = PROJECT_ROOT / "results" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def results_subdir(experiment_name: str, *parts: str) -> Path:
    """Subdirectory of ``results/<experiment_name>/``."""
    path = experiment_dir(experiment_name).joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path
