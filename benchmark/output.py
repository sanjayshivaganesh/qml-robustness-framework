"""Write new runs outside the archived results tree.

This is the only writer the benchmark runner uses. It does not call the
archive locator in ``src.paths`` and it does not plot. A path is refused
when it resolves to the archived results directory, to something inside
that directory, or to a parent of that directory.

The default location for a new run is ``<repo>/runs/<run_id>/``. An
existing run directory is not replaced unless ``overwrite=True``, and that
flag still cannot select the archived tree.
"""

from __future__ import annotations

import csv
import importlib.metadata
import io
import json
import os
import platform
import subprocess
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

from benchmark.log import log_event
from src.paths import PROJECT_ROOT

STATUS_RUNNING = "running"
STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED = "failed"
OUTPUT_CLASS = "framework_run"
_PACKAGE_NAMES = ("numpy", "pandas", "pennylane", "scipy", "matplotlib")


class ArchivedResultsRefused(RuntimeError):
    """The output path would land on the archived results tree."""


class RunDirectoryExists(RuntimeError):
    """The run directory already exists and is not an empty directory."""


def archived_results_root() -> Path:
    """Resolved ``<repo>/results`` directory. This function does not create it."""
    return (PROJECT_ROOT / "results").resolve()


def default_runs_root() -> Path:
    """Resolved ``<repo>/runs`` directory. This function does not create it.

    This is not the archived results tree and it does not call the historical
    path helpers.
    """
    return (PROJECT_ROOT / "runs").resolve()


def assert_output_allowed(path: Path) -> Path:
    """Return the resolved path, or refuse it before any file is created."""
    resolved = Path(path).resolve()
    archive = archived_results_root()
    if resolved == archive or archive in resolved.parents or resolved in archive.parents:
        raise ArchivedResultsRefused(
            f"{resolved} is the archived results directory, inside it, or a parent of it"
        )
    return resolved


def create_run_dir(path: Path, *, overwrite: bool = False) -> Path:
    """Create a run directory after the archive check.

    An existing non-empty directory is refused unless ``overwrite`` is true.
    ``overwrite`` still cannot select the archived results tree: the archive
    check runs first, and a refused path is not created or modified.
    An existing empty directory is reused.
    """
    resolved = assert_output_allowed(path)
    if resolved.exists() and not resolved.is_dir():
        raise RunDirectoryExists(f"{resolved} exists and is not a directory")
    if resolved.is_dir() and any(resolved.iterdir()) and not overwrite:
        raise RunDirectoryExists(
            f"{resolved} is not empty. Pass overwrite=True to replace files "
            "in this framework directory. That flag does not allow the archived results tree."
        )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def allocate_run_directory(parent: Path | None, experiment_id: str) -> tuple[Path, str]:
    """Create a new ``parent/<run_id>`` directory and return both.

    ``parent`` defaults to ``<repo>/runs``. The run id is unique to this
    call. An existing directory of that name is refused rather than reused.
    """
    base = default_runs_root() if parent is None else Path(parent)
    resolved_parent = assert_output_allowed(base)
    if resolved_parent.exists() and not resolved_parent.is_dir():
        raise RunDirectoryExists(f"{resolved_parent} exists and is not a directory")
    resolved_parent.mkdir(parents=True, exist_ok=True)
    run_id = new_run_id(experiment_id)
    run_dir = resolved_parent / run_id
    if run_dir.exists():
        raise RunDirectoryExists(f"{run_dir} already exists and was not overwritten")
    run_dir.mkdir()
    return run_dir, run_id


def new_run_id(experiment_id: str) -> str:
    """Filesystem-safe id: experiment, UTC timestamp, and a random suffix."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{_safe_token(experiment_id)}-{stamp}-{uuid.uuid4().hex[:8]}"


def begin_run(
    run_dir: Path,
    *,
    run_id: str,
    experiment_id: str,
    configuration: dict,
    models,
    dataset: dict,
    perturbation: dict,
    fields: dict | None = None,
) -> dict:
    """Write ``configuration.json`` and a manifest with status ``running``.

    The manifest is not marked succeeded. A later ``complete_run`` or
    ``fail_run`` records the outcome.
    """
    manifest = {
        "run_id": run_id,
        "experiment_id": experiment_id,
        "configuration": configuration,
        "started_at": _utc_now(),
        "completed_at": None,
        "python_version": platform.python_version(),
        "package_versions": package_versions(),
        "git_commit": git_commit(),
        "models": list(models),
        "dataset": dataset,
        "perturbation": perturbation,
        "execution_status": STATUS_RUNNING,
        "output_class": OUTPUT_CLASS,
        "comparable_to_archive": False,
        "resumable": False,
        "files": ["configuration.json", "manifest.json"],
        "error": None,
        "error_type": None,
        "traceback": None,
    }
    if fields:
        for key, value in fields.items():
            if key in {
                "execution_status",
                "completed_at",
                "error",
                "error_type",
                "traceback",
                "comparable_to_archive",
                "resumable",
            }:
                continue
            manifest[key] = value
    manifest["execution_status"] = STATUS_RUNNING
    manifest["completed_at"] = None
    manifest["error"] = None
    manifest["error_type"] = None
    manifest["traceback"] = None
    manifest["comparable_to_archive"] = False
    manifest["resumable"] = False
    write_json(run_dir, "configuration.json", configuration)
    write_json(run_dir, "manifest.json", manifest)
    log_event(
        "manifest_written",
        run_id=run_id,
        status=STATUS_RUNNING,
        resumable=False,
    )
    return manifest


def complete_run(run_dir: Path, manifest: dict, *, files, fields: dict | None = None) -> dict:
    """Mark the manifest succeeded after the result tables exist."""
    updated = dict(manifest)
    if fields:
        updated.update(fields)
    updated["execution_status"] = STATUS_SUCCEEDED
    updated["completed_at"] = _utc_now()
    updated["error"] = None
    updated["error_type"] = None
    updated["traceback"] = None
    updated["comparable_to_archive"] = False
    updated["resumable"] = False
    updated["output_class"] = OUTPUT_CLASS
    updated["files"] = list(files)
    write_json(run_dir, "manifest.json", updated)
    log_event(
        "run_recorded",
        run_id=updated.get("run_id"),
        status=STATUS_SUCCEEDED,
        resumable=False,
    )
    return updated


def fail_run(run_dir: Path, manifest: dict, error: BaseException) -> dict:
    """Record a failed run. The status is ``failed``, not ``succeeded``."""
    updated = dict(manifest)
    updated["execution_status"] = STATUS_FAILED
    updated["completed_at"] = _utc_now()
    updated["error"] = f"{type(error).__name__}: {error}"
    updated["error_type"] = type(error).__name__
    updated["traceback"] = "".join(
        traceback.format_exception(type(error), error, error.__traceback__)
    )
    updated["comparable_to_archive"] = False
    updated["resumable"] = False
    updated["output_class"] = OUTPUT_CLASS
    names = sorted(
        path.name
        for path in Path(run_dir).iterdir()
        if path.is_file() and not path.name.startswith(".")
    )
    if "manifest.json" not in names:
        names.append("manifest.json")
        names.sort()
    updated["files"] = names
    write_json(run_dir, "manifest.json", updated)
    log_event(
        "run_recorded",
        run_id=updated.get("run_id"),
        status=STATUS_FAILED,
        error_type=updated["error_type"],
        resumable=False,
    )
    return updated


def package_versions() -> dict:
    """Installed versions for the scientific stack. Missing packages are null."""
    versions = {}
    for name in _PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def git_commit() -> str | None:
    """HEAD commit, or None when git cannot be read. Does not change the repo."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    commit = completed.stdout.strip()
    return commit or None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_token(value: str) -> str:
    characters = []
    for character in str(value):
        if character.isalnum() or character in "-_":
            characters.append(character)
        else:
            characters.append("-")
    token = "".join(characters).strip("-")
    return token[:80] or "run"


def write_json(run_dir: Path, filename: str, payload) -> Path:
    """Write JSON via a temporary file and ``os.replace``."""
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    return _replace_bytes(run_dir, filename, text.encode("utf-8"))


def write_csv_table(run_dir: Path, filename: str, rows, columns) -> Path:
    """Write a CSV via a temporary file and ``os.replace``.

    ``columns`` is the header and the key order. Missing keys become empty
    cells. Extra keys are not written.
    """
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(columns), extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in columns})
    return _replace_bytes(run_dir, filename, buffer.getvalue().encode("utf-8"))


def _replace_bytes(run_dir: Path, filename: str, data: bytes) -> Path:
    directory = assert_output_allowed(run_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"{directory} is not an existing run directory")
    target = directory / filename
    temporary = directory / f".{filename}.partial"
    temporary.write_bytes(data)
    os.replace(temporary, target)
    return target
