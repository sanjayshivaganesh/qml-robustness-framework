"""The run writer refuses the archived results tree."""

from pathlib import Path

import pytest

from benchmark.output import (
    ArchivedResultsRefused,
    RunDirectoryExists,
    create_run_dir,
    write_csv_table,
    write_json,
)
from src.paths import PROJECT_ROOT


def _snapshot(root: Path):
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): path.stat().st_mtime_ns
        for path in root.rglob("*")
        if path.is_file()
    }


def test_archive_paths_are_refused_before_any_write():
    archive = PROJECT_ROOT / "results"
    before = _snapshot(archive)
    refused = [
        PROJECT_ROOT,
        archive,
        archive / "noise_robustness_sweep",
        archive / "noise_robustness_sweep" / "summary",
    ]
    for path in refused:
        with pytest.raises(ArchivedResultsRefused):
            create_run_dir(path)
    assert _snapshot(archive) == before


def test_symlink_into_the_archive_is_refused(tmp_path):
    archive = PROJECT_ROOT / "results"
    before = _snapshot(archive)
    link = tmp_path / "archive_alias"
    link.symlink_to(archive / "noise_robustness_sweep", target_is_directory=True)
    with pytest.raises(ArchivedResultsRefused):
        create_run_dir(link)
    with pytest.raises(ArchivedResultsRefused):
        create_run_dir(link / "extra")
    assert _snapshot(archive) == before


def test_non_empty_run_directory_is_refused(tmp_path):
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "already.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(RunDirectoryExists):
        create_run_dir(occupied)
    assert (occupied / "already.txt").read_text(encoding="utf-8") == "keep"


def test_tables_land_in_an_empty_temporary_directory(tmp_path):
    run_dir = create_run_dir(tmp_path / "smoke-run")
    write_json(run_dir, "manifest.json", {"comparable_to_archive": False})
    write_csv_table(run_dir, "rows.csv", [{"procedure_id": "hopfield_synchronous"}], ("procedure_id",))
    assert (run_dir / "manifest.json").is_file()
    assert (run_dir / "rows.csv").read_text(encoding="utf-8").splitlines()[0] == "procedure_id"
    assert not (run_dir / ".manifest.json.partial").exists()
    assert run_dir.is_relative_to(tmp_path)
    writer = (PROJECT_ROOT / "benchmark" / "output.py").read_text(encoding="utf-8")
    assert "results_subdir" not in writer
