import pickle
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Generator, List

import pytest

# We need a Datalab instance to create our legacy test files.
# By placing this test file next to `test_serialize.py`, pytest can
# discover and reuse the `simple_datalab` fixture automatically.

from cleanlab.datalab.datalab import Datalab
from tests.datalab.test_serialize import simple_datalab  # noqa: F401

# --- Constants ---
LEGACY_FILENAME: str = "datalab.pkl"
NEW_INFO_FILENAME: str = "info.json"
NEW_ISSUES_FILENAME: str = "issues.parquet"


@pytest.fixture
def nested_temp_dir() -> Generator[Path, None, None]:
    """Creates a temporary directory with nested subdirectories for migration testing."""
    base_path = Path("test_migration_temp_dir")
    if base_path.exists():
        shutil.rmtree(base_path)

    (base_path / "deep" / "project_b").mkdir(parents=True, exist_ok=True)
    (base_path / "project_a").mkdir(exist_ok=True)
    (base_path / "project_c_corrupted").mkdir(exist_ok=True)

    yield base_path
    shutil.rmtree(base_path)


@pytest.fixture
def populated_legacy_projects(nested_temp_dir: Path, simple_datalab: Datalab) -> Path:
    """Populates a directory structure with both valid and corrupted legacy Datalab projects."""
    project_paths: List[Path] = [d for d in nested_temp_dir.glob("**/*") if d.is_dir()]

    for project_path in project_paths:
        legacy_pkl_path: Path = project_path / LEGACY_FILENAME
        data_path: Path = project_path / "data"

        if "corrupted" in project_path.name:
            # Create a deliberately corrupted pickle file.
            with open(legacy_pkl_path, "wb") as f:
                f.write(b"this is not valid pickle data")
        else:
            # Create a valid legacy project.
            with open(legacy_pkl_path, "wb") as f:
                pickle.dump(simple_datalab, f)
            simple_datalab.data.save_to_disk(str(data_path))
    return nested_temp_dir


def test_migration_cli_e2e(populated_legacy_projects: Path) -> None:
    """
    Tests the full command-line migration script via a subprocess call.
    This end-to-end test verifies that legacy projects are successfully migrated,
    corrupted ones are handled gracefully, and the CLI returns the correct output.
    """
    base_dir: Path = populated_legacy_projects
    glob_pattern: str = str(base_dir / "**" / LEGACY_FILENAME)

    project_root: Path = Path(__file__).resolve().parent.parent.parent
    migrate_script_path: Path = project_root / "cleanlab" / "datalab" / "internal" / "migrate.py"

    # Execute the migration script as a separate process to simulate real-world usage.
    result: subprocess.CompletedProcess[str] = subprocess.run(
        [sys.executable, "-Wd", str(migrate_script_path), glob_pattern],
        capture_output=True,
        text=True,
        check=False,  # Don't raise an exception on non-zero exit codes
    )

    # 1. Assert script execution and output.
    assert (
        result.returncode == 0
    ), f"Script failed unexpectedly.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    # 2. Assert the specific ERROR for the corrupted file is in STDOUT (from tqdm.write).
    assert "[ERROR] Failed to migrate" in result.stdout
    assert str(base_dir / "project_c_corrupted") in result.stdout

    # 3. Assert the final summary in STDERR shows one failure.
    assert "Failed migrations: 1" in result.stderr
    assert "Successfully migrated: 3" in result.stderr
    assert "Some projects failed to migrate" in result.stderr

    # 4. Verify file system state for successfully migrated projects.
    # Note: We check one of the nested projects to ensure recursive search works.
    successful_project_path: Path = base_dir / "deep" / "project_b"
    assert not (
        successful_project_path / LEGACY_FILENAME
    ).exists(), f"Legacy file not deleted in {successful_project_path}"
    assert (
        successful_project_path / NEW_INFO_FILENAME
    ).exists(), f"New info file not created in {successful_project_path}"

    # 5. Verify file system state for the corrupted project.
    corrupted_path: Path = base_dir / "project_c_corrupted"
    assert (
        corrupted_path / LEGACY_FILENAME
    ).exists(), "FAIL: Corrupted file was incorrectly deleted."
    assert not (
        corrupted_path / NEW_INFO_FILENAME
    ).exists(), "FAIL: New files were created for a failed migration."


def test_migration_cli_dry_run(populated_legacy_projects: Path) -> None:
    """
    Tests that the --dry-run flag finds projects but performs no actions.
    A dry run that modifies data is a critical failure. This test ensures it's read-only.
    """
    base_dir: Path = populated_legacy_projects
    glob_pattern: str = str(base_dir / "**" / LEGACY_FILENAME)
    project_root: Path = Path(__file__).resolve().parent.parent.parent
    migrate_script_path: Path = project_root / "cleanlab" / "datalab" / "internal" / "migrate.py"

    result: subprocess.CompletedProcess[str] = subprocess.run(
        [sys.executable, str(migrate_script_path), glob_pattern, "--dry-run"],
        capture_output=True,
        text=True,
        check=True,
    )

    # 1. Assert the output indicates a dry run.
    assert "--- Dry Run Mode ---" in result.stderr
    assert "Would migrate:" in result.stderr

    # 2. CRITICAL: Assert that no files were changed.
    # We check one of the projects to confirm the legacy file still exists.
    project_a_path: Path = base_dir / "project_a"
    assert (project_a_path / LEGACY_FILENAME).exists(), "FAIL: Dry run deleted a legacy file."
    assert not (project_a_path / NEW_INFO_FILENAME).exists(), "FAIL: Dry run created a new file."


def test_migration_cli_log_file(populated_legacy_projects: Path, tmp_path: Path) -> None:
    """Tests that the --log-file argument successfully creates and writes to a log file."""
    base_dir: Path = populated_legacy_projects
    glob_pattern: str = str(base_dir / "**" / LEGACY_FILENAME)
    log_file: Path = tmp_path / "migration.log"

    project_root: Path = Path(__file__).resolve().parent.parent.parent
    migrate_script_path: Path = project_root / "cleanlab" / "datalab" / "internal" / "migrate.py"

    subprocess.run(
        [sys.executable, str(migrate_script_path), glob_pattern, "--log-file", str(log_file)],
        check=True,
    )

    # Assert the log file was created and contains expected output.
    assert log_file.exists(), "Log file was not created."
    log_content: str = log_file.read_text()
    assert "--- Migration Complete ---" in log_content
    assert "Successfully migrated: 3" in log_content


def test_migration_no_files_found(tmp_path: Path) -> None:
    """
    Tests the script's behavior when the glob pattern finds no files.
    This covers the 'if not legacy_pkl_paths:' branch in the script.
    """
    empty_dir: Path = tmp_path
    glob_pattern: str = str(empty_dir / "**" / LEGACY_FILENAME)

    project_root: Path = Path(__file__).resolve().parent.parent.parent
    migrate_script_path: Path = project_root / "cleanlab" / "datalab" / "internal" / "migrate.py"

    result: subprocess.CompletedProcess[str] = subprocess.run(
        [sys.executable, str(migrate_script_path), glob_pattern],
        capture_output=True,
        text=True,
        check=True,
    )

    # Assert we get the specific "not found" message.
    assert "No legacy 'datalab.pkl' files found" in result.stderr
