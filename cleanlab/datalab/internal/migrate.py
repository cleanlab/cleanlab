"""
This script is the core of our bulk-migration strategy. It finds every legacy
.pkl file matching a user's pattern and migrates it automatically. It's designed
to be run from the command line, providing clear feedback as it works.
"""

from __future__ import annotations

import argparse
import glob
import logging
import sys
from pathlib import Path
from typing import List

from tqdm import tqdm

if __package__ is None and not hasattr(sys, "frozen"):
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(project_root))

from cleanlab.datalab.datalab import Datalab

logger: logging.Logger = logging.getLogger("cleanlab.migration")


def setup_logging(log_file: str | None = None) -> None:
    """Configures the logger to output to console and optionally a file."""
    log_format = logging.Formatter("[%(levelname)s] %(message)s")

    # Always log to the console (stderr)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(log_format)

    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)


def main() -> None:
    """
    Command-line interface for migrating legacy Datalab .pkl files
    to the new, secure component-based format.
    """
    parser = argparse.ArgumentParser(
        description="Bulk migrate legacy .pkl Datalab projects to the new secure format."
    )
    parser.add_argument(
        "glob_pattern",
        type=str,
        help="Glob pattern to find Datalab directories (e.g., './projects/**/'). "
        "The pattern should find directories containing 'datalab.pkl'.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Find and list projects to be migrated without performing any changes.",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to a file to write logs to.",
    )

    args = parser.parse_args()
    setup_logging(args.log_file)

    legacy_pkl_paths: List[str] = glob.glob(args.glob_pattern, recursive=True)

    if not legacy_pkl_paths:
        logger.info("No legacy 'datalab.pkl' files found matching the pattern.")
        return

    # Use a set to get unique parent directories
    project_dirs: List[Path] = sorted(list(set(Path(p).parent for p in legacy_pkl_paths)))

    if not project_dirs:
        logger.info("No legacy Datalab projects found matching the pattern.")
        return

    logger.info(f"Found {len(project_dirs)} potential legacy projects.")

    if args.dry_run:
        logger.info("--- Dry Run Mode ---")
        for proj_dir in project_dirs:
            logger.info(f"Would migrate: {proj_dir}")
        logger.info("--- End Dry Run ---")
        return

    success_count: int = 0
    fail_count: int = 0

    for project_dir in tqdm(project_dirs, desc="Migrating", unit="project"):
        try:
            # Datalab.load handles the actual migration.
            # We just need to call it and trust it to do its job.
            _ = Datalab.load(str(project_dir))
            # tqdm.write ensures logging doesn't mess up the progress bar.
            tqdm.write(f"[INFO] Successfully migrated: {project_dir}")
            success_count += 1
        except Exception as e:
            # We catch any failure during load.
            tqdm.write(f"[ERROR] Failed to migrate {project_dir}. Reason: {e}")
            fail_count += 1

    logger.info("--- Migration Complete ---")
    logger.info(f"  Successfully migrated: {success_count}")
    logger.info(f"  Failed migrations: {fail_count}")
    if fail_count > 0:
        logger.warning("Some projects failed to migrate. Check the logs for details.")


if __name__ == "__main__":
    main()
