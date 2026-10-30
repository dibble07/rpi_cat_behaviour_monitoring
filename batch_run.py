from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"

# make application modules importable when this script runs directly
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# keep batch progress separate from application logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())

MANUAL_SUBSET = []


def main(video_paths: list[Path], clear_output: bool = False) -> int:
    """Run the batch processing workflow"""

    # remove files from the previous batch when requested
    if clear_output:
        from config import OUTPUT_DIR

        for path in Path(OUTPUT_DIR).rglob("*"):
            if path.is_file():
                path.unlink()

    app_path = SRC_ROOT / "app.py"

    # run each video in its own application process
    for index, video_path in enumerate(video_paths, start=1):
        logger.info(f"[{index}/{len(video_paths)}] Running {video_path}")
        result = subprocess.run(
            [
                sys.executable,
                "-u",
                str(app_path),
                "--video",
                str(video_path),
            ],
            cwd=REPO_ROOT,
        )
        if result.returncode != 0:
            logger.error(
                f"Stopping after failure in {video_path} "
                f"(exit code {result.returncode})"
            )
            return result.returncode

        logger.debug(f"[{index}/{len(video_paths)}] Completed {video_path}")

    logger.info("All videos completed successfully")
    return 0


if __name__ == "__main__":
    from utils import get_video_paths

    # select the configured mock videos for this batch
    video_paths = get_video_paths(
        mock_inputs=True, mock_inputs_long=True, raw_behaviour=True
    )

    if MANUAL_SUBSET:
        video_paths = [p for p in video_paths if p.name in MANUAL_SUBSET]

    raise SystemExit(main(video_paths))
