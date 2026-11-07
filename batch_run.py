from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import tenacity

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


@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=5),
    before_sleep=tenacity.before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def run_video(video_path: Path) -> None:
    """Run the application for one video"""
    subprocess.run(
        [sys.executable, "-u", str(SRC_ROOT / "app.py"), "--video", str(video_path)],
        check=True,
        cwd=REPO_ROOT,
    )


def main(video_paths: list[Path], clear_output: bool = False) -> int:
    """Run the batch processing workflow"""

    # remove files from the previous batch when requested
    if clear_output:
        from config import OUTPUT_DIR

        for path in Path(OUTPUT_DIR).rglob("*"):
            if path.is_file():
                path.unlink()

    # run each video in its own application process
    lg = logger.info if len(video_paths) > 1 else logger.debug
    for index, video_path in enumerate(video_paths, start=1):
        lg(f"[{index}/{len(video_paths)}] Running {video_path}")
        try:
            run_video(video_path)
        except subprocess.CalledProcessError as error:
            logger.error(
                f"Stopping after failure in {video_path} (exit code {error.returncode})"
            )
            return error.returncode

        logger.debug(f"[{index}/{len(video_paths)}] Completed {video_path}")

    logger.debug("All videos completed successfully")
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
