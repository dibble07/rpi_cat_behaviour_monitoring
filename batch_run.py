from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

manual_subset = []


def main(video_paths: list[Path], clear_output: bool = False) -> int:
    """Run the batch processing workflow"""
    if clear_output:
        from config import OUTPUT_DIR

        for path in Path(OUTPUT_DIR).rglob("*"):
            if path.is_file():
                path.unlink()

    app_path = SRC_ROOT / "app.py"

    for index, video_path in enumerate(video_paths, start=1):
        print(f"[{index}/{len(video_paths)}] Running {video_path}", flush=True)
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
            print(
                f"Stopping after failure in {video_path} (exit code {result.returncode})",
                file=sys.stderr,
                flush=True,
            )
            return result.returncode

        print(f"[{index}/{len(video_paths)}] Completed {video_path}", flush=True)

    print("All videos completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    from utils import get_video_paths

    video_paths = get_video_paths(
        mock_inputs=True, mock_inputs_long=True, raw_behaviour=True
    )

    if manual_subset:
        video_paths = [p for p in video_paths if p.name in manual_subset]

    raise SystemExit(main(video_paths))
