from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import utils


def main() -> int:
    video_paths = utils.get_video_paths(mock_inputs=True, raw_video=False)
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

    print("All mock input videos completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
