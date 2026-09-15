from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from src import utils


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    video_paths = utils.get_video_paths(mock_inputs=True, raw_video=False)
    app_path = repo_root / "src" / "app.py"

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
            cwd=repo_root,
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
