import logging
import os
import sys
from pathlib import Path

import pandas as pd
from flask import Flask, abort, jsonify, request, send_file, send_from_directory

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    METADATA_DIR,
    OUTPUT_DIR,
    TRACK_SUMMARIES_PATH,
    WEB_PLAYER_LOG_PATH,
    settings,
)

logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(WEB_PLAYER_LOG_PATH, "a"),
    ],
)

app = Flask(__name__, static_folder=None)
app.logger.setLevel(settings.LOG_LEVEL)
logging.getLogger("werkzeug").setLevel(settings.LOG_LEVEL)

logger = logging.getLogger(__name__)
HOST, PORT = "127.0.0.1", 5000


class TrainingVideoStore:
    """Persists the set of video filenames selected for training"""

    def __init__(self) -> None:
        self._path = Path(METADATA_DIR) / "training_videos.txt"
        self._videos = (
            set(self._path.read_text().splitlines()) if self._path.exists() else set()
        )
        self._write()

    def get(self) -> set[str]:
        return self._videos

    def update(self, video_name: str, selected: bool) -> set[str]:
        if selected:
            self._videos.add(video_name)
        else:
            self._videos.discard(video_name)
        self._write()
        return self._videos

    def clear(self) -> set[str]:
        self._videos.clear()
        self._write()
        return self._videos

    def _write(self) -> None:
        self._path.write_text("\n".join(self._videos))


training_video_store = TrainingVideoStore()


def get_video_file_path(filename):
    if (path := Path(OUTPUT_DIR) / filename).exists():
        return path.resolve()
    else:
        return None


def get_annotated_video_file_path(filename):
    video_path = get_video_file_path(filename)
    if video_path is None:
        return None
    annotated_path = video_path.with_name(
        f"{video_path.stem}_annotated{video_path.suffix}"
    )
    return annotated_path if annotated_path.is_file() else None


class TrackSummaryStore:
    """Caches parsed track summaries and reloads them when the file changes."""

    def __init__(self) -> None:
        self._df = pd.DataFrame()
        self._mtime = 0.0

    def get(self) -> pd.DataFrame:
        if not self._df.empty:
            pending = ~self._df["files_ready"]
            if pending.any():
                self._df.loc[pending, "files_ready"] = self._df.loc[pending].apply(
                    self._files_ready, axis=1
                )

        mtime = os.path.getmtime(TRACK_SUMMARIES_PATH)
        if mtime == self._mtime:
            return self._df

        self._df = pd.read_json(TRACK_SUMMARIES_PATH, dtype={"manager_id": str})
        self._df["track_start_dt_tm"] = pd.to_datetime(
            self._df["track_start_dt_tm"]
        ).dt.tz_localize(None)
        self._df["behaviours"] = self._df["behaviours"].apply(sorted)
        self._df["duration_s"] = (
            self._df["track_elapsed_end_s"] - self._df["track_elapsed_start_s"]
        )
        self._df["files_ready"] = self._df.apply(self._files_ready, axis=1)

        self._mtime = mtime
        return self._df

    @staticmethod
    def _files_ready(row: pd.Series) -> bool:
        """Check whether the validated annotated copy is ready for playback."""
        return get_annotated_video_file_path(row["video_name"]) is not None


track_summary_store = TrackSummaryStore()

assets = Path(__file__).parent / "static"


@app.route("/")
def index():
    return send_from_directory(assets, "index.html")


@app.route("/style.css")
def style():
    return send_from_directory(assets, "style.css", mimetype="text/css")


@app.route("/app.js")
def app_js():
    return send_from_directory(assets, "app.js", mimetype="text/javascript")


@app.route("/api/tracks")
def get_tracks():
    df = track_summary_store.get()
    training_videos = training_video_store.get()
    df = df.copy()
    df["is_training_video"] = df["video_name"].isin(training_videos)
    filters = {
        "cats": sorted(df["cat_id"].dropna().unique().tolist()),
        "behaviours": sorted({b for v in df["behaviours"] for b in v}),
    }

    if cat := request.args.get("filter_cat_id", ""):
        df = df.loc[df.get("cat_id") == cat]

    if behaviours := set(request.args.getlist("filter_behaviour")):
        df = df.loc[df["behaviours"].apply(lambda v: bool(behaviours.intersection(v)))]

    match request.args.get("filter_training_video"):
        case "training":
            df = df.loc[df["is_training_video"]]
        case "not_training":
            df = df.loc[~df["is_training_video"]]
        case None:
            pass

    if after := request.args.get("filter_track_time_after", ""):
        df = df.loc[df["track_start_dt_tm"] >= pd.Timestamp(after).tz_localize(None)]

    if before := request.args.get("filter_track_time_before", ""):
        df = df.loc[df["track_start_dt_tm"] <= pd.Timestamp(before).tz_localize(None)]

    sort_by = request.args.get("sort_by", "track_start_dt_tm")
    reverse = request.args.get("sort_dir", "desc") == "desc"
    df = df.sort_values(by=sort_by, ascending=not reverse)
    df["track_start_dt_tm"] = df["track_start_dt_tm"].apply(lambda ts: ts.isoformat())
    return jsonify(
        {
            "tracks": df.to_dict(orient="records"),
            "filters": filters,
            "training_videos": list(training_videos),
        }
    )


@app.route("/api/training-videos", methods=["POST", "DELETE"])
def update_training_videos():
    if request.method == "DELETE":
        return jsonify({"training_videos": list(training_video_store.clear())})

    data = request.get_json(silent=True)
    return jsonify(
        {
            "training_videos": list(
                training_video_store.update(
                    data["video_name"], data["is_training_video"]
                )
            )
        }
    )


@app.route("/video/<filename>")
def serve_video(filename):
    video_path = get_annotated_video_file_path(filename) or get_video_file_path(
        filename
    )
    if video_path is None:
        abort(404)
    return send_file(video_path, mimetype="video/mp4")


if __name__ == "__main__":
    logger.info(f"Starting web player on {HOST}:{PORT} behind tailscale serve")
    app.run(host=HOST, port=PORT, debug=False)
