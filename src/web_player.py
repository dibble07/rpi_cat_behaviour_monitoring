import logging
import os
import sys
from pathlib import Path

import pandas as pd
from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    request,
    send_file,
    send_from_directory,
)

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    METADATA_DIR,
    OUTPUT_DIR,
    TRACK_SUMMARIES_PATH,
    WEB_PLAYER_LOG_PATH,
    configure_logging,
    settings,
)

configure_logging(WEB_PLAYER_LOG_PATH)

app = Flask(__name__, static_folder=None)
app.logger.setLevel(settings.LOG_LEVEL)
logging.getLogger("werkzeug").setLevel(settings.LOG_LEVEL)

logger = logging.getLogger(__name__)
HOST, PORT = "127.0.0.1", 5000


class TrainingVideoStore:
    """Persists the set of video filenames selected for training"""

    def __init__(self) -> None:
        self._path = METADATA_DIR / "training_videos.txt"
        self.videos = (
            set(self._path.read_text().splitlines()) if self._path.exists() else set()
        )
        self._save()

    def update(self, video_name: str, selected: bool) -> set[str]:
        """Add or remove a video from the training selection"""
        if selected:
            self.videos.add(video_name)
        else:
            self.videos.discard(video_name)
        return self._save()

    def clear(self) -> set[str]:
        """Remove all videos from the training selection"""
        self.videos.clear()
        return self._save()

    def _save(self) -> set[str]:
        """Write the training selection to disk"""
        self._path.write_text("\n".join(self.videos))
        return self.videos


training_video_store = TrainingVideoStore()


def get_video_file_path(filename: str) -> Path | None:
    """Locate a recorded video in the output directory"""
    if (path := Path(OUTPUT_DIR) / filename).exists():
        return path.resolve()
    else:
        return None


def get_annotated_video_file_path(filename: str) -> Path | None:
    """Locate the annotated copy of a recorded video, if it exists"""
    video_path = get_video_file_path(filename)
    if video_path is None:
        return None
    annotated_path = video_path.with_name(
        f"{video_path.stem}_annotated{video_path.suffix}"
    )
    return annotated_path if annotated_path.is_file() else None


class TrackSummaryStore:
    """Caches parsed track summaries and reloads them when the file changes"""

    def __init__(self) -> None:
        self._df = pd.DataFrame()
        self._mtime = 0.0

    def get(self) -> pd.DataFrame:
        """Provide track summaries, reloading them if the file has changed"""

        # reload summaries when the file changes
        mtime = os.path.getmtime(TRACK_SUMMARIES_PATH)
        if mtime != self._mtime:
            self._df = pd.read_json(TRACK_SUMMARIES_PATH, dtype={"manager_id": str})
            self._df["cat_id"] = self._df["cat_id"].fillna("not a cat")
            self._df["track_start_dt_tm"] = pd.to_datetime(
                self._df["track_start_dt_tm"]
            ).dt.tz_localize(None)
            self._df["behaviours"] = self._df["behaviours"].apply(sorted)
            self._df["duration_s"] = (
                self._df["track_elapsed_end_s"] - self._df["track_elapsed_start_s"]
            )
            self._df["files_ready"] = False
            self._mtime = mtime

        # check whether annotated copies are now ready for playback
        if not self._df.empty:
            pending = ~self._df["files_ready"]
            if pending.any():
                self._df.loc[pending, "files_ready"] = (
                    self._df.loc[pending, "video_name"]
                    .map(get_annotated_video_file_path)
                    .notna()
                )

        return self._df


track_summary_store = TrackSummaryStore()

ASSETS = Path(__file__).parent / "static"


@app.route("/")
def index() -> Response:
    """Serve the web player page"""
    return send_from_directory(ASSETS, "index.html")


@app.route("/style.css")
def style() -> Response:
    """Serve the web player stylesheet"""
    return send_from_directory(ASSETS, "style.css", mimetype="text/css")


@app.route("/app.js")
def app_js() -> Response:
    """Serve the web player script"""
    return send_from_directory(ASSETS, "app.js", mimetype="text/javascript")


@app.route("/api/tracks")
def get_tracks() -> Response:
    """List tracks matching the requested filters and sort order"""

    # load tracks and available filter options
    df = track_summary_store.get().copy()
    training_videos = training_video_store.videos
    df["is_training_video"] = df["video_name"].isin(training_videos)
    filters = {
        "cats": sorted(df["cat_id"].unique().tolist()),
        "behaviours": sorted({b for v in df["behaviours"] for b in v}),
    }

    # filter by cat
    if cat := request.args.get("filter_cat_id", ""):
        df = df.loc[df["cat_id"] == cat]

    # filter by behaviour
    if behaviours := set(request.args.getlist("filter_behaviour")):
        df = df.loc[df["behaviours"].apply(lambda v: bool(behaviours.intersection(v)))]

    # filter by training selection
    match request.args.get("filter_training_video"):
        case "training":
            df = df.loc[df["is_training_video"]]
        case "not_training":
            df = df.loc[~df["is_training_video"]]
        case None:
            pass

    # filter by track start time
    if after := request.args.get("filter_track_time_after", ""):
        df = df.loc[df["track_start_dt_tm"] >= pd.Timestamp(after).tz_localize(None)]
    if before := request.args.get("filter_track_time_before", ""):
        df = df.loc[df["track_start_dt_tm"] <= pd.Timestamp(before).tz_localize(None)]

    # sort and serialise tracks
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
def update_training_videos() -> Response:
    """Change which videos are selected for training"""

    # clear all selections
    if request.method == "DELETE":
        return jsonify({"training_videos": list(training_video_store.clear())})

    # update selection for one video
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
def serve_video(filename: str) -> Response:
    """Stream the annotated video if available, otherwise the original"""
    video_path = get_annotated_video_file_path(filename) or get_video_file_path(
        filename
    )
    if video_path is None:
        abort(404)
    return send_file(video_path, mimetype="video/mp4")


if __name__ == "__main__":
    logger.info(f"Starting web player on {HOST}:{PORT} behind tailscale serve")
    app.run(host=HOST, port=PORT, debug=False)
