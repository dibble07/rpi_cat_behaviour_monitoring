import json
import logging
import os
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import pandas as pd
from flask import Flask, abort, jsonify, request, send_file, send_from_directory

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    EXT_OUTPUT_DIR,
    INT_OUTPUT_DIR,
    METADATA_DIR,
    TRACK_SUMMARIES_PATH,
    WEB_PLAYER_LOG_PATH,
    settings,
)
from utils import CAT_COLOUR_MAP, OBJECT_COLOUR_MAP

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


def get_video_file_path(filename):
    if (external_path := EXT_OUTPUT_DIR / filename).exists():
        return external_path
    elif (internal_path := INT_OUTPUT_DIR / filename).exists():
        return internal_path
    else:
        return None


@lru_cache(maxsize=None)
def get_video_fps(video_path: str) -> float:
    capture = cv2.VideoCapture(video_path)
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    return fps


class TrackSummaryStore:
    """Caches parsed track summaries and only re-reads lines appended since the last load."""

    def __init__(self) -> None:
        self._df = pd.DataFrame()
        self._mtime = 0.0
        self._offset = 0

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

        with open(TRACK_SUMMARIES_PATH) as f:
            f.seek(self._offset)
            new_df = pd.read_json(f, lines=True, dtype={"manager_id": str})
            self._offset = f.tell()

        if not new_df.empty:
            new_df["track_start_dt_tm"] = pd.to_datetime(
                new_df["track_start_dt_tm"]
            ).dt.tz_localize(None)
            new_df["files_ready"] = new_df.apply(self._files_ready, axis=1)
            self._df = pd.concat([self._df, new_df], ignore_index=True)

        self._mtime = mtime
        return self._df

    @staticmethod
    def _files_ready(row: pd.Series) -> bool:
        """Check all required files are present"""
        hashes_path = Path(METADATA_DIR) / f"video-{row['video_name']}.json"
        bbox_path = (
            Path(METADATA_DIR) / f"track-{row['manager_id']}-{row['track_id']}.json"
        )
        return all(
            [
                get_video_file_path(row["video_name"]) is not None,
                hashes_path.exists(),
                bbox_path.exists(),
            ]
        )


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

    if cat := request.args.get("filter_cat_id", ""):
        df = df.loc[df.get("cat_id") == cat]

    if after := request.args.get("filter_track_time_after", ""):
        df = df.loc[df["track_start_dt_tm"] >= pd.Timestamp(after).tz_localize(None)]

    if before := request.args.get("filter_track_time_before", ""):
        df = df.loc[df["track_start_dt_tm"] <= pd.Timestamp(before).tz_localize(None)]

    sort_by = request.args.get("sort_by", "track_start_dt_tm")
    reverse = request.args.get("sort_dir", "desc") == "desc"
    df = df.sort_values(by=sort_by, ascending=not reverse)
    df["track_start_dt_tm"] = df["track_start_dt_tm"].apply(lambda ts: ts.isoformat())
    return jsonify(df.to_dict(orient="records"))


@app.route("/video/<filename>")
def serve_video(filename):
    video_path = get_video_file_path(filename)
    if video_path is None:
        abort(404)
    return send_file(video_path, mimetype="video/mp4")


@app.route("/api/tracks/<manager_id>/<int:track_id>/annotations")
def get_track_annotations(manager_id, track_id):
    # load track info
    df = track_summary_store.get()
    matches = df[(df["manager_id"] == manager_id) & (df["track_id"] == track_id)]
    if len(matches) != 1:
        abort(404)
    row = matches.iloc[0]

    # read fps from the video file itself, since settings.FPS may have changed since recording
    video_path = get_video_file_path(row["video_name"])
    if video_path is None:
        abort(404)

    # load ordered video frame hashes
    hashes_path = Path(METADATA_DIR) / f"video-{row['video_name']}.json"
    if not hashes_path.exists():
        abort(404)
    hashes = json.loads(hashes_path.read_text())

    # load bbox coordinates
    bbox_path = Path(METADATA_DIR) / f"track-{manager_id}-{track_id}.json"
    if not bbox_path.exists():
        abort(404)
    boxes = json.loads(bbox_path.read_text())

    # annotation colour based on cat id or object
    colour = CAT_COLOUR_MAP.get(
        row["cat_id"], OBJECT_COLOUR_MAP.get(row.get("object_name"), (200, 200, 200))
    )

    return jsonify(
        {
            "fps": get_video_fps(str(video_path)),
            "history_duration_s": settings.TRACK_HISTORY_DUR,
            "frames": [boxes.get(h) for h in hashes],
            "colour": list(reversed(colour)),
        }
    )


if __name__ == "__main__":
    logger.info(f"Starting web player on {HOST}:{PORT} behind tailscale serve")
    app.run(host=HOST, port=PORT, debug=False)
