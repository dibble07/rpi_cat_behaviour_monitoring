import logging
import os
import sys
from pathlib import Path

import pandas as pd
from flask import Flask, abort, jsonify, request, send_file, send_from_directory

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    EXT_OUTPUT_DIR,
    INT_OUTPUT_DIR,
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


def get_video_file_path(filename):
    if (external_path := EXT_OUTPUT_DIR / filename).exists():
        return external_path
    elif (internal_path := INT_OUTPUT_DIR / filename).exists():
        return internal_path
    else:
        return None


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
    df = pd.read_json(TRACK_SUMMARIES_PATH, lines=True)
    df["track_start_dt_tm"] = pd.to_datetime(df["track_start_dt_tm"]).dt.tz_localize(
        None
    )

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


if __name__ == "__main__":
    logger.info(f"Starting web player on {HOST}:{PORT} behind tailscale serve")
    app.run(host=HOST, port=PORT, debug=False)
