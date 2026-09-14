import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_file, send_from_directory

sys.path.insert(0, os.path.dirname(__file__))
from config import EXT_MOUNT, settings

os.makedirs(settings.OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(
                settings.OUTPUT_DIR,
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_web_player_logs.txt",
            ),
            "a",
        ),
    ],
)

app = Flask(__name__, static_folder=None)
app.logger.setLevel(settings.LOG_LEVEL)
logging.getLogger("werkzeug").setLevel(settings.LOG_LEVEL)

INT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / settings.OUTPUT_DIR
EXT_OUTPUT_DIR = Path(EXT_MOUNT) / settings.OUTPUT_DIR
logger = logging.getLogger(__name__)


def parse_dt(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)


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
    lines = (INT_OUTPUT_DIR / "track_summaries.jsonl").read_text().strip().split("\n")
    tracks = [json.loads(line) for line in lines if line]

    cat = request.args.get("filter_cat_id", "")
    if cat:
        tracks = [t for t in tracks if t.get("cat_id") == cat]

    after = request.args.get("filter_track_time_after", "")
    if after:
        after_dt = parse_dt(after)
        tracks = [t for t in tracks if parse_dt(t["track_start_timestamp"]) >= after_dt]

    before = request.args.get("filter_track_time_before", "")
    if before:
        before_dt = parse_dt(before)
        tracks = [
            t for t in tracks if parse_dt(t["track_start_timestamp"]) <= before_dt
        ]

    sort_by = request.args.get("sort_by", "track_start_timestamp")
    reverse = request.args.get("sort_dir", "desc") == "desc"
    tracks.sort(key=lambda t: t.get(sort_by, ""), reverse=reverse)
    return jsonify(tracks)


@app.route("/video/<filename>")
def serve_video(filename):
    video_path = get_video_file_path(filename)
    if video_path is None:
        abort(404)
    return send_file(video_path, mimetype="video/mp4")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    logger.info(f"Starting web player on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
