import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory

sys.path.insert(0, os.path.dirname(__file__))
from config import settings

app = Flask(__name__, static_folder=None)
OUTPUT_DIR = Path(settings.OUTPUT_DIR)
if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = Path(__file__).parent.parent / OUTPUT_DIR


def parse_dt(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)


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
    lines = (OUTPUT_DIR / "track_summaries.jsonl").read_text().strip().split("\n")
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
    return send_file(OUTPUT_DIR / filename, mimetype="video/mp4")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=False)
