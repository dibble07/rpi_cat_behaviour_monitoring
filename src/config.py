import json
import logging
import os
from datetime import datetime
from math import ceil
from pathlib import Path
from platform import system

from dynaconf import Dynaconf

logger = logging.getLogger(__name__)

# establish current platform
SYSTEM = system()
INT_MOUNT = "/"
EXT_MOUNT = "/mnt/hdd"
OUTPUT_DIR_NAME = "object_clips"
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"

# define directories and filenames
if os.path.ismount(EXT_MOUNT):
    OUTPUT_DIR = os.path.join(EXT_MOUNT, OUTPUT_DIR_NAME)
else:
    OUTPUT_DIR = OUTPUT_DIR_NAME
    logger.warning(f"HDD not mounted at {EXT_MOUNT}, falling back to internal storage")

RUN_TIMESTAMP = datetime.now().strftime(TIMESTAMP_FORMAT)
APP_LOG_PATH = os.path.join(OUTPUT_DIR, f"{RUN_TIMESTAMP}_logs.txt")
RAW_LOG_PATH = os.path.join(OUTPUT_DIR, f"{RUN_TIMESTAMP}_raw_logs.txt")
WEB_PLAYER_LOG_PATH = os.path.join(OUTPUT_DIR, f"{RUN_TIMESTAMP}_web_player_logs.txt")
METADATA_DIR = Path(OUTPUT_DIR) / "metadata"
TRACK_SUMMARIES_PATH = METADATA_DIR / "track_summaries.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)

# load settings
_SETTINGS_PATH_GENERAL = os.path.join(os.path.dirname(__file__), "settings.toml")
settings = Dynaconf(settings_files=_SETTINGS_PATH_GENERAL)

# process excluded objects
settings.EXCLUDED_OBJECTS = {
    x.strip() for x in settings.EXCLUDED_OBJECTS.split(",") if x.strip()
}

# define scaled detection image size
settings.DETECTION_IMGSZ = (
    ceil(settings.DETECTION_IMGSZ_W / settings.FRAME_WIDTH * settings.FRAME_HEIGHT / 32)
    * 32,
    settings.DETECTION_IMGSZ_W,
)

# define number of frames a track may remain new
settings.TRACK_NEW_FRAMES = ceil(settings.FPS * settings.TRACK_NEW_DUR)


# Map annotation colours based on object or cat name
OBJECT_COLOUR_MAP = {
    "person": (0, 0, 255),  # red
    "cat": (0, 192, 0),  # green
}
CAT_COLOUR_MAP = {
    "fluffy": (255, 0, 0),  # blue
    "tabby": (0, 165, 255),  # orange
}


def configure_logging(log_path: str) -> None:
    """Send application logs to the console and a log file"""
    logging.basicConfig(
        level=settings.LOG_LEVEL,
        format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, "a"),
        ],
    )


def load_track_summaries() -> list[dict]:
    """Read the saved track summaries, if any have been written yet"""
    if not TRACK_SUMMARIES_PATH.exists():
        return []
    return json.loads(TRACK_SUMMARIES_PATH.read_text())
