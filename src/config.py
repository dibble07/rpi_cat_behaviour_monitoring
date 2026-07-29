import os
from platform import system

from dynaconf import Dynaconf

# establish current platform
SYSTEM = system()

# load settings
_SETTINGS_PATH_GENERAL = os.path.join(os.path.dirname(__file__), "settings.toml")
settings = Dynaconf(settings_files=_SETTINGS_PATH_GENERAL)

allowed_save_raw_video_modes = {"no", "only", "both"}
if settings.SAVE_RAW_VIDEO not in allowed_save_raw_video_modes:
    raise ValueError(
        f"SAVE_RAW_VIDEO value {settings.SAVE_RAW_VIDEO} must be one of {allowed_save_raw_video_modes}"
    )

settings.EXCLUDED_OBJECTS = {
    x.strip() for x in settings.EXCLUDED_OBJECTS.split(",") if x.strip()
}
