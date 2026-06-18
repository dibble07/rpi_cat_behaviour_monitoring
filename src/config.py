import os
from platform import system

from dynaconf import Dynaconf

# establish current platform
SYSTEM = system()

# load settings
_SETTINGS_PATH_GENERAL = os.path.join(os.path.dirname(__file__), "settings.toml")
settings = Dynaconf(settings_files=_SETTINGS_PATH_GENERAL)
settings.EXCLUDED_CLASSES = {
    int(x.strip()) for x in settings.EXCLUDED_CLASSES.split(",")
}
