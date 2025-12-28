import os

from dynaconf import Dynaconf

_SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "settings.toml")

settings = Dynaconf(settings_files=[_SETTINGS_PATH])
