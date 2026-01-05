import os
from platform import system

from dynaconf import Dynaconf

# establish current platform
SYSTEM = system()

# define path to plaform specific settings
match SYSTEM:
    case "Darwin":
        _SETTINGS_PATH_PLATFORM = os.path.join(
            os.path.dirname(__file__), "settings_mac.toml"
        )
    case "Linux":
        _SETTINGS_PATH_PLATFORM = os.path.join(
            os.path.dirname(__file__), "settings_pi.toml"
        )
    case _:
        raise ValueError(f"Unexpected system: {SYSTEM}")

# load settings
_SETTINGS_PATH_GENEREAL = os.path.join(os.path.dirname(__file__), "settings.toml")
settings = Dynaconf(settings_files=[_SETTINGS_PATH_GENEREAL, _SETTINGS_PATH_PLATFORM])
