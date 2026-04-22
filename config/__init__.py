"""
config/__init__.py
Loads settings.yaml and .env, exposes a single Settings object.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Resolve paths relative to this file
_CONFIG_DIR = Path(__file__).parent
_ENV_FILE = _CONFIG_DIR / ".env"
_SETTINGS_FILE = _CONFIG_DIR / "settings.yaml"

# Load .env (silently skip if missing — use .env.example as template)
load_dotenv(dotenv_path=_ENV_FILE, override=False)


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


class Settings:
    """
    Central settings object.
    Merges YAML config with environment variable overrides.
    Access nested keys via attribute or dict-style: settings.data.provider
    """

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Settings(value))
            else:
                setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def __repr__(self) -> str:
        return f"Settings({self._data})"

    def as_dict(self) -> dict[str, Any]:
        return self._data


def load_settings(path: Path | None = None) -> Settings:
    """Load and return the Settings object from settings.yaml."""
    yaml_path = path or _SETTINGS_FILE
    raw = _load_yaml(yaml_path)
    return Settings(raw)


# ── Convenience accessors for environment variables ───────────────────────────

def get_env(key: str, default: str | None = None) -> str | None:
    """Get an environment variable value."""
    return os.environ.get(key, default)


def require_env(key: str) -> str:
    """Get an environment variable or raise if missing."""
    value = os.environ.get(key)
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set. "
            f"Check config/.env (copy from config/.env.example)."
        )
    return value


def is_kill_switch_active() -> bool:
    """
    Check if the global kill switch is active.
    Activated by KILL_SWITCH=true in .env OR by presence of a KILL_SWITCH file
    in the project root.
    """
    env_flag = get_env("KILL_SWITCH", "false").lower() in ("true", "1", "yes")
    file_flag = (Path(__file__).parent.parent / "KILL_SWITCH").exists()
    return env_flag or file_flag


# Module-level singleton — import this everywhere
settings: Settings = load_settings()
