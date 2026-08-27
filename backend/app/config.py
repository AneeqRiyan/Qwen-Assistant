"""
Configuration loader for the Voice Personal Assistant.

Reads config.yaml and exposes typed Pydantic settings objects.
Supports environment variable overrides for deployment flexibility.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import yaml
from pydantic import BaseModel, Field


# ── Resolve config path relative to backend/ directory ──
_BACKEND_DIR = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _BACKEND_DIR / "config.yaml"


# ── Nested config models ──

class AssistantConfig(BaseModel):
    """Assistant identity configuration."""
    name: str = "QWEN"


class HardwareConfig(BaseModel):
    """Hardware acceleration settings."""
    device: str = "cpu"  # "cpu" | "cuda"


class LLMConfig(BaseModel):
    """Local LLM (Ollama) configuration."""
    model_name: str = "qwen2.5:3b"
    ollama_base_url: str = "http://localhost:11434"
    temperature: float = 0.7
    context_window_turns: int = 10


class STTConfig(BaseModel):
    """Speech-to-Text configuration."""
    engine: str = "faster-whisper"
    model_size: str = "base.en"


class TTSConfig(BaseModel):
    """Text-to-Speech configuration."""
    engine: str = "piper"  # "piper" | "pyttsx3"
    voice: str = "en_US-lessac-medium"


class WeatherUnitsConfig(BaseModel):
    """Weather units configuration."""
    temperature: str = "celsius"  # "celsius" | "fahrenheit"


class WeatherConfig(BaseModel):
    """Weather provider configuration."""
    provider: str = "open-meteo"  # "open-meteo" | "wttr-in"
    units: WeatherUnitsConfig = Field(default_factory=WeatherUnitsConfig)


class CalendarConfig(BaseModel):
    """Calendar provider configuration."""
    provider: str = "sqlite"  # "sqlite" | "google"


class LocaleConfig(BaseModel):
    """Locale and timezone settings."""
    timezone: str = "Europe/Berlin"


class ServerConfig(BaseModel):
    """FastAPI server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    file: str = "logs/assistant.log"
    max_days: int = 7


class AppConfig(BaseModel):
    """
    Root application configuration.

    Loads from config.yaml at backend/ directory root.
    All sections are optional with sensible defaults.
    """
    assistant: AssistantConfig = Field(default_factory=AssistantConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    weather: WeatherConfig = Field(default_factory=WeatherConfig)
    calendar: CalendarConfig = Field(default_factory=CalendarConfig)
    locale: LocaleConfig = Field(default_factory=LocaleConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def load_config(config_path: Path | str | None = None) -> AppConfig:
    """
    Load application configuration from a YAML file.

    Args:
        config_path: Optional path to config.yaml. Defaults to backend/config.yaml.

    Returns:
        Fully populated AppConfig instance with defaults for missing values.
    """
    path = Path(config_path) if config_path else _CONFIG_PATH

    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    else:
        raw = {}

    # Allow environment variable overrides for key settings
    env_overrides = {
        "ASSISTANT_NAME": ("assistant", "name"),
        "HARDWARE_DEVICE": ("hardware", "device"),
        "LLM_MODEL_NAME": ("llm", "model_name"),
        "OLLAMA_BASE_URL": ("llm", "ollama_base_url"),
        "STT_MODEL_SIZE": ("stt", "model_size"),
        "TTS_ENGINE": ("tts", "engine"),
        "TIMEZONE": ("locale", "timezone"),
        "LOG_LEVEL": ("logging", "level"),
    }

    for env_var, (section, key) in env_overrides.items():
        value = os.environ.get(env_var)
        if value is not None:
            if section not in raw:
                raw[section] = {}
            raw[section][key] = value

    return AppConfig(**raw)


# ── Module-level singleton ──
_config: AppConfig | None = None


def get_config() -> AppConfig:
    """
    Get the application configuration singleton.

    Loads config.yaml on first access, then returns the cached instance.
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reset_config() -> None:
    """Reset the configuration singleton (useful for testing)."""
    global _config
    _config = None
