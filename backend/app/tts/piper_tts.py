"""
Local Text-to-Speech synthesis.

Primary: Piper TTS (ONNX neural voice, fast on CPU)
Fallback: pyttsx3 (Windows SAPI5 system voices)

Both produce 16kHz mono WAV audio bytes.
"""

from __future__ import annotations

import io
import struct
import wave
from abc import ABC, abstractmethod
from typing import Any

from app.config import get_config
from app.logger import get_logger

logger = get_logger("tts")


class BaseTTSEngine(ABC):
    """Abstract base for TTS engines."""

    @abstractmethod
    def synthesize(self, text: str) -> bytes:
        """
        Convert text to speech audio.

        Args:
            text: The text to synthesize.

        Returns:
            16kHz mono WAV audio bytes.
        """
        ...


class PiperTTSEngine(BaseTTSEngine):
    """
    Piper TTS — fast neural ONNX-based text-to-speech.

    Produces natural-sounding speech at faster-than-real-time on CPU.
    """

    def __init__(self) -> None:
        self._voice: Any = None
        config = get_config()
        self.voice_name = config.tts.voice

    def _ensure_loaded(self) -> None:
        """Lazy-load the Piper voice model."""
        if self._voice is not None:
            return

        try:
            from piper import PiperVoice

            logger.info(f"Loading Piper TTS voice: {self.voice_name}")
            # Piper downloads voices on first use
            self._voice = PiperVoice.load(self.voice_name)
            logger.info("Piper TTS voice loaded successfully")

        except ImportError:
            logger.error(
                "piper-tts is not installed. "
                "Install with: pip install piper-tts"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load Piper TTS: {e}")
            raise

    def synthesize(self, text: str) -> bytes:
        """Synthesize text to 16kHz mono WAV using Piper TTS."""
        self._ensure_loaded()

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(22050)  # Piper default output rate

            # Generate audio
            audio_stream = self._voice.synthesize_stream_raw(text)
            for audio_chunk in audio_stream:
                wf.writeframes(audio_chunk)

        audio_bytes = buf.getvalue()
        logger.debug(f"Piper TTS: synthesized {len(text)} chars → {len(audio_bytes)} bytes")
        return audio_bytes


class Pyttsx3TTSEngine(BaseTTSEngine):
    """
    Fallback TTS using pyttsx3 (Windows SAPI5 / eSpeak).

    Lower quality but guaranteed to work on Windows without extra setup.
    """

    def __init__(self) -> None:
        self._engine: Any = None

    def _ensure_loaded(self) -> None:
        """Lazy-load the pyttsx3 engine."""
        if self._engine is not None:
            return

        try:
            import pyttsx3

            self._engine = pyttsx3.init()
            # Configure voice settings
            self._engine.setProperty("rate", 170)  # Words per minute
            self._engine.setProperty("volume", 0.9)

            logger.info("pyttsx3 TTS engine initialized (fallback)")

        except ImportError:
            logger.error(
                "pyttsx3 is not installed. "
                "Install with: pip install pyttsx3"
            )
            raise

    def synthesize(self, text: str) -> bytes:
        """Synthesize text to WAV using pyttsx3."""
        self._ensure_loaded()

        import tempfile
        import os

        # pyttsx3 can only save to file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            self._engine.save_to_file(text, tmp_path)
            self._engine.runAndWait()

            with open(tmp_path, "rb") as f:
                audio_bytes = f.read()

            logger.debug(f"pyttsx3 TTS: synthesized {len(text)} chars → {len(audio_bytes)} bytes")
            return audio_bytes

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


# ── Module-level singleton ──
_engine: BaseTTSEngine | None = None


def get_tts_engine() -> BaseTTSEngine:
    """
    Get the configured TTS engine, with automatic fallback.

    Attempts Piper TTS first; falls back to pyttsx3 if Piper fails.

    Returns:
        A BaseTTSEngine instance.
    """
    global _engine
    if _engine is not None:
        return _engine

    config = get_config()

    if config.tts.engine == "pyttsx3":
        logger.info("Using pyttsx3 TTS engine (configured)")
        _engine = Pyttsx3TTSEngine()
        return _engine

    # Try Piper first, fall back to pyttsx3
    try:
        _engine = PiperTTSEngine()
        _engine._ensure_loaded()  # Force early load to detect issues
        return _engine
    except Exception as e:
        logger.warning(f"Piper TTS unavailable ({e}), falling back to pyttsx3")
        try:
            _engine = Pyttsx3TTSEngine()
            return _engine
        except Exception as fallback_error:
            logger.error(f"All TTS engines failed: {fallback_error}")
            raise RuntimeError(
                "No TTS engine available. Install piper-tts or pyttsx3."
            )


def synthesize_speech(text: str) -> bytes:
    """
    Synthesize text to speech using the configured engine.

    Args:
        text: Text to convert to speech.

    Returns:
        WAV audio bytes.
    """
    engine = get_tts_engine()
    return engine.synthesize(text)


def reset_engine() -> None:
    """Reset the TTS engine singleton (useful for testing)."""
    global _engine
    _engine = None
