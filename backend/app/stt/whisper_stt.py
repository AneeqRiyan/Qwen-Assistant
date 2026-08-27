"""
Local Speech-to-Text using Faster-Whisper.

Provides lazy-loaded transcription from 16kHz mono WAV audio bytes.
Supports CPU execution with seamless GPU migration via config.
"""

from __future__ import annotations

import io
import wave
from pathlib import Path
from typing import Any

from app.config import get_config
from app.logger import get_logger

logger = get_logger("stt")

# Lazy-loaded model instance
_model: Any = None


def _get_model() -> Any:
    """
    Lazy-load the Faster-Whisper model on first use.

    Returns:
        A WhisperModel instance configured per config.yaml.
    """
    global _model
    if _model is not None:
        return _model

    config = get_config()

    try:
        from faster_whisper import WhisperModel

        device = config.hardware.device  # "cpu" or "cuda"
        compute_type = "int8" if device == "cpu" else "float16"

        logger.info(
            f"Loading Faster-Whisper model '{config.stt.model_size}' "
            f"on device='{device}' (compute_type={compute_type})"
        )

        _model = WhisperModel(
            config.stt.model_size,
            device=device,
            compute_type=compute_type,
        )

        logger.info("Faster-Whisper model loaded successfully")
        return _model

    except ImportError:
        logger.error(
            "faster-whisper is not installed. "
            "Install with: pip install faster-whisper"
        )
        raise
    except Exception as e:
        logger.error(f"Failed to load Faster-Whisper model: {e}")
        raise


def transcribe_audio(audio_bytes: bytes) -> str:
    """
    Transcribe audio bytes to text using Faster-Whisper.

    Args:
        audio_bytes: Raw bytes of a 16kHz mono WAV file.

    Returns:
        Transcribed text string.

    Raises:
        ValueError: If the audio format is invalid.
        RuntimeError: If transcription fails.
    """
    model = _get_model()

    # Validate WAV format
    try:
        with io.BytesIO(audio_bytes) as buf:
            with wave.open(buf, "rb") as wf:
                channels = wf.getnchannels()
                sample_rate = wf.getframerate()
                if channels != 1:
                    logger.warning(f"Expected mono audio, got {channels} channels")
                if sample_rate != 16000:
                    logger.warning(f"Expected 16kHz, got {sample_rate}Hz")
    except wave.Error as e:
        raise ValueError(f"Invalid WAV audio: {e}")

    # Write to a temporary buffer for faster-whisper
    # faster-whisper can accept file paths or numpy arrays
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        segments, info = model.transcribe(
            tmp_path,
            language="en",
            beam_size=5,
            vad_filter=True,  # Filter out silence
        )

        # Collect all segment texts
        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())

        transcribed_text = " ".join(text_parts).strip()

        logger.info(
            f"Transcribed {len(audio_bytes)} bytes → '{transcribed_text[:80]}...' "
            f"(language={info.language}, prob={info.language_probability:.2f})"
        )

        return transcribed_text

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise RuntimeError(f"STT transcription failed: {e}")
    finally:
        os.unlink(tmp_path)


def transcribe_file(file_path: str | Path) -> str:
    """
    Transcribe a WAV file to text.

    Args:
        file_path: Path to a 16kHz mono WAV file.

    Returns:
        Transcribed text string.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    audio_bytes = path.read_bytes()
    return transcribe_audio(audio_bytes)


def reset_model() -> None:
    """Reset the model (useful for testing or switching models)."""
    global _model
    _model = None
