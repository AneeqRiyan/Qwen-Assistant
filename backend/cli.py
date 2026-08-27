"""
Interactive CLI harness for testing the Voice Personal Assistant.

Supports text input (default) and audio file input.
Resumes the last session by default; use --new-session for a fresh start.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Add backend directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.config import get_config
from app.core.orchestrator import Orchestrator
from app.logger import setup_logging, get_logger

logger = get_logger("cli")


def print_banner() -> None:
    """Print the assistant welcome banner."""
    config = get_config()
    name = config.assistant.name

    print("\n" + "=" * 60)
    print(f"  🤖  {name} — Voice Personal Assistant (Phase 1 CLI)")
    print("=" * 60)
    print(f"  Model: {config.llm.model_name}")
    print(f"  Timezone: {config.locale.timezone}")
    print(f"  TTS: {config.tts.engine} | STT: {config.stt.engine}")
    print("-" * 60)
    print("  Type your message and press Enter.")
    print("  Commands: /quit, /new (new session), /history")
    print("=" * 60 + "\n")


async def run_cli(new_session: bool = False, audio_file: str | None = None) -> None:
    """
    Run the interactive CLI loop.

    Args:
        new_session: If True, start a new session instead of resuming.
        audio_file: Optional path to a WAV file for one-shot audio processing.
    """
    setup_logging()
    orchestrator = Orchestrator()

    if new_session:
        orchestrator.start_new_session()
        print("🆕 Started a new session.\n")

    # One-shot audio mode
    if audio_file:
        path = Path(audio_file)
        if not path.exists():
            print(f"❌ Audio file not found: {path}")
            return

        print(f"🎤 Processing audio: {path.name}")
        audio_bytes = path.read_bytes()
        result = await orchestrator.process_audio(audio_bytes)

        print(f"\n📝 Transcription: {result.get('transcription', '(none)')}")
        print(f"🤖 {orchestrator.context.repo.conn}: {result['text']}")
        _print_metrics(result.get("metrics", {}))
        return

    # Interactive text mode
    print_banner()
    config = get_config()
    name = config.assistant.name

    while True:
        try:
            user_input = input(f"  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Goodbye!")
            break

        if not user_input:
            continue

        # Handle CLI commands
        if user_input.lower() in ("/quit", "/exit", "/q"):
            print("\n👋 Goodbye!")
            break

        elif user_input.lower() == "/new":
            orchestrator.start_new_session()
            print("🆕 Started a new session.\n")
            continue

        elif user_input.lower() == "/history":
            turns = orchestrator.repo.get_recent_turns(orchestrator.session_id, n=20)
            print(f"\n📜 Last {len(turns)} turns (Session {orchestrator.session_id}):")
            print("-" * 40)
            for t in turns:
                role = t["role"].upper()
                content = t["content"][:100]
                print(f"  [{role}] {content}")
            print("-" * 40 + "\n")
            continue

        # Process the message
        print(f"  💭 Thinking...")
        result = await orchestrator.process_text(user_input)

        print(f"  {name}: {result['text']}")
        _print_metrics(result.get("metrics", {}))
        print()


def _print_metrics(metrics: dict) -> None:
    """Print latency metrics in a compact format."""
    parts = []
    for key in ("stt_ms", "context_ms", "llm_ms", "tool_ms", "tts_ms", "total_ms"):
        if key in metrics:
            label = key.replace("_ms", "").upper()
            parts.append(f"{label}={metrics[key]:.0f}ms")

    if parts:
        print(f"  ⏱  {' | '.join(parts)}")


def main() -> None:
    """Parse CLI arguments and run."""
    parser = argparse.ArgumentParser(
        description="Voice Personal Assistant — CLI Testing Harness",
    )
    parser.add_argument(
        "--new-session",
        action="store_true",
        help="Start a new conversation session (default: resume last)",
    )
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Path to a WAV audio file for one-shot transcription and response",
    )

    args = parser.parse_args()
    asyncio.run(run_cli(new_session=args.new_session, audio_file=args.audio))


if __name__ == "__main__":
    main()
