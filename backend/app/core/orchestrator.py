"""
Master orchestrator — the central pipeline connecting all subsystems.

Flow: Input → [STT] → Context → LLM → [Tool Execution] → [TTS] → Output
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

from app.config import get_config
from app.core.context_manager import ContextManager
from app.database.models import initialize_database
from app.database.repository import Repository
from app.llm.ollama_client import OllamaClient
from app.logger import RequestTimer, get_logger
from app.tools.base import BaseCalendarProvider, BaseWeatherProvider
from app.tools.calendar import SQLiteCalendarAdapter
from app.tools.weather import get_weather_provider

logger = get_logger("orchestrator")


class Orchestrator:
    """
    Central pipeline connecting STT, LLM, tools, and TTS.

    Manages the full request lifecycle from user input to spoken response.
    """

    def __init__(self) -> None:
        # Initialize database
        self.conn = initialize_database()
        self.repo = Repository(self.conn)

        # Resume or create session
        session = self.repo.get_or_create_latest_session()
        self.session_id = session["id"]

        # Initialize subsystems
        self.context = ContextManager(self.repo, self.session_id)
        self.llm = OllamaClient()
        self.weather: BaseWeatherProvider = get_weather_provider()
        self.calendar: BaseCalendarProvider = SQLiteCalendarAdapter(self.repo)

        logger.info(f"Orchestrator initialized (session={self.session_id})")

    def start_new_session(self) -> int:
        """Start a new conversation session."""
        session = self.repo.create_new_session()
        self.session_id = session["id"]
        self.context = ContextManager(self.repo, self.session_id)
        logger.info(f"Started new session: {self.session_id}")
        return self.session_id

    def switch_session(self, session_id: int) -> int:
        """Switch active session."""
        if self.session_id != session_id:
            self.session_id = session_id
            self.context = ContextManager(self.repo, self.session_id)
            logger.info(f"Switched active session to: {self.session_id}")
        return self.session_id

    async def process_text(self, user_text: str, session_id: int | None = None) -> dict[str, Any]:
        """
        Process a text input through the full pipeline.

        Args:
            user_text: The user's text message.
            session_id: Optional session ID override.

        Returns:
            Dict with 'text' (response string) and 'audio' (WAV bytes or None).
        """
        if session_id is not None and session_id != self.session_id:
            self.switch_session(session_id)

        timer = RequestTimer()

        # Save user turn
        self.context.save_turn("user", user_text)

        # Build messages with context
        with timer.measure("context"):
            messages = self.context.get_context_messages()

            # Inject reference context if available
            ref_summary = self.context.get_reference_summary()
            if ref_summary:
                # Add a system note with current context references
                messages.append({
                    "role": "system",
                    "content": f"[Context references: {ref_summary}]",
                })

        # Send to LLM
        with timer.measure("llm"):
            response = await self.llm.chat(messages)

        # Handle tool calls if present
        response_text = response.get("content", "")
        tool_calls = response.get("tool_calls", [])

        if tool_calls:
            with timer.measure("tool"):
                response_text = await self._execute_tool_calls(
                    tool_calls, messages
                )

        # Save assistant turn
        tool_calls_json = json.dumps(tool_calls) if tool_calls else None
        self.context.save_turn("assistant", response_text, tool_calls_json)

        # Synthesize speech (optional, may fail gracefully)
        audio_bytes = None
        try:
            with timer.measure("tts"):
                from app.tts.piper_tts import synthesize_speech
                audio_bytes = synthesize_speech(response_text)
        except Exception as e:
            logger.warning(f"TTS synthesis failed (text-only response): {e}")

        timer.log_metrics(logger)

        return {
            "text": response_text,
            "audio": audio_bytes,
            "session_id": self.session_id,
            "request_id": timer.request_id,
            "metrics": timer.metrics,
        }

    async def process_audio(self, audio_bytes: bytes, session_id: int | None = None) -> dict[str, Any]:
        """
        Process audio input through the full pipeline.

        Args:
            audio_bytes: 16kHz mono WAV audio bytes.
            session_id: Optional session ID override.

        Returns:
            Dict with 'text', 'audio', 'transcription', and metrics.
        """
        if session_id is not None and session_id != self.session_id:
            self.switch_session(session_id)

        timer = RequestTimer()

        # Transcribe audio to text
        with timer.measure("stt"):
            from app.stt.whisper_stt import transcribe_audio
            transcription = transcribe_audio(audio_bytes)

        logger.info(f"Transcribed: '{transcription}'")

        # Process as text
        result = await self.process_text(transcription, session_id=session_id)
        result["transcription"] = transcription
        result["metrics"]["stt_ms"] = timer.metrics.get("stt_ms", 0)

        return result

    async def _execute_tool_calls(
        self,
        tool_calls: list[dict],
        messages: list[dict[str, str]],
    ) -> str:
        """
        Execute tool calls and feed results back to the LLM.

        Args:
            tool_calls: List of tool call dicts from the LLM response.
            messages: The current message history.

        Returns:
            The final assistant response text after tool execution.
        """
        for tc in tool_calls:
            func = tc.get("function", {})
            name = func.get("name", "")
            args = func.get("arguments", {})

            logger.info(f"Executing tool: {name}({json.dumps(args)})")

            try:
                result = await self._dispatch_tool(name, args)
                result_str = json.dumps(result, ensure_ascii=False)
            except Exception as e:
                logger.error(f"Tool execution failed: {name} — {e}")
                result_str = json.dumps({"error": str(e)})

            # Add tool call and result to messages
            messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [tc],
            })
            messages.append({
                "role": "tool",
                "content": result_str,
            })

        # Send tool results back to LLM for natural language response
        final_response = await self.llm.chat(messages, tools=[])
        return final_response.get("content", "I processed your request but couldn't generate a response.")

    async def _dispatch_tool(self, name: str, args: dict[str, Any]) -> Any:
        """
        Route a tool call to the appropriate handler.

        Args:
            name: Tool function name.
            args: Tool function arguments.

        Returns:
            Tool execution result (dict or list).
        """
        if name == "get_weather":
            city = args.get("city", "")
            date = args.get("date")

            # Update context references
            self.context.update_references(location=city)

            if date and date != "today":
                # Resolve relative dates
                resolved = self.context.resolve_relative_date(date)
                if resolved:
                    date = resolved
                    self.context.update_references(date=date)
                return await self.weather.get_forecast(city, date)
            else:
                return await self.weather.get_current(city)

        elif name == "get_upcoming_events":
            limit = args.get("limit", 5)
            return self.calendar.get_upcoming(limit=limit)

        elif name == "get_events_by_date":
            date = args.get("date", "")
            resolved = self.context.resolve_relative_date(date)
            if resolved:
                date = resolved
                self.context.update_references(date=date)
            return self.calendar.get_events_by_date(date)

        elif name == "create_event":
            title = args.get("title", "")
            start_time = args.get("start_time")
            end_time = args.get("end_time")
            location = args.get("location", "")
            description = args.get("description", "")

            # Check for conflicts if time is provided
            if start_time:
                if not end_time:
                    st = datetime.fromisoformat(start_time)
                    end_time = (st + timedelta(hours=1)).isoformat()

                conflicts = self.calendar.check_conflicts(start_time, end_time)
                if conflicts:
                    # Set pending action for conflict resolution
                    self.context.set_pending_action(
                        "conflict_resolution",
                        data={"title": title, "start_time": start_time, "end_time": end_time,
                              "location": location, "conflicts": conflicts},
                    )
                    conflict_names = [c["title"] for c in conflicts]
                    return {
                        "status": "conflict",
                        "message": f"Conflict detected with: {', '.join(conflict_names)}",
                        "conflicts": conflicts,
                    }

            event = self.calendar.create_event(
                title=title,
                start_time=start_time,
                end_time=end_time,
                location=location,
                description=description,
            )
            self.context.update_references(
                event_id=event.get("id"),
                action_event_id=event.get("id"),
            )
            return event

        elif name == "update_event":
            event_id = args.pop("event_id", None)
            if event_id is not None:
                result = self.calendar.update_event(event_id, **args)
                self.context.update_references(action_event_id=event_id)
                return result or {"error": "Event not found"}
            return {"error": "event_id is required"}

        elif name == "delete_event":
            event_id = args.get("event_id")
            if event_id is not None:
                success = self.calendar.delete_event(event_id)
                self.context.update_references(action_event_id=event_id)
                return {"deleted": success, "event_id": event_id}
            return {"error": "event_id is required"}

        elif name == "check_calendar_conflicts":
            start = args.get("start_time", "")
            end = args.get("end_time", "")
            return self.calendar.check_conflicts(start, end)

        else:
            logger.warning(f"Unknown tool: {name}")
            return {"error": f"Unknown tool: {name}"}

    async def health_check(self) -> dict[str, Any]:
        """
        Check the health of all subsystems.

        Returns:
            Dict with status of each subsystem.
        """
        ollama_ok = await self.llm.check_health()

        return {
            "status": "healthy" if ollama_ok else "degraded",
            "ollama": "connected" if ollama_ok else "disconnected",
            "database": "connected",
            "session_id": self.session_id,
            "model": self.llm.model,
        }
