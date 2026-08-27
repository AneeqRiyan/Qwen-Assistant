"""
System prompts and tool schema definitions for Ollama function calling.

The system prompt defines QWEN's personality, capabilities, and boundaries.
Tool schemas define the JSON structure for weather and calendar operations.
"""

from __future__ import annotations

from datetime import datetime

from app.config import get_config


def build_system_prompt(current_datetime: datetime | None = None) -> str:
    """
    Build the system prompt for the LLM.

    Args:
        current_datetime: Optional override for current time. Uses now() if None.

    Returns:
        The full system prompt string.
    """
    config = get_config()
    name = config.assistant.name
    tz = config.locale.timezone

    if current_datetime is None:
        import pytz
        timezone = pytz.timezone(tz)
        current_datetime = datetime.now(timezone)

    dt_str = current_datetime.strftime("%A, %B %d, %Y at %H:%M")

    return f"""You are {name}, a friendly and helpful personal voice assistant.

## Current Context
- Current date and time: {dt_str}
- Timezone: {tz}

## Your Personality
- You are warm, conversational, and approachable
- You enjoy casual small talk: greetings, jokes, how-are-you exchanges, general chat
- You give concise but helpful responses — suitable for spoken output
- You speak naturally, as if having a face-to-face conversation

## Your Capabilities
You have access to two tools:

1. **Weather Lookup** — You can check current weather conditions and forecasts for any city.
   When reporting weather, include: temperature, conditions, and chance of rain.

2. **Calendar Management** — You can create, view, update, and delete calendar appointments.
   When creating events:
   - If the user doesn't specify a time, ask them what time the appointment should be
   - Before creating, check for scheduling conflicts
   - If a conflict exists, inform the user and ask how they want to proceed
   - Confirm successful operations clearly

## Context Resolution Rules
- If the user says "there" or "that place", refer to the most recently mentioned location
- If the user says "that day" or uses a pronoun for time, refer to the most recently mentioned date
- If the user says "the previously created appointment" or "that appointment", refer to the last calendar action
- Relative dates like "today", "tomorrow", "this Friday" should be resolved based on the current date above

## Boundaries
- You CANNOT browse the internet or access any external service beyond weather and calendar
- You CANNOT generate code, write files, or perform system operations
- You CANNOT access email, messages, or any other personal data
- If asked about something outside your capabilities, politely explain what you can help with
- Keep responses concise — they will be spoken aloud via text-to-speech

## Response Format
- Respond in natural, spoken English
- Do NOT use markdown formatting, bullet points, or code blocks in your responses
- Keep answers concise (1-3 sentences for simple queries)
- For weather: state the key facts naturally (e.g., "It will be 22 degrees and partly cloudy in Marburg today, with a 15% chance of rain.")
- For calendar confirmations: be clear and specific (e.g., "Done! I've added 'Team Meeting' to your calendar for January 12th at 2pm.")
"""


# ── Tool Schemas for Ollama Function Calling ──

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather or forecast for a city on a specific date. Use this when the user asks about weather conditions, temperature, rain, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name to get weather for (e.g., 'Marburg', 'Frankfurt')",
                    },
                    "date": {
                        "type": "string",
                        "description": "The date for the forecast in YYYY-MM-DD format. Use 'today' for current weather.",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_upcoming_events",
            "description": "Get upcoming calendar events. Use when the user asks about their schedule, next appointment, or what's coming up.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of events to return. Default is 5.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_events_by_date",
            "description": "Get all calendar events on a specific date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "The date in YYYY-MM-DD format.",
                    },
                },
                "required": ["date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_event",
            "description": "Create a new calendar appointment. If the user doesn't specify a time, ask before creating.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "The title/name of the event.",
                    },
                    "start_time": {
                        "type": "string",
                        "description": "Start time in ISO format (YYYY-MM-DDTHH:MM:SS). If unknown, set to null.",
                    },
                    "end_time": {
                        "type": "string",
                        "description": "End time in ISO format. Defaults to start_time + 1 hour if not specified.",
                    },
                    "location": {
                        "type": "string",
                        "description": "Event location. Empty string if not specified.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Event description. Empty string if not specified.",
                    },
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_event",
            "description": "Update an existing calendar event. Use when the user wants to change details of an appointment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_id": {
                        "type": "integer",
                        "description": "The ID of the event to update.",
                    },
                    "title": {
                        "type": "string",
                        "description": "New title (if changing).",
                    },
                    "start_time": {
                        "type": "string",
                        "description": "New start time in ISO format (if changing).",
                    },
                    "end_time": {
                        "type": "string",
                        "description": "New end time in ISO format (if changing).",
                    },
                    "location": {
                        "type": "string",
                        "description": "New location (if changing).",
                    },
                },
                "required": ["event_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_event",
            "description": "Delete a calendar event by its ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_id": {
                        "type": "integer",
                        "description": "The ID of the event to delete.",
                    },
                },
                "required": ["event_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_calendar_conflicts",
            "description": "Check if there are any scheduling conflicts for a given time range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {
                        "type": "string",
                        "description": "Start time in ISO format.",
                    },
                    "end_time": {
                        "type": "string",
                        "description": "End time in ISO format.",
                    },
                },
                "required": ["start_time", "end_time"],
            },
        },
    },
]
