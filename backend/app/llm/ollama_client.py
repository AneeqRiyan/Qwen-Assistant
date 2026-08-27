"""
Ollama REST API client for local LLM inference.

Communicates with the local Ollama server to perform chat completion
with optional tool/function calling support.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from app.config import get_config
from app.llm.prompts import TOOL_SCHEMAS, build_system_prompt
from app.logger import get_logger

logger = get_logger("llm")


class OllamaClient:
    """
    Async client for the Ollama REST API.

    Handles chat completion requests with system prompts,
    conversation history, and tool calling.
    """

    def __init__(self) -> None:
        config = get_config()
        self.base_url = config.llm.ollama_base_url.rstrip("/")
        self.model = config.llm.model_name
        self.temperature = config.llm.temperature

    async def check_health(self) -> bool:
        """
        Check if the Ollama server is reachable and the model is loaded.

        Returns:
            True if healthy, False otherwise.
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                resp.raise_for_status()
                data = resp.json()

                models = [m["name"] for m in data.get("models", [])]
                # Check if our model (or its base name) is available
                model_available = any(
                    self.model in m or m.startswith(self.model.split(":")[0])
                    for m in models
                )

                if model_available:
                    logger.info(f"Ollama health OK — model '{self.model}' available")
                else:
                    logger.warning(
                        f"Ollama running but model '{self.model}' not found. "
                        f"Available: {models}. Run: ollama pull {self.model}"
                    )

                return True

        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False

    async def chat(
        self,
        messages: list[dict[str, str]],
        tools: list[dict] | None = None,
    ) -> dict[str, Any]:
        """
        Send a chat completion request to Ollama.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
                      Roles: 'system', 'user', 'assistant', 'tool'
            tools: Optional list of tool schemas for function calling.
                   Defaults to TOOL_SCHEMAS if None.

        Returns:
            Dict with:
                - 'content': The assistant's text response
                - 'tool_calls': List of tool call dicts (if any)
                - 'role': Always 'assistant'
        """
        if tools is None:
            tools = TOOL_SCHEMAS

        # Build the system prompt with current datetime
        system_prompt = build_system_prompt()

        # Prepend system message if not already present
        if not messages or messages[0].get("role") != "system":
            messages = [{"role": "system", "content": system_prompt}] + messages

        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "stream": False,
            "options": {
                "temperature": self.temperature,
            },
        }

        logger.debug(
            f"Sending chat request: model={self.model}, "
            f"messages={len(messages)}, tools={len(tools)}"
        )

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()

        except httpx.TimeoutException:
            logger.error("Ollama request timed out (120s)")
            return {
                "role": "assistant",
                "content": "I'm sorry, I took too long to think about that. Could you try again?",
                "tool_calls": [],
            }
        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama HTTP error: {e.response.status_code} — {e.response.text}")
            return {
                "role": "assistant",
                "content": "I'm having trouble thinking right now. Please try again in a moment.",
                "tool_calls": [],
            }
        except Exception as e:
            logger.error(f"Ollama request failed: {e}")
            return {
                "role": "assistant",
                "content": "I'm sorry, I encountered an error. Please make sure Ollama is running.",
                "tool_calls": [],
            }

        # Parse the response
        message = data.get("message", {})
        content = message.get("content", "")
        tool_calls = message.get("tool_calls", [])

        if tool_calls:
            logger.info(
                f"LLM returned {len(tool_calls)} tool call(s): "
                f"{[tc.get('function', {}).get('name', '?') for tc in tool_calls]}"
            )
        else:
            logger.debug(f"LLM response: '{content[:100]}...'")

        return {
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls,
        }

    async def chat_simple(self, user_message: str) -> str:
        """
        Simple single-turn chat without tool calling.

        Useful for small talk or quick responses.

        Args:
            user_message: The user's text input.

        Returns:
            The assistant's text response.
        """
        messages = [{"role": "user", "content": user_message}]
        result = await self.chat(messages, tools=[])
        return result.get("content", "")
