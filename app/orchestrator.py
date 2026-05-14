"""
Orchestrator — single LLM call that returns intent + sentiment together.
Produces an OrchestratorResult consumed by route_message() in router.py.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Literal

from app.agent_bridge import IntentType
from app.utils import clean_text

logger = logging.getLogger(__name__)

SentimentType = Literal["frustrated", "confused", "positive", "neutral"]

# Sentinel values used when the LLM returns malformed JSON
_FALLBACK_INTENT: IntentType = "off_topic"
_FALLBACK_SENTIMENT: SentimentType = "neutral"

# Sentiment → enriched_context injection strings
_SENTIMENT_CONTEXT: dict[str, str] = {
    "frustrated": "User seems frustrated. Be very patient, keep response short and encouraging.",
    "confused": "User seems confused. Simplify language, use a basic example.",
    "positive": "User is engaged. Maintain current difficulty.",
    "neutral": "",
}


@dataclass
class OrchestratorResult:
    """Carries pre-computed intent, sentiment, and an enriched context string."""
    cleaned_text: str
    intent: IntentType
    sentiment: SentimentType
    enriched_context: str


async def analyse(body: str, user_state: str) -> OrchestratorResult:
    """
    Entry point called by webhook.py after user lookup.

    Steps:
      1. Normalise the message text (original body preserved for LLM).
      2. Single LLM call → JSON with intent + sentiment.
      3. Build enriched_context from the sentiment map.
      4. Return OrchestratorResult.
    """
    cleaned = clean_text(body)
    loop = asyncio.get_event_loop()
    intent, sentiment = await loop.run_in_executor(
        None, _classify_sync, cleaned, user_state
    )
    enriched_context = _SENTIMENT_CONTEXT.get(sentiment, "")
    return OrchestratorResult(
        cleaned_text=cleaned,
        intent=intent,
        sentiment=sentiment,
        enriched_context=enriched_context,
    )


def _classify_sync(cleaned_text: str, user_state: str) -> tuple[IntentType, SentimentType]:
    """
    Synchronous LLM call (runs in thread executor).
    Uses gemini-2.0-flash-lite — lightweight model kept on the critical path.
    Falls back to ('off_topic', 'neutral') on any error or bad JSON.
    """
    try:
        import os
        from langchain_google_genai import ChatGoogleGenerativeAI

        # Resolve API key the same way agent.py does
        api_key: str = ""
        try:
            from app.config import settings
            api_key = settings.gemini_api_key or settings.google_api_key or ""
        except Exception:
            pass
        if not api_key:
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or ""

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            google_api_key=api_key,
        )

        prompt = (
            "Classify this WhatsApp message for a learning assistant.\n\n"
            f"State: {user_state}\n"
            f'Message: "{cleaned_text}"\n\n'
            "Intent categories:\n"
            "- greeting: Saying hello, hi, how are you, or introducing themselves.\n"
            "- learning_request: Wants to learn a topic. Includes natural phrases like "
            '"teach me", "explain", "I want to understand", or just a bare topic name.\n'
            "- command: Issuing commands like reset, help, start, yes, no, cancel, "
            "continue, resume. Also includes requests to send a lesson to someone else "
            '("teach my friend", "send to").\n'
            "- off_topic: Questions about the bot itself (\"who are you\", \"what can you do\"), "
            "or anything unrelated to learning.\n"
            "- lesson_answer: Responding to lesson content (only relevant in in_lesson state).\n\n"
            "Return ONLY valid JSON (no markdown, no explanation):\n"
            '{"intent": "<greeting|learning_request|command|off_topic|lesson_answer>", '
            '"sentiment": "<frustrated|confused|positive|neutral>"}'
        )

        response = llm.invoke(prompt)
        raw = (response.content if isinstance(response.content, str) else str(response.content)).strip()

        # Strip accidental markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        data = json.loads(raw)

        valid_intents: set[str] = {"greeting", "learning_request", "command", "off_topic", "lesson_answer"}
        valid_sentiments: set[str] = {"frustrated", "confused", "positive", "neutral"}

        intent: IntentType = data.get("intent", _FALLBACK_INTENT)
        if intent not in valid_intents:
            intent = _FALLBACK_INTENT

        sentiment: SentimentType = data.get("sentiment", _FALLBACK_SENTIMENT)
        if sentiment not in valid_sentiments:
            sentiment = _FALLBACK_SENTIMENT

        return intent, sentiment

    except Exception as exc:
        logger.warning("orchestrator._classify_sync failed (%s); using fallback.", exc)
        return _FALLBACK_INTENT, _FALLBACK_SENTIMENT
