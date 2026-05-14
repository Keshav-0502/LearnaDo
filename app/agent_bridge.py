"""
Thin async bridge between the WhatsApp webhook and the LangGraph agent.
All sync agent calls run in a thread executor so they don't block FastAPI.

Note: classify_intent has been removed — intent is now computed by orchestrator.py
in a single LLM call that also captures sentiment.
"""

import asyncio
from typing import Literal

from app.agent import generate_outline_from_topic, synthesize_single_lesson

IntentType = Literal["greeting", "learning_request", "command", "off_topic", "lesson_answer"]


async def get_outline(topic: str) -> list[dict]:
    """Returns [{"title": "...", "description": ""}, ...]"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, generate_outline_from_topic, topic)


async def get_lesson_content(
    topic: str, lesson_title: str, description: str, context: str = ""
) -> str:
    """
    Generate full lesson content for one lesson node. Returns plain text.
    ``context`` is an enriched_context string from the orchestrator that is
    prepended to the synthesis prompt to influence tone/difficulty.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, _get_lesson_content_sync, topic, lesson_title, description, context
    )


def _get_lesson_content_sync(topic: str, lesson_title: str, description: str, context: str) -> str:
    """Sync wrapper that injects context before delegating to synthesize_single_lesson."""
    # synthesize_single_lesson builds its own prompt internally, so we wrap the
    # result with a context-aware preamble passed via a thin shim prompt when
    # context is non-empty.  For simplicity, and to avoid modifying agent.py,
    # we call synthesize_single_lesson and post-process only if needed.
    # A cleaner approach: pass context as a prefix via description.
    enriched_description = f"{context}\n{description}".strip() if context else description
    return synthesize_single_lesson(topic, lesson_title, enriched_description)


async def score_confusion(lesson_content: str, learner_response: str, context: str = "") -> float:
    """
    Returns 0.0 (fully understood) → 1.0 (completely confused).
    Uses Gemini Flash — no Tavily needed.
    ``context`` is an enriched_context string prepended to the evaluation prompt.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, _score_confusion_sync, lesson_content, learner_response, context
    )


async def simplify_lesson(content: str, context: str = "") -> str:
    """
    Rewrite lesson content in simpler, shorter language.
    ``context`` is an enriched_context string prepended to the rewrite prompt.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _simplify_lesson_sync, content, context)


def _score_confusion_sync(lesson_content: str, learner_response: str, context: str) -> float:
    from app.agent import get_tool_llm

    context_prefix = f"{context}\n\n" if context else ""
    prompt = (
        f"{context_prefix}"
        "You are evaluating how well a learner understood a lesson.\n\n"
        f"Lesson content:\n{lesson_content[:1500]}\n\n"
        f"Learner's response:\n{learner_response[:500]}\n\n"
        "Score the learner's confusion level from 0.0 to 1.0:\n"
        "- 0.0 = fully understood, clear correct answer\n"
        "- 0.3 = mostly understood, minor gaps\n"
        "- 0.6 = partially understood, significant confusion\n"
        "- 1.0 = completely lost, wrong or no answer\n\n"
        "Reply with ONLY a single float like 0.2 or 0.7. Nothing else."
    )

    try:
        llm = get_tool_llm()
        response = llm.invoke(prompt)
        return float(response.content.strip())
    except (ValueError, Exception):
        return 0.5


def _simplify_lesson_sync(content: str, context: str) -> str:
    from app.agent import get_tool_llm

    context_prefix = f"{context}\n\n" if context else ""
    prompt = (
        f"{context_prefix}"
        "Rewrite this lesson in much simpler language for WhatsApp.\n"
        "- Use very short sentences\n"
        "- Avoid jargon completely\n"
        "- Use a real-life everyday example\n"
        "- Keep it under 600 characters\n"
        '- End with one simple question to check understanding (e.g. "Quick check: ...")\n\n'
        f"Original lesson:\n{content[:1500]}"
    )

    try:
        llm = get_tool_llm()
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        return f"{content}\n\n_(Simplified version unavailable: {e})_"
