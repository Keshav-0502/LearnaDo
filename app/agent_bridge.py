"""
Thin async bridge between the WhatsApp webhook and the LangGraph agent.
All sync agent calls run in a thread executor so they don't block FastAPI.
"""

import asyncio

from app.agent import generate_outline_from_topic, synthesize_single_lesson


async def get_outline(topic: str) -> list[dict]:
    """Returns [{"title": "...", "description": ""}, ...]"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, generate_outline_from_topic, topic)


async def get_lesson_content(topic: str, lesson_title: str, description: str) -> str:
    """Generate full lesson content for one lesson node. Returns plain text."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, synthesize_single_lesson, topic, lesson_title, description
    )


async def score_confusion(lesson_content: str, learner_response: str) -> float:
    """
    Returns 0.0 (fully understood) → 1.0 (completely confused).
    Uses Gemini Flash — no Tavily needed.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, _score_confusion_sync, lesson_content, learner_response
    )


async def simplify_lesson(content: str) -> str:
    """Rewrite lesson content in simpler, shorter language."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _simplify_lesson_sync, content)


def _score_confusion_sync(lesson_content: str, learner_response: str) -> float:
    from app.agent import get_tool_llm

    prompt = f"""You are evaluating how well a learner understood a lesson.

Lesson content:
{lesson_content[:1500]}

Learner's response:
{learner_response[:500]}

Score the learner's confusion level from 0.0 to 1.0:
- 0.0 = fully understood, clear correct answer
- 0.3 = mostly understood, minor gaps
- 0.6 = partially understood, significant confusion
- 1.0 = completely lost, wrong or no answer

Reply with ONLY a single float like 0.2 or 0.7. Nothing else."""

    try:
        llm = get_tool_llm()
        response = llm.invoke(prompt)
        return float(response.content.strip())
    except (ValueError, Exception):
        return 0.5


def _simplify_lesson_sync(content: str) -> str:
    from app.agent import get_tool_llm

    prompt = f"""Rewrite this lesson in much simpler language for WhatsApp.
- Use very short sentences
- Avoid jargon completely
- Use a real-life everyday example
- Keep it under 600 characters
- End with one simple question to check understanding (e.g. "Quick check: ...")

Original lesson:
{content[:1500]}"""

    try:
        llm = get_tool_llm()
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        return f"{content}\n\n_(Simplified version unavailable: {e})_"
