"""
All graph nodes for the LearnaDo LangGraph agent.

Each async function receives the full LearnaDoState dict and returns a
partial dict that gets merged back into state by the graph runner.

Nodes that send WhatsApp messages do so via side-effects (calling
send_message directly); the webhook never needs to handle replies.
"""

import logging
import os
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langchain_anthropic import ChatAnthropic
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from app.graph.state import LearnaDoState

logger = logging.getLogger(__name__)

CONFUSION_THRESHOLD = 0.65
MAX_ATTEMPTS = 3

# ── LLM singleton ──────────────────────────────────────────────────────────────

_llm = None


def _get_api_key() -> str:
    try:
        from app.config import settings

        key = settings.anthropic_api_key
        if key:
            return key
    except Exception:
        pass
    return os.getenv("ANTHROPIC_API_KEY") or ""


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatAnthropic(
            model="claude-haiku-4-5-20251001", api_key=_get_api_key()
        )
    return _llm


def _extract_text(response) -> str:
    """Handle v4 content blocks (list) or plain string."""
    raw = response.content
    if isinstance(raw, list):
        return next(
            (b["text"] for b in raw if isinstance(b, dict) and b.get("type") == "text"),
            "",
        ).strip()
    return str(raw).strip()


# ── Structured-output schemas ──────────────────────────────────────────────────


class _Classification(BaseModel):
    intent: Literal[
        "greeting", "learning_request", "command", "off_topic", "lesson_answer"
    ] = Field(description="Intent of the WhatsApp message.")
    sentiment: Literal["frustrated", "confused", "positive", "neutral"] = Field(
        description="Emotional tone of the message."
    )


class _LessonRequest(BaseModel):
    topic: str = Field(description="The topic the user wants to learn about.")
    learner_phone: str | None = Field(
        default=None,
        description=(
            "Phone number of another person who should learn this, if explicitly "
            "mentioned. Include country code e.g. '+919876543210'. "
            "Null if the user wants to learn themselves."
        ),
    )


_SENTIMENT_CONTEXT: dict[str, str] = {
    "frustrated": "User seems frustrated. Be very patient, keep response short and encouraging.",
    "confused": "User seems confused. Simplify language, use a basic example.",
    "positive": "User is engaged. Maintain current difficulty.",
    "neutral": "",
}


# ── Node functions ─────────────────────────────────────────────────────────────


async def classify_intent(state: LearnaDoState) -> dict:
    """Single structured-output LLM call → intent + sentiment."""
    messages = state.get("messages", [])
    last_msg = messages[-1].content if messages else ""

    llm = _get_llm()
    structured_llm = llm.with_structured_output(
        schema=_Classification.model_json_schema(),
        method="json_schema",
    )

    prompt = (
        "Classify this WhatsApp message for a learning assistant.\n\n"
        f'Message: "{last_msg}"\n\n'
        "Intent categories:\n"
        "- greeting: Saying hello, hi, how are you, or introducing themselves.\n"
        "- learning_request: Wants to learn a topic, OR wants someone else to "
        'learn a topic (e.g. "teach my friend about X", "I want Naman to learn Y", '
        '"send a lesson on Z to 98378..."). Phrases like '
        '"teach me", "explain", "I want to understand", or just a bare topic name.\n'
        "- command: Reset, help, start, yes, no, cancel, continue, resume.\n"
        "- off_topic: Questions about the bot itself, or anything unrelated to learning.\n"
        "- lesson_answer: A direct response to lesson content."
    )

    try:
        result: dict = structured_llm.invoke(prompt)
        intent = result.get("intent", "off_topic")
        sentiment = result.get("sentiment", "neutral")
    except Exception as exc:
        logger.warning("classify_intent failed (%s); using fallback.", exc)
        intent = "off_topic"
        sentiment = "neutral"

    logger.info("Classified: intent=%s sentiment=%s", intent, sentiment)
    return {"intent": intent, "sentiment": sentiment}


async def chat_node(state: LearnaDoState) -> dict:
    """Dynamic conversational reply with multi-turn memory."""
    from app.whatsapp import send_message

    phone = state["phone_number"]
    sentiment = state.get("sentiment", "neutral")
    sentiment_hint = _SENTIMENT_CONTEXT.get(sentiment, "")

    system_prompt = (
        "You are LearnaDo, a friendly AI learning assistant on WhatsApp.\n"
        "Keep replies concise (under 300 chars), conversational, and WhatsApp-friendly.\n"
        "You help people learn things — if they seem interested in a topic, "
        "encourage them to say 'teach me about [topic]' to start a personalised course.\n"
        "Do NOT use markdown headers. Use WhatsApp formatting: *bold*, _italic_.\n"
    )
    if sentiment_hint:
        system_prompt += f"\n{sentiment_hint}\n"

    trimmed = trim_messages(
        state.get("messages", []),
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=4000,
        start_on="human",
    )

    llm = _get_llm()
    response = llm.invoke([SystemMessage(content=system_prompt)] + trimmed)
    reply = _extract_text(response) or "Hey! Tell me what you'd like to learn about."

    await send_message(phone, reply)
    return {"messages": [AIMessage(content=reply)]}


async def extract_lesson_info(state: LearnaDoState) -> dict:
    """Extract topic and optional learner phone via structured output."""
    messages = state.get("messages", [])
    last_msg = messages[-1].content if messages else ""

    llm = _get_llm()
    structured_llm = llm.with_structured_output(
        schema=_LessonRequest.model_json_schema(),
        method="json_schema",
    )

    prompt = (
        "Extract the learning topic and optional learner phone number from this message.\n\n"
        f'Message: "{last_msg}"\n\n'
        "If the user mentions a phone number for someone else to learn, include it "
        "with country code (e.g. +919876543210). If the user wants to learn themselves, "
        "set learner_phone to null."
    )

    try:
        result: dict = structured_llm.invoke(prompt)
        topic = result.get("topic", last_msg)
        learner_phone = result.get("learner_phone")
    except Exception as exc:
        logger.warning("extract_lesson_info failed (%s); using raw message.", exc)
        topic = last_msg
        learner_phone = None

    if learner_phone:
        cleaned = learner_phone.replace(" ", "").replace("-", "")
        if not cleaned.startswith("+"):
            cleaned = "+" + cleaned
        if len(cleaned) < 10 or not cleaned[1:].isdigit():
            learner_phone = None
        else:
            learner_phone = cleaned

    return {"topic": topic, "learner_phone": learner_phone}


async def generate_outline(state: LearnaDoState) -> dict:
    """Generate lesson outline via agent_bridge, send preview to user."""
    from app.agent_bridge import get_outline
    from app.whatsapp import send_message

    phone = state["phone_number"]
    topic = state.get("topic", "")

    await send_message(phone, f"Generating a lesson plan for *{topic}*... give me a moment!")

    try:
        outline = await get_outline(topic)
    except Exception as exc:
        logger.error("Outline generation failed: %s", exc)
        await send_message(
            phone,
            "Sorry, I couldn't generate an outline right now. Try again in a bit!",
        )
        return {"outline": []}

    outline_text = "\n".join(
        f"{i + 1}. *{item.get('title', '')}*" for i, item in enumerate(outline)
    )

    learner_phone = state.get("learner_phone")
    target_msg = f"\n\nThis will be sent to {learner_phone}." if learner_phone else ""

    await send_message(
        phone,
        f"Here's the lesson plan for *{topic}*:\n\n"
        f"{outline_text}{target_msg}\n\n"
        "Reply *yes* to start, or *no* to cancel.",
    )

    return {"outline": outline}


async def review_outline(state: LearnaDoState) -> dict:
    """Interrupt — wait for user to approve or reject the outline."""
    decision = interrupt({"type": "outline_review"})
    message = decision.get("user_message", "").lower().strip()

    approved = message not in ("help", "reset", "start over", "cancel", "no", "n", "nope")
    return {
        "outline_approved": approved,
        "messages": [HumanMessage(content=decision.get("user_message", ""))],
    }


async def cancelled_node(state: LearnaDoState) -> dict:
    """Send cancellation message and end the flow."""
    from app.whatsapp import send_message

    phone = state["phone_number"]
    await send_message(
        phone, "Mission cancelled. Send another message anytime to create a new one."
    )
    return {"messages": [AIMessage(content="Mission cancelled.")]}


async def create_mission(state: LearnaDoState) -> dict:
    """Create mission + lessons in DB, notify accordingly."""
    from sqlalchemy import select

    from app import services
    from app.database import AsyncSessionLocal
    from app.models import User
    from app.whatsapp import send_message

    phone = state["phone_number"]
    topic = state.get("topic", "")
    learner_phone = state.get("learner_phone")
    outline = state.get("outline", [])

    if not outline:
        await send_message(phone, "Something went wrong — no outline. Try again?")
        return {"flow": "error", "mission_id": None}

    target_phone = learner_phone if learner_phone else phone
    if not target_phone.startswith("+"):
        target_phone = "+" + target_phone

    is_single = not learner_phone or target_phone in (phone, "+" + phone)

    async with AsyncSessionLocal() as db:
        user = await services.get_or_create_user(db, phone)
        mission = await services.create_mission_with_outline(
            db, user, target_phone, topic, outline
        )

        result = await db.execute(select(User).where(User.id == mission.learner_id))
        learner = result.scalar_one_or_none()

        if is_single:
            await services.activate_mission(db, mission, learner or user)
            await send_message(
                phone,
                f"Your *{topic}* mission is ready! "
                f"{len(outline)} lessons — let's begin!",
            )
            return {
                "mission_id": str(mission.id),
                "flow": "single",
                "messages": [AIMessage(content=f"Mission created: {topic}")],
            }

        # Dual-user
        await services.activate_mission(db, mission, learner)
        await send_message(
            learner.phone_number,
            f"Hi! *{phone}* has set up a learning mission for you:\n\n"
            f"*{topic}*\n\n"
            f"There are {len(outline)} lessons waiting for you.\n\n"
            "Reply *start* when you're ready to begin!",
        )
        await send_message(
            phone,
            f"Mission sent to {learner.phone_number}!\n\n"
            "I'll message you when they complete each lesson.",
        )
        return {
            "mission_id": str(mission.id),
            "flow": "dual",
            "messages": [AIMessage(content=f"Mission sent to {learner.phone_number}")],
        }


async def wait_for_start(state: LearnaDoState) -> dict:
    """Interrupt — wait for single-user learner to say 'start'."""
    decision = interrupt({"type": "wait_for_start"})
    msg = decision.get("user_message", "").lower().strip()

    if msg in ("help", "reset", "start over", "cancel"):
        return {"flow": "cancelled"}

    return {
        "user_response": msg,
        "messages": [HumanMessage(content=decision.get("user_message", ""))],
    }


async def lesson_handler(state: LearnaDoState) -> dict:
    """Check DB for active mission (entry point for dual-user learner)."""
    from app import services
    from app.database import AsyncSessionLocal
    from app.whatsapp import send_message

    phone = state["phone_number"]

    async with AsyncSessionLocal() as db:
        user = await services.get_or_create_user(db, phone)
        mission = await services.get_active_mission_as_learner(db, user.id)

        if not mission:
            await send_message(
                phone,
                "You don't have an active learning mission right now.\n\n"
                "Tell me what you'd like to learn, or ask someone to set one up for you!",
            )
            return {"mission_id": None}

        is_single = mission.goal_setter_id == mission.learner_id
        return {
            "mission_id": str(mission.id),
            "flow": "single" if is_single else "dual",
            "topic": mission.topic,
        }


async def deliver_lesson(state: LearnaDoState) -> dict:
    """Fetch or generate lesson content and send it to the learner."""
    from app import services
    from app.agent_bridge import get_lesson_content
    from app.database import AsyncSessionLocal
    from app.whatsapp import send_message

    phone = state["phone_number"]
    mission_id = state.get("mission_id")
    sentiment = state.get("sentiment", "neutral")
    enriched = _SENTIMENT_CONTEXT.get(sentiment, "")

    image_url: str | None = None
    youtube_url: str | None = None
    sources: list[dict] = []

    async with AsyncSessionLocal() as db:
        user = await services.get_or_create_user(db, phone)
        mission = await services.get_mission_by_id(db, mission_id)
        if not mission:
            await send_message(phone, "Mission not found. Start over with a new topic!")
            return {"lesson_id": None}

        lesson = await services.get_current_lesson(db, mission.id)
        if not lesson:
            return {"lesson_id": None}

        if not lesson.content_md:
            await send_message(
                phone, f"Loading lesson *{lesson.title}*... one moment!"
            )
            try:
                result = await get_lesson_content(
                    mission.topic, lesson.title, "", context=enriched
                )
                lesson.content_md = result.get("content", "")
                image_url = result.get("image_url")
                youtube_url = result.get("youtube_url")
                sources = result.get("sources", [])
                await db.commit()
            except Exception as e:
                await send_message(
                    phone, f"Sorry, couldn't load that lesson right now. ({e})"
                )
                return {"lesson_id": None}

        await services.create_or_get_progress(db, user.id, mission.id, lesson.id)

        summary = await services.get_mission_progress_summary(db, mission.id)
        progress_line = f"_Lesson {summary['completed'] + 1} of {summary['total']}_\n\n"

        reply = (
            f"{progress_line}"
            f"*{lesson.title}*\n\n"
            f"{lesson.content_md}\n\n"
            "---\n"
            "Reply with what you understood from this lesson."
        )

        if image_url:
            await send_message(phone, reply, media_url=image_url)
        else:
            await send_message(phone, reply)

        return {
            "lesson_id": str(lesson.id),
            "lesson_content": lesson.content_md,
            "lesson_image_url": image_url or "",
            "lesson_youtube_url": youtube_url or "",
            "lesson_sources": sources,
            "attempts": 0,
            "messages": [AIMessage(content=reply)],
        }


async def wait_for_response(state: LearnaDoState) -> dict:
    """Interrupt — wait for the learner's response to a lesson."""
    decision = interrupt({"type": "lesson_response"})
    msg = decision.get("user_message", "")
    return {
        "user_response": msg,
        "messages": [HumanMessage(content=msg)],
    }


async def evaluate_response(state: LearnaDoState) -> dict:
    """Score the learner's reply for confusion."""
    from app import services
    from app.agent_bridge import score_confusion
    from app.database import AsyncSessionLocal

    phone = state["phone_number"]
    user_response = state.get("user_response", "")
    lesson_content = state.get("lesson_content", "")
    mission_id = state.get("mission_id")
    sentiment = state.get("sentiment", "neutral")
    enriched = _SENTIMENT_CONTEXT.get(sentiment, "")
    attempts = state.get("attempts", 0)

    async with AsyncSessionLocal() as db:
        user = await services.get_or_create_user(db, phone)
        mission = await services.get_mission_by_id(db, mission_id)
        if not mission:
            return {"should_simplify": False, "has_next_lesson": False}

        progress = await services.get_current_progress(db, user.id, mission.id)
        if not progress:
            return {"should_simplify": False, "has_next_lesson": False}

        confusion = await score_confusion(
            lesson_content, user_response, context=enriched
        )
        await services.record_attempt(db, progress, confusion)

    new_attempts = attempts + 1

    if new_attempts >= MAX_ATTEMPTS:
        return {
            "confusion_score": confusion,
            "should_simplify": False,
            "attempts": new_attempts,
        }

    if confusion > CONFUSION_THRESHOLD:
        return {
            "confusion_score": confusion,
            "should_simplify": True,
            "attempts": new_attempts,
        }

    return {
        "confusion_score": confusion,
        "should_simplify": False,
        "attempts": new_attempts,
    }


async def simplify_node(state: LearnaDoState) -> dict:
    """Rewrite lesson in simpler language, send to learner."""
    from app.agent_bridge import simplify_lesson
    from app.whatsapp import send_message

    phone = state["phone_number"]
    lesson_content = state.get("lesson_content", "")
    sentiment = state.get("sentiment", "neutral")
    enriched = _SENTIMENT_CONTEXT.get(sentiment, "")
    attempts = state.get("attempts", 0)

    simplified = await simplify_lesson(lesson_content, context=enriched)
    attempts_left = MAX_ATTEMPTS - attempts
    suffix = "s" if attempts_left != 1 else ""

    reply = (
        f"Let me explain that differently!\n\n"
        f"{simplified}\n\n"
        "---\n"
        f"Have another go — what did you understand? "
        f"_({attempts_left} attempt{suffix} left)_"
    )

    await send_message(phone, reply)
    return {"lesson_content": simplified, "messages": [AIMessage(content=reply)]}


async def advance_lesson(state: LearnaDoState) -> dict:
    """Mark lesson complete, notify goal-setter (dual), check for next lesson."""
    from sqlalchemy import select

    from app import services
    from app.database import AsyncSessionLocal
    from app.models import User
    from app.whatsapp import send_message

    phone = state["phone_number"]
    mission_id = state.get("mission_id")
    confusion = state.get("confusion_score", 0.0)
    flow = state.get("flow", "single")

    async with AsyncSessionLocal() as db:
        user = await services.get_or_create_user(db, phone)
        mission = await services.get_mission_by_id(db, mission_id)
        if not mission:
            return {"has_next_lesson": False}

        lesson = await services.get_current_lesson(db, mission.id)
        if not lesson:
            return {"has_next_lesson": False}

        progress = await services.get_current_progress(db, user.id, mission.id)
        if progress:
            forced = (
                state.get("attempts", 0) >= MAX_ATTEMPTS
                and confusion > CONFUSION_THRESHOLD
            )
            await services.complete_lesson(db, progress, lesson)

            if flow == "dual":
                gs_result = await db.execute(
                    select(User).where(User.id == mission.goal_setter_id)
                )
                goal_setter = gs_result.scalar_one_or_none()
                if goal_setter and goal_setter.id != user.id:
                    summary = await services.get_mission_progress_summary(
                        db, mission.id
                    )
                    forced_note = (
                        " _(moved on after max attempts)_" if forced else ""
                    )
                    await send_message(
                        goal_setter.phone_number,
                        f"*{mission.topic}* update:\n\n"
                        f"Your learner completed: *{lesson.title}*{forced_note}\n\n"
                        f"Progress: {summary['completed']}/{summary['total']} lessons done.",
                    )

        next_lesson = await services.get_next_lesson(db, mission.id, lesson.order_index)

    if next_lesson:
        if confusion <= CONFUSION_THRESHOLD:
            await send_message(phone, "Great job! You understood that well.\n")
        return {"has_next_lesson": True}

    return {"has_next_lesson": False}


async def mission_complete(state: LearnaDoState) -> dict:
    """Mark mission done, notify both users."""
    from datetime import datetime, timezone

    from sqlalchemy import select

    from app import services
    from app.database import AsyncSessionLocal
    from app.models import User
    from app.whatsapp import send_message

    phone = state["phone_number"]
    mission_id = state.get("mission_id")
    flow = state.get("flow", "single")

    async with AsyncSessionLocal() as db:
        mission = await services.get_mission_by_id(db, mission_id)
        if mission:
            mission.status = "completed"
            mission.completed_at = datetime.now(timezone.utc)
            await db.commit()

            if flow == "dual":
                gs_result = await db.execute(
                    select(User).where(User.id == mission.goal_setter_id)
                )
                goal_setter = gs_result.scalar_one_or_none()
                if goal_setter:
                    await send_message(
                        goal_setter.phone_number,
                        f"Your learner has completed the full *{mission.topic}* course!\n\n"
                        "All lessons finished. Well done to both of you!",
                    )

            topic = mission.topic
        else:
            topic = "your course"

    await send_message(
        phone,
        f"Congratulations! You've completed all lessons on *{topic}*!\n\n"
        "You can start a new learning mission anytime by messaging again.",
    )
    return {"messages": [AIMessage(content=f"Mission {topic} completed!")]}
