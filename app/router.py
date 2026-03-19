"""
WhatsApp state machine — dual-user flow (goal-setter vs learner).
Phase 4: full lesson delivery, confusion scoring, adaptive retry, completion.
"""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent_bridge import get_lesson_content, get_outline, score_confusion, simplify_lesson
from app.models import User
from app.services import (
    activate_mission,
    cancel_mission,
    complete_lesson,
    create_mission_with_outline,
    create_or_get_progress,
    get_active_mission_as_learner,
    get_current_lesson,
    get_current_progress,
    get_mission_by_id,
    get_mission_progress_summary,
    get_next_lesson,
    record_attempt,
    set_user_state,
)
from app.whatsapp import send_message

CONFUSION_THRESHOLD = 0.65
MAX_ATTEMPTS = 3


async def route_message(
    db: AsyncSession,
    user: User,
    body: str,
    media_url: str | None,
    media_type: str | None,
) -> tuple[str, str | None]:
    """
    Returns (reply_text, reply_media_url).
    reply_media_url is None for text-only replies.
    """
    state = user.wa_session_state or "idle"
    body_clean = body.strip().lower()

    # ── SHARED: any user can type 'help' or 'reset' ──────────────────
    if body_clean in ("help", "reset", "start over"):
        await set_user_state(db, user, "idle")
        return (
            "Restarted! Reply *teach* to set up a learning mission, "
            "or just wait if someone has already set one up for you.",
            None,
        )

    # ── IDLE ─────────────────────────────────────────────────────────
    if state == "idle":
        if "phone:" in body_clean and "topic:" in body_clean:
            await set_user_state(db, user, "creating_mission")
            return await handle_creating_mission(db, user, body)

        mission = await get_active_mission_as_learner(db, user.id)
        if mission:
            await set_user_state(db, user, "mission_notified")
            return (
                f"You have a learning mission waiting: *{mission.topic}*\n\n"
                "Reply *start* when you're ready to begin!",
                None,
            )
        await set_user_state(db, user, "creating_mission")
        return (
            "Welcome to LearnaDo!\n\n"
            "Who do you want to teach, and what topic?\n\n"
            "Reply in this format:\n"
            "phone: +91XXXXXXXXXX\n"
            "topic: UPI safety",
            None,
        )

    # ── GOAL-SETTER: creating mission ────────────────────────────────
    if state == "creating_mission":
        return await handle_creating_mission(db, user, body)

    # ── GOAL-SETTER: waiting for outline approval ────────────────────
    if state.startswith("confirming_outline:"):
        mission_id = state.split(":", 1)[1]
        return await handle_confirming_outline(db, user, body, mission_id)

    # ── GOAL-SETTER: monitoring active mission ───────────────────────
    if state == "monitoring":
        return (
            "Your learner is still working through the lessons. "
            "I'll message you when they complete one!",
            None,
        )

    # ── LEARNER: mission notified → deliver first lesson ─────────────
    if state == "mission_notified":
        if body_clean in ("start", "yes", "ready", "ok", "begin"):
            mission = await get_active_mission_as_learner(db, user.id)
            if not mission:
                await set_user_state(db, user, "idle")
                return ("Couldn't find your mission. Ask your mentor to resend it.", None)
            return await deliver_lesson(db, user, mission)
        return ("Reply *start* when you're ready to begin your lesson!", None)

    # ── LEARNER: mid-lesson → evaluate response ───────────────────────
    if state == "in_lesson":
        mission = await get_active_mission_as_learner(db, user.id)
        if not mission:
            await set_user_state(db, user, "idle")
            return ("No active mission found. Ask your mentor to set one up.", None)
        return await evaluate_response(db, user, mission, body, media_url, media_type)

    # ── FALLBACK ─────────────────────────────────────────────────────
    return (
        "I didn't quite understand that. Type *help* to restart.",
        None,
    )


# ── Goal-setter handlers ──────────────────────────────────────────────────────

async def handle_creating_mission(
    db: AsyncSession, user: User, body: str
) -> tuple[str, str | None]:
    lines = body.strip().split("\n")
    phone, topic = None, None
    for line in lines:
        lower = line.lower()
        if lower.startswith("phone:"):
            phone = line.split(":", 1)[1].strip()
        elif lower.startswith("topic:"):
            topic = line.split(":", 1)[1].strip()

    if not phone or not topic:
        return (
            "Please use this exact format:\n\n"
            "phone: +91XXXXXXXXXX\n"
            "topic: what you want them to learn\n\n"
            "For example:\n"
            "phone: +919876543210\n"
            "topic: UPI safety for elderly users",
            None,
        )

    phone = phone.replace(" ", "").replace("-", "")
    if not phone.startswith("+") or len(phone) < 10:
        return (
            f"That phone number doesn't look right: *{phone}*\n"
            "Make sure to include the country code, e.g. +919876543210",
            None,
        )

    await send_message(
        user.phone_number,
        f"Generating a lesson plan for *{topic}*... give me a moment!",
    )

    try:
        outline = await get_outline(topic)
    except Exception:
        await set_user_state(db, user, "idle")
        return (
            "Sorry — I couldn't generate the outline right now (Gemini quota/limits).\n\n"
            "Try again in a bit, or double-check your GEMINI_API_KEY / billing.",
            None,
        )

    mission = await create_mission_with_outline(db, user, phone, topic, outline)
    outline_text = "\n".join(
        f"{i+1}. *{item.get('title', '')}*"
        for i, item in enumerate(outline)
    )

    await set_user_state(db, user, f"confirming_outline:{mission.id}")
    return (
        f"Here's the lesson plan for *{topic}*:\n\n"
        f"{outline_text}\n\n"
        f"This will be sent to {phone}.\n\n"
        "Reply *yes* to send it, or *no* to cancel.",
        None,
    )


async def handle_confirming_outline(
    db: AsyncSession, user: User, body: str, mission_id: str
) -> tuple[str, str | None]:
    mission = await get_mission_by_id(db, mission_id)
    if not mission:
        await set_user_state(db, user, "idle")
        return ("Something went wrong — couldn't find that mission. Start over?", None)

    body_lower = body.strip().lower()
    if body_lower in ("yes", "y", "ok", "sure", "send it"):
        result = await db.execute(select(User).where(User.id == mission.learner_id))
        learner = result.scalar_one_or_none()
        if not learner:
            await set_user_state(db, user, "idle")
            return ("Learner not found. Start over?", None)

        await activate_mission(db, mission, learner)
        await set_user_state(db, user, "monitoring")

        outline_len = len(mission.outline_json) if mission.outline_json else 0
        await send_message(
            learner.phone_number,
            f"Hi! *{user.phone_number}* has set up a learning mission for you:\n\n"
            f"*{mission.topic}*\n\n"
            f"There are {outline_len} lessons waiting for you.\n\n"
            "Reply *start* when you're ready to begin!",
        )
        return (
            f"Mission sent to {learner.phone_number}!\n\n"
            "I'll message you when they complete each lesson.",
            None,
        )

    if body_lower in ("no", "n", "cancel", "nope"):
        await cancel_mission(db, mission)
        await set_user_state(db, user, "idle")
        return (
            "Mission cancelled. Send another message anytime to create a new one.",
            None,
        )

    return ("Reply *yes* to send this lesson plan, or *no* to cancel.", None)


# ── Learner handlers ──────────────────────────────────────────────────────────

async def deliver_lesson(
    db: AsyncSession, user: User, mission
) -> tuple[str, str | None]:
    """Fetch or generate lesson content and send it to the learner."""
    lesson = await get_current_lesson(db, mission.id)
    if not lesson:
        return await handle_mission_complete(db, user, mission)

    if not lesson.content_md:
        await send_message(
            user.phone_number,
            f"Loading lesson *{lesson.title}*... one moment!",
        )
        try:
            content = await get_lesson_content(mission.topic, lesson.title, "")
            lesson.content_md = content
            await db.commit()
        except Exception as e:
            return (
                f"Sorry, couldn't load that lesson right now. Try again? ({e})",
                None,
            )

    await create_or_get_progress(db, user.id, mission.id, lesson.id)
    await set_user_state(db, user, "in_lesson")

    summary = await get_mission_progress_summary(db, mission.id)
    progress_line = f"_Lesson {summary['completed'] + 1} of {summary['total']}_\n\n"

    return (
        f"{progress_line}"
        f"*{lesson.title}*\n\n"
        f"{lesson.content_md}\n\n"
        "---\n"
        "Reply with what you understood from this lesson.",
        None,
    )


async def evaluate_response(
    db: AsyncSession,
    user: User,
    mission,
    body: str,
    media_url: str | None,
    media_type: str | None,
) -> tuple[str, str | None]:
    """Score the learner's reply and advance or retry."""
    lesson = await get_current_lesson(db, mission.id)
    if not lesson:
        return await handle_mission_complete(db, user, mission)

    progress = await get_current_progress(db, user.id, mission.id)
    if not progress:
        return await deliver_lesson(db, user, mission)

    confusion = await score_confusion(lesson.content_md or "", body)
    await record_attempt(db, progress, confusion)

    # Force-advance after MAX_ATTEMPTS regardless of score
    if progress.attempts >= MAX_ATTEMPTS:
        await complete_lesson(db, progress, lesson)
        await _notify_goal_setter(db, mission, lesson, user, forced=True)
        next_lesson = await get_next_lesson(db, mission.id, lesson.order_index)
        if not next_lesson:
            return await handle_mission_complete(db, user, mission)
        return await deliver_lesson(db, user, mission)

    if confusion > CONFUSION_THRESHOLD:
        simplified = await simplify_lesson(lesson.content_md or "")
        attempts_left = MAX_ATTEMPTS - progress.attempts
        return (
            f"Let me explain that differently!\n\n"
            f"*{lesson.title}*\n\n"
            f"{simplified}\n\n"
            "---\n"
            f"Have another go — what did you understand? "
            f"_({attempts_left} attempt{'s' if attempts_left != 1 else ''} left)_",
            None,
        )

    # Understood — advance
    await complete_lesson(db, progress, lesson)
    await _notify_goal_setter(db, mission, lesson, user)

    next_lesson = await get_next_lesson(db, mission.id, lesson.order_index)
    if not next_lesson:
        return await handle_mission_complete(db, user, mission)

    # Deliver next lesson immediately
    great_job = "Great job! You understood that well.\n\n"
    next_reply, media = await deliver_lesson(db, user, mission)
    return (great_job + next_reply, media)


async def handle_mission_complete(
    db: AsyncSession, user: User, mission
) -> tuple[str, str | None]:
    from datetime import datetime, timezone

    mission.status = "completed"
    mission.completed_at = datetime.now(timezone.utc)
    await set_user_state(db, user, "idle")
    await db.commit()

    result = await db.execute(select(User).where(User.id == mission.goal_setter_id))
    goal_setter = result.scalar_one_or_none()
    if goal_setter:
        await send_message(
            goal_setter.phone_number,
            f"Your learner has completed the full *{mission.topic}* course!\n\n"
            "All lessons finished. Well done to both of you!",
        )
        await set_user_state(db, goal_setter, "idle")

    return (
        f"Congratulations! You've completed all lessons on *{mission.topic}*!\n\n"
        "You can start a new learning mission anytime by messaging again.",
        None,
    )


async def _notify_goal_setter(
    db: AsyncSession, mission, lesson, learner: User, forced: bool = False
) -> None:
    result = await db.execute(select(User).where(User.id == mission.goal_setter_id))
    goal_setter = result.scalar_one_or_none()
    if not goal_setter:
        return

    summary = await get_mission_progress_summary(db, mission.id)
    forced_note = " _(moved on after max attempts)_" if forced else ""

    await send_message(
        goal_setter.phone_number,
        f"*{mission.topic}* update:\n\n"
        f"Your learner completed: *{lesson.title}*{forced_note}\n\n"
        f"Progress: {summary['completed']}/{summary['total']} lessons done.",
    )
