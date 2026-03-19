"""
User lookup, session state, and mission lifecycle for the WhatsApp flow.
"""

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Lesson, Mission, User, UserProgress


async def get_or_create_user(db: AsyncSession, phone: str) -> User:
    result = await db.execute(select(User).where(User.phone_number == phone))
    user = result.scalar_one_or_none()
    if not user:
        user = User(phone_number=phone, wa_session_state="idle")
        db.add(user)
        await db.commit()
        await db.refresh(user)
    return user


async def get_active_mission_as_learner(db: AsyncSession, user_id: uuid.UUID) -> Mission | None:
    result = await db.execute(
        select(Mission).where(
            Mission.learner_id == user_id,
            Mission.status == "active",
        )
    )
    return result.scalar_one_or_none()


async def get_current_lesson(db: AsyncSession, mission_id: uuid.UUID) -> Lesson | None:
    """Get the lowest order_index lesson that isn't completed yet."""
    result = await db.execute(
        select(Lesson)
        .where(Lesson.mission_id == mission_id, Lesson.status != "completed")
        .order_by(Lesson.order_index)
        .limit(1)
    )
    return result.scalar_one_or_none()


async def set_user_state(db: AsyncSession, user: User, state: str) -> None:
    user.wa_session_state = state
    await db.commit()


async def create_mission_with_outline(
    db: AsyncSession,
    goal_setter: User,
    learner_phone: str,
    topic: str,
    outline: list[dict],
) -> Mission:
    """Create mission + lesson rows from the generated outline."""
    result = await db.execute(select(User).where(User.phone_number == learner_phone))
    learner = result.scalar_one_or_none()
    if not learner:
        learner = User(phone_number=learner_phone, wa_session_state="idle")
        db.add(learner)
        await db.flush()

    mission = Mission(
        goal_setter_id=goal_setter.id,
        learner_id=learner.id,
        topic=topic,
        status="pending_approval",
        outline_json=outline,
    )
    db.add(mission)
    await db.flush()

    for i, item in enumerate(outline):
        lesson = Lesson(
            mission_id=mission.id,
            order_index=i,
            title=item.get("title", f"Lesson {i + 1}"),
            content_md="",
            status="pending",
        )
        db.add(lesson)

    await db.commit()
    await db.refresh(mission)
    return mission


async def get_mission_by_id(db: AsyncSession, mission_id: str | uuid.UUID) -> Mission | None:
    if isinstance(mission_id, str):
        try:
            mission_id = uuid.UUID(mission_id)
        except ValueError:
            return None
    return await db.get(Mission, mission_id)


async def activate_mission(db: AsyncSession, mission: Mission, learner: User) -> None:
    mission.status = "active"
    learner.wa_session_state = "mission_notified"
    await db.commit()


async def cancel_mission(db: AsyncSession, mission: Mission) -> None:
    mission.status = "cancelled"
    await db.commit()


# ── Phase 4: progress tracking ────────────────────────────────────────────────

async def get_current_progress(
    db: AsyncSession, user_id: uuid.UUID, mission_id: uuid.UUID
) -> UserProgress | None:
    result = await db.execute(
        select(UserProgress).where(
            UserProgress.user_id == user_id,
            UserProgress.mission_id == mission_id,
            UserProgress.status != "completed",
        )
    )
    return result.scalar_one_or_none()


async def create_or_get_progress(
    db: AsyncSession,
    user_id: uuid.UUID,
    mission_id: uuid.UUID,
    lesson_id: uuid.UUID,
) -> UserProgress:
    progress = await get_current_progress(db, user_id, mission_id)
    if not progress:
        progress = UserProgress(
            user_id=user_id,
            mission_id=mission_id,
            lesson_id=lesson_id,
            status="in_progress",
            confusion_score=0.0,
            attempts=0,
        )
        db.add(progress)
        await db.commit()
        await db.refresh(progress)
    return progress


async def record_attempt(
    db: AsyncSession,
    progress: UserProgress,
    confusion_score: float,
) -> None:
    from datetime import datetime, timezone

    progress.attempts += 1
    progress.confusion_score = confusion_score
    progress.last_attempted_at = datetime.now(timezone.utc)
    await db.commit()


async def complete_lesson(
    db: AsyncSession, progress: UserProgress, lesson: Lesson
) -> None:
    progress.status = "completed"
    lesson.status = "completed"
    await db.commit()


async def get_next_lesson(
    db: AsyncSession, mission_id: uuid.UUID, current_order_index: int
) -> Lesson | None:
    result = await db.execute(
        select(Lesson)
        .where(
            Lesson.mission_id == mission_id,
            Lesson.order_index > current_order_index,
            Lesson.status != "completed",
        )
        .order_by(Lesson.order_index)
        .limit(1)
    )
    return result.scalar_one_or_none()


async def get_mission_progress_summary(
    db: AsyncSession, mission_id: uuid.UUID
) -> dict:
    result = await db.execute(select(Lesson).where(Lesson.mission_id == mission_id))
    lessons = result.scalars().all()
    completed = [l for l in lessons if l.status == "completed"]
    return {
        "total": len(lessons),
        "completed": len(completed),
        "remaining": len(lessons) - len(completed),
    }
