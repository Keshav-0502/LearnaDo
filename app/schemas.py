"""
Pydantic v2 schemas for LearnaDo API request/response validation.
"""

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# ── Shared config ─────────────────────────────────────────────────────────────

class _Base(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# ── User ──────────────────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    phone_number: str = Field(..., max_length=20)
    name: str | None = Field(None, max_length=255)
    preferred_language: str | None = Field("en", max_length=10)
    wa_session_state: str | None = None


class UserUpdate(BaseModel):
    name: str | None = Field(None, max_length=255)
    preferred_language: str | None = Field(None, max_length=10)
    wa_session_state: str | None = None


class UserOut(_Base):
    id: uuid.UUID
    phone_number: str
    name: str | None
    preferred_language: str | None
    wa_session_state: str | None
    created_at: datetime


# ── Mission ───────────────────────────────────────────────────────────────────

class MissionCreate(BaseModel):
    goal_setter_id: uuid.UUID
    learner_id: uuid.UUID
    topic: str = Field(..., max_length=500)
    outline_json: dict[str, Any] | None = None


class MissionUpdate(BaseModel):
    status: str | None = Field(None, max_length=50)
    outline_json: dict[str, Any] | None = None
    completed_at: datetime | None = None


class MissionOut(_Base):
    id: uuid.UUID
    goal_setter_id: uuid.UUID
    learner_id: uuid.UUID
    topic: str
    status: str
    outline_json: dict[str, Any] | None
    created_at: datetime
    completed_at: datetime | None


# ── Lesson ────────────────────────────────────────────────────────────────────

class LessonCreate(BaseModel):
    mission_id: uuid.UUID
    order_index: int = Field(..., ge=0)
    title: str = Field(..., max_length=500)
    content_md: str | None = None
    status: str = Field("draft", max_length=50)


class LessonUpdate(BaseModel):
    title: str | None = Field(None, max_length=500)
    content_md: str | None = None
    status: str | None = Field(None, max_length=50)
    order_index: int | None = Field(None, ge=0)


class LessonOut(_Base):
    id: uuid.UUID
    mission_id: uuid.UUID
    order_index: int
    title: str
    content_md: str | None
    status: str


# ── UserProgress ──────────────────────────────────────────────────────────────

class UserProgressCreate(BaseModel):
    mission_id: uuid.UUID
    lesson_id: uuid.UUID
    user_id: uuid.UUID
    status: str = Field("not_started", max_length=50)
    confusion_score: float | None = Field(None, ge=0.0, le=1.0)
    attempts: int = Field(0, ge=0)


class UserProgressUpdate(BaseModel):
    status: str | None = Field(None, max_length=50)
    confusion_score: float | None = Field(None, ge=0.0, le=1.0)
    attempts: int | None = Field(None, ge=0)
    last_attempted_at: datetime | None = None


class UserProgressOut(_Base):
    id: uuid.UUID
    mission_id: uuid.UUID
    lesson_id: uuid.UUID
    user_id: uuid.UUID
    status: str
    confusion_score: float | None
    attempts: int
    last_attempted_at: datetime | None


# ── Message ───────────────────────────────────────────────────────────────────

class MessageCreate(BaseModel):
    user_id: uuid.UUID
    mission_id: uuid.UUID | None = None
    role: str = Field(..., max_length=20)
    content: str
    media_type: str | None = Field(None, max_length=50)
    wa_message_id: str | None = Field(None, max_length=255)


class MessageOut(_Base):
    id: uuid.UUID
    user_id: uuid.UUID
    mission_id: uuid.UUID | None
    role: str
    content: str
    media_type: str | None
    wa_message_id: str | None
    created_at: datetime


# ── Document ──────────────────────────────────────────────────────────────────

class DocumentCreate(BaseModel):
    mission_id: uuid.UUID | None = None
    uploaded_by: uuid.UUID
    filename: str = Field(..., max_length=500)
    storage_url: str | None = Field(None, max_length=2048)
    vector_namespace: str | None = Field(None, max_length=255)


class DocumentOut(_Base):
    id: uuid.UUID
    mission_id: uuid.UUID | None
    uploaded_by: uuid.UUID
    filename: str
    storage_url: str | None
    vector_namespace: str | None
    uploaded_at: datetime


# ── Generic responses ─────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str


class ErrorResponse(BaseModel):
    detail: str
    code: str | None = None
