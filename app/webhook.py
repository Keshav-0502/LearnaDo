"""
Twilio WhatsApp webhook — receives incoming messages, routes, and replies via API.
"""

import logging

from fastapi import APIRouter, Depends, Form
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.router import route_message
from app.services import get_or_create_user
from app.whatsapp import send_message

logger = logging.getLogger(__name__)
webhook_router = APIRouter()


@webhook_router.post("/webhook/whatsapp")
async def whatsapp_webhook(
    From: str = Form(...),
    Body: str = Form(default=""),
    MediaUrl0: str | None = Form(default=None),
    MediaContentType0: str | None = Form(default=None),
    db: AsyncSession = Depends(get_db),
):
    phone = From.replace("whatsapp:", "").strip()
    logger.info("WhatsApp webhook: from=%s body=%r", phone, (Body or "")[:80])

    user = await get_or_create_user(db, phone)
    reply_text, reply_media = await route_message(
        db, user, Body, MediaUrl0, MediaContentType0
    )
    logger.info("Sending reply to %s: %r", phone, (reply_text or "")[:80])

    try:
        await send_message(phone, reply_text, reply_media)
    except Exception as e:
        logger.exception("Twilio send_message failed: %s", e)
        # Still return 200 so Twilio doesn't retry; fix config and try again
        return {"status": "ok", "error": str(e)}

    return {"status": "ok"}
