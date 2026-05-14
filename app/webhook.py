"""
Meta WhatsApp webhook — receives incoming messages, routes, and replies via API.
"""

import logging

from fastapi import APIRouter, Depends, Request, Response, Query
from fastapi.responses import PlainTextResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.orchestrator import analyse
from app.router import route_message
from app.services import get_or_create_user
from app.whatsapp import send_message
from app.config import settings

logger = logging.getLogger(__name__)
webhook_router = APIRouter()


@webhook_router.get("/webhook")
async def verify_webhook(
    hub_mode: str = Query(default=None, alias="hub.mode"),
    hub_challenge: str = Query(default=None, alias="hub.challenge"),
    hub_verify_token: str = Query(default=None, alias="hub.verify_token")
):
    """Verify webhook for Meta Cloud API."""
    if hub_verify_token == settings.meta_verify_token:
        logger.info("Meta webhook verified successfully!")
        return PlainTextResponse(content=hub_challenge)
    return PlainTextResponse(content="Forbidden", status_code=403)


@webhook_router.post("/webhook")
async def whatsapp_webhook(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Handle incoming WhatsApp messages from Meta."""
    data = await request.json()
    
    try:
        entry = data.get("entry", [])[0]
        changes = entry.get("changes", [])[0]
        value = changes.get("value", {})
        
        # We only care about messages, ignore statuses
        if "messages" not in value:
            return {"status": "ok"}
            
        message_info = value["messages"][0]
        phone = message_info.get("from", "").strip()
        
        # Extract body based on message type
        msg_type = message_info.get("type", "text")
        body = ""
        media_url = None
        media_content_type = None
        
        if msg_type == "text":
            body = message_info.get("text", {}).get("body", "")
        else:
            body = f"[{msg_type} message received]"
            
        logger.info("WhatsApp webhook: from=%s body=%r", phone, (body or "")[:80])

        user = await get_or_create_user(db, phone)

        # Orchestrate: single LLM call → intent + sentiment + enriched_context
        result = await analyse(body, user.wa_session_state or "idle")
        logger.info(
            "Orchestrator: intent=%s sentiment=%s", result.intent, result.sentiment
        )

        reply_text, reply_media = await route_message(
            db, user, body, media_url, media_content_type, result
        )
        logger.info("Sending reply to %s: %r", phone, (reply_text or "")[:80])

        try:
            await send_message(phone, reply_text, reply_media)
        except Exception as e:
            logger.exception("Meta send_message failed: %s", e)
            return {"status": "ok", "error": str(e)}

    except Exception as e:
        logger.warning("Error parsing Meta webhook payload: %s", e)

    return {"status": "ok"}

