"""
Meta WhatsApp webhook — invokes the LangGraph agent for every inbound message.

On each POST the webhook checks whether the user's graph thread has a pending
interrupt (outline review, waiting for start, lesson response).  If so it
resumes; otherwise it starts a fresh graph run from classify_intent.
"""

import asyncio
import logging
from collections import defaultdict

from fastapi import APIRouter, Query, Request
from fastapi.responses import PlainTextResponse
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from app.config import settings

logger = logging.getLogger(__name__)
webhook_router = APIRouter()

_phone_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
_seen_wamids: set[str] = set()
_MAX_SEEN = 2000


@webhook_router.get("/webhook")
async def verify_webhook(
    hub_mode: str = Query(default=None, alias="hub.mode"),
    hub_challenge: str = Query(default=None, alias="hub.challenge"),
    hub_verify_token: str = Query(default=None, alias="hub.verify_token"),
):
    """Verify webhook for Meta Cloud API."""
    if hub_verify_token == settings.meta_verify_token:
        logger.info("Meta webhook verified successfully!")
        return PlainTextResponse(content=hub_challenge)
    return PlainTextResponse(content="Forbidden", status_code=403)


@webhook_router.post("/webhook")
async def whatsapp_webhook(request: Request):
    """Handle incoming WhatsApp messages via the LangGraph agent."""
    data = await request.json()

    try:
        entry = data.get("entry", [])[0]
        changes = entry.get("changes", [])[0]
        value = changes.get("value", {})

        if "messages" not in value:
            return {"status": "ok"}

        message_info = value["messages"][0]
        wamid = message_info.get("id", "")
        phone = message_info.get("from", "").strip()
        if phone and not phone.startswith("+"):
            phone = "+" + phone

        # Deduplicate — WhatsApp may retry or user may double-tap
        if wamid and wamid in _seen_wamids:
            logger.info("Duplicate wamid %s from %s — skipping", wamid, phone)
            return {"status": "ok"}
        if wamid:
            _seen_wamids.add(wamid)
            if len(_seen_wamids) > _MAX_SEEN:
                to_remove = list(_seen_wamids)[:_MAX_SEEN // 2]
                for r in to_remove:
                    _seen_wamids.discard(r)

        msg_type = message_info.get("type", "text")
        if msg_type == "text":
            body = message_info.get("text", {}).get("body", "")
        else:
            body = f"[{msg_type} message received]"

        logger.info("WhatsApp webhook: from=%s body=%r", phone, (body or "")[:80])

        # Persist incoming message to DB
        try:
            from app.database import AsyncSessionLocal
            from app import services

            async with AsyncSessionLocal() as db:
                await services.save_message(db, phone, "user", body)
        except Exception:
            logger.debug("Failed to save incoming message to DB", exc_info=True)

        # Serialize graph runs per phone number
        async with _phone_locks[phone]:
            graph = request.app.state.graph
            config = {"configurable": {"thread_id": phone}}

            state_snapshot = await graph.aget_state(config)
            has_interrupt = bool(state_snapshot.tasks) and any(
                t.interrupts for t in state_snapshot.tasks
            )

            if has_interrupt:
                logger.info("Resuming interrupt for %s", phone)
                await graph.ainvoke(
                    Command(resume={"user_message": body}),
                    config=config,
                )
            else:
                logger.info("New graph run for %s", phone)
                await graph.ainvoke(
                    {
                        "messages": [HumanMessage(content=body)],
                        "phone_number": phone,
                    },
                    config=config,
                )

    except Exception as e:
        logger.exception("Error in webhook: %s", e)

    return {"status": "ok"}
