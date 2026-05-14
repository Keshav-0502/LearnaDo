"""
Meta WhatsApp send logic; kept separate from webhook routing.
"""

import logging
from typing import Any

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


async def send_message(
    to_phone: str,
    body: str,
    media_url: str | None = None,
) -> str:
    """Send a WhatsApp message via Meta Cloud API."""
    if not settings.whatsapp_access_token or not settings.whatsapp_phone_number_id:
        logger.warning("Meta WhatsApp credentials not configured.")
        return ""

    url = f"https://graph.facebook.com/v19.0/{settings.whatsapp_phone_number_id}/messages"
    headers = {
        "Authorization": f"Bearer {settings.whatsapp_access_token}",
        "Content-Type": "application/json",
    }

    # Meta has a 4096 character limit
    max_len = 4096
    chunks = [body[i : i + max_len] for i in range(0, len(body), max_len)] or [""]

    last_msg_id = ""
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        for i, chunk in enumerate(chunks):
            payload: dict[str, Any] = {
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": to_phone,
            }

            if media_url and i == 0:
                # If we have a media URL, send it as an image with caption
                payload["type"] = "image"
                payload["image"] = {"link": media_url}
                if chunk:
                    payload["image"]["caption"] = chunk
            else:
                payload["type"] = "text"
                payload["text"] = {"preview_url": False, "body": chunk}

            response = await client.post(url, headers=headers, json=payload)
            if response.status_code >= 400:
                logger.error(
                    "WhatsApp API %s for %s: %s",
                    response.status_code, to_phone, response.text,
                )
            response.raise_for_status()
            data = response.json()
            if "messages" in data and len(data["messages"]) > 0:
                last_msg_id = data["messages"][0].get("id", "")

    # Persist outgoing message to DB
    try:
        from app.database import AsyncSessionLocal
        from app import services

        async with AsyncSessionLocal() as db:
            await services.save_message(
                db,
                to_phone,
                "assistant",
                body,
                wa_message_id=last_msg_id,
                media_type="image" if media_url else None,
            )
    except Exception:
        logger.debug("Failed to save outgoing message to DB", exc_info=True)

    return last_msg_id
