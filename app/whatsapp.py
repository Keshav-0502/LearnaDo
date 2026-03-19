"""
Twilio WhatsApp send logic; kept separate from webhook routing.
"""

import asyncio
from typing import Any

from twilio.rest import Client

from app.config import settings

_client: Client | None = None


def get_twilio_client() -> Client:
    global _client
    if not _client:
        _client = Client(
            settings.twilio_account_sid,
            settings.twilio_auth_token,
        )
    return _client


async def send_message(
    to_phone: str,
    body: str,
    media_url: str | None = None,
) -> str:
    """Send a WhatsApp message. to_phone should be plain e.g. +917042881303."""
    # Twilio WhatsApp has a strict body limit (~1600 chars). Split proactively.
    # Keep chunks a bit smaller to avoid edge cases with encoding/concat.
    max_len = 1500
    chunks = [body[i : i + max_len] for i in range(0, len(body), max_len)] or [""]

    last_sid = ""
    for i, chunk in enumerate(chunks):
        kwargs: dict[str, Any] = {
            "from_": settings.twilio_whatsapp_from,
            "to": f"whatsapp:{to_phone}",
            "body": chunk,
        }
        # Only attach media to the first chunk
        if media_url and i == 0:
            kwargs["media_url"] = [media_url]

        def _create() -> Any:
            return get_twilio_client().messages.create(**kwargs)

        msg = await asyncio.to_thread(_create)
        last_sid = msg.sid

    return last_sid
