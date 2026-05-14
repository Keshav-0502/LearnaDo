"""
Shared tool functions available to all graph nodes and agents.

- search_web: Tavily search with images and source metadata
- get_youtube_context: Find relevant YouTube videos + fetch transcripts
- fetch_transcript: youtube-transcript-api wrapper
"""

import asyncio
import logging
import os
import re

logger = logging.getLogger(__name__)


def _get_tavily_key() -> str:
    try:
        from app.config import settings

        if settings.tavily_api_key:
            return settings.tavily_api_key
    except Exception:
        pass
    return os.getenv("TAVILY_API_KEY") or ""


# ── Web Search ─────────────────────────────────────────────────────────────────


def _search_web_sync(
    query: str,
    max_results: int = 5,
    include_images: bool = True,
    include_domains: list[str] | None = None,
) -> dict:
    """Synchronous Tavily search returning results + images."""
    from tavily import TavilyClient

    client = TavilyClient(_get_tavily_key())

    kwargs: dict = {
        "query": query,
        "search_depth": "advanced",
        "max_results": max_results,
        "include_images": include_images,
        "include_image_descriptions": include_images,
    }
    if include_domains:
        kwargs["include_domains"] = include_domains

    try:
        response = client.search(**kwargs)
    except Exception as exc:
        logger.warning("Tavily search failed for %r: %s", query, exc)
        return {"results": [], "images": []}

    return {
        "results": response.get("results", []),
        "images": response.get("images", []),
    }


async def search_web(
    query: str,
    max_results: int = 5,
    include_images: bool = True,
    include_domains: list[str] | None = None,
) -> dict:
    """Async Tavily search.

    Returns::

        {
            "results": [{"title", "url", "content", "score", "images": [...]}],
            "images":  [{"url", "description"} | str, ...],
        }
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, _search_web_sync, query, max_results, include_images, include_domains
    )


# ── YouTube Transcript ─────────────────────────────────────────────────────────


_YT_VIDEO_ID_RE = re.compile(
    r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})"
)


def _extract_video_id(url: str) -> str | None:
    m = _YT_VIDEO_ID_RE.search(url)
    return m.group(1) if m else None


MAX_VIDEO_DURATION_SECS = 600  # 10 minutes — microlearning limit


def _fetch_transcript_sync(
    video_id: str, max_words: int = 500
) -> tuple[str, float]:
    """Fetch transcript for a YouTube video.

    Returns (first *max_words* words of text, estimated duration in seconds).
    Duration is derived from the last transcript segment's start time.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        api = YouTubeTranscriptApi()
        transcript = api.fetch(video_id)
        snippets = list(transcript)
        full_text = " ".join(s.text for s in snippets)
        words = full_text.split()

        duration_secs = 0.0
        if snippets:
            last = snippets[-1]
            duration_secs = getattr(last, "start", 0.0) + getattr(last, "duration", 0.0)

        return " ".join(words[:max_words]), duration_secs
    except Exception as exc:
        logger.debug("Transcript fetch failed for %s: %s", video_id, exc)
        return "", 0.0


async def fetch_transcript(video_id: str, max_words: int = 500) -> tuple[str, float]:
    """Async wrapper around youtube-transcript-api."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _fetch_transcript_sync, video_id, max_words)


def _get_youtube_context_sync(query: str, max_videos: int = 2) -> list[dict]:
    """Search YouTube via Tavily, fetch transcripts, and filter out long videos."""
    search = _search_web_sync(
        query, max_results=max_videos + 2, include_images=False, include_domains=["youtube.com"]
    )

    videos: list[dict] = []
    for result in search.get("results", []):
        url = result.get("url", "")
        vid = _extract_video_id(url)
        if not vid:
            continue

        transcript_text, duration_secs = _fetch_transcript_sync(vid, max_words=500)

        if duration_secs > MAX_VIDEO_DURATION_SECS:
            logger.debug(
                "Skipping video %s (%.0fs > %ds limit)", vid, duration_secs, MAX_VIDEO_DURATION_SECS
            )
            continue

        videos.append(
            {
                "url": url,
                "title": result.get("title", ""),
                "video_id": vid,
                "transcript_snippet": transcript_text,
                "duration_secs": duration_secs,
            }
        )

        if len(videos) >= max_videos:
            break

    return videos


async def get_youtube_context(query: str, max_videos: int = 2) -> list[dict]:
    """Async: search YouTube + fetch transcripts.

    Returns::

        [{"url", "title", "video_id", "transcript_snippet"}, ...]
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _get_youtube_context_sync, query, max_videos)
