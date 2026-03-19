"""
FastAPI entrypoint for the LearnADo application.
Handles file uploads, preprocessing, and AI-powered learning assistance.
"""

import logging

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(name)s: %(message)s",
)

from app.routes import router as app_router
from app.webhook import webhook_router

app = FastAPI(
    title="LearnADo",
    description="AI-powered learning assistant for document processing and analysis",
    version="1.0.0"
)

@app.get("/")
async def root():
    """Redirect to API documentation"""
    return RedirectResponse(url="/docs")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "LearnADo"}

# API routes under /api
app.include_router(app_router, prefix="/api")
# WhatsApp webhook at /webhook/whatsapp (no prefix; Twilio calls this URL)
app.include_router(webhook_router)