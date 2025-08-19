"""
FastAPI entrypoint for the LearnADo application.
Handles file uploads, preprocessing, and AI-powered learning assistance.
"""

from fastapi import FastAPI
from app.routes import router as app_router

app = FastAPI(title="LearnADo")
# Attach routes under /api
app.include_router(app_router, prefix="/api")