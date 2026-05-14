"""
FastAPI entrypoint for the LearnADo application.
Sets up AsyncPostgresSaver for LangGraph checkpointing at startup.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg_pool import AsyncConnectionPool

    from app.config import settings
    from app.graph import build_graph

    conninfo = (
        f"postgresql://{settings.postgres_user}:{settings.postgres_password}"
        f"@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
    )

    async with AsyncConnectionPool(
        conninfo=conninfo,
        max_size=10,
        kwargs={"autocommit": True, "prepare_threshold": 0},
    ) as pool:
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        app.state.graph = build_graph(checkpointer)
        logging.getLogger(__name__).info("LangGraph compiled with Postgres checkpointer")
        yield


from app.routes import router as app_router
from app.webhook import webhook_router

app = FastAPI(
    title="LearnADo",
    description="AI-powered learning assistant for document processing and analysis",
    version="1.0.0",
    redirect_slashes=False,
    lifespan=lifespan,
)


@app.get("/")
async def root():
    """Redirect to API documentation"""
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "LearnADo"}


app.include_router(app_router, prefix="/api")
app.include_router(webhook_router)
