"""FastAPI application entrypoint — Workshop 6.

Run locally::

    uv run uvicorn rag.api.main:app --reload

Open http://127.0.0.1:8000/docs for the auto-generated Swagger UI.
"""

# ruff: noqa: F401  -- imports become used once Workshop 6 is implemented.

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag.api.routes import router
from rag.generation.pipeline import RAGPipeline
from rag.providers.factory import get_embeddings, get_llm
from rag.retrieval.retriever import Retriever
from rag.utils.logging import get_logger
from rag.vectorstore.chroma_store import ChromaStore

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    embeddings = get_embeddings()
    llm = get_llm()

    store = ChromaStore(embeddings=embeddings)
    retriever = Retriever(store=store)
    pipeline = RAGPipeline(retriever=retriever, llm=llm)

    app.state.store = store
    app.state.retriever = retriever
    app.state.pipeline = pipeline

    yield

    log.info("Application shutting down.")

def create_app() -> FastAPI:
    app = FastAPI(
        title="Skincare RAG API",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    return app


try:
    app: FastAPI | None = create_app()
    """Module-level app instance used by ``uvicorn rag.api.main:app``."""
except NotImplementedError:
    # Starter: ``create_app`` is a Workshop 6 stub. Importing this module
    # should not crash before Workshop 6 (e.g. during test collection).
    app = None
