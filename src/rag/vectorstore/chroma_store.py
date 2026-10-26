"""ChromaDB-backed vector store — Workshop 3.

Wraps the file-backed ``PersistentClient``. Embeddings are computed by an
injected :class:`EmbeddingsProvider`, *not* by Chroma's built-in embedder —
this keeps the abstraction visible and lets students swap providers freely.
"""

# ruff: noqa: F401  -- imports become used once Workshop 3 is implemented.

from __future__ import annotations

from pathlib import Path
from typing import Any

from rag.config import settings
from rag.ingestion.models import Chunk
from rag.providers.base import EmbeddingsProvider
from rag.utils.logging import get_logger

import chromadb

log = get_logger(__name__)


def _scalarize(value: Any) -> str | int | float | bool:
    """Coerce a metadata value into a Chroma-allowed scalar."""
    if isinstance(value, str | int | float | bool):
        return value
    return str(value)


class ChromaStore:
    """File-backed ChromaDB collection for chunk embeddings."""

    def __init__(
        self,
        embeddings: EmbeddingsProvider,
        persist_dir: Path | None = None,
        collection_name: str | None = None,
    ) -> None:
        """Construct a ChromaStore and ensure the collection exists.

        Args:
            embeddings: Provider used for both ``add`` and ``query``. Must produce
                vectors of consistent dimension across calls.
            persist_dir: Filesystem directory for the persistent store.
                Defaults to ``settings.chroma_persist_dir``.
            collection_name: Collection name within the store.
                Defaults to ``settings.chroma_collection``.
        """
        # TODO Workshop 3:
        # 1. Store the embeddings provider on ``self``.
        # 2. Resolve ``persist_dir`` (default: ``settings.chroma_persist_dir``).
        #    Make sure the directory exists (``mkdir(parents=True, exist_ok=True)``).
        # 3. Resolve ``collection_name`` (default: ``settings.chroma_collection``).
        # 4. Build a ``chromadb.PersistentClient(path=str(self.persist_dir))``.
        # 5. Get-or-create the collection with ``metadata={"hnsw:space": "cosine"}``.
        self.embeddings = embeddings
        self.persist_dir = Path(persist_dir or settings.chroma_persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name or settings.chroma_collection

        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: list[Chunk]) -> None:
        """Embed ``chunks`` and upsert them into the collection.

        Idempotent on chunk ``id`` (``"<source>::<chunk_index>"``): re-running
        on the same chunks overwrites prior entries with the same id.
        """
        # TODO Workshop 3:
        # 1. If ``chunks`` is empty, return early.
        # 2. Build parallel lists of ids, documents (chunk text), metadatas
        #    (use ``_scalarize`` so every value is a Chroma-allowed scalar; also
        #    include the chunk's ``source`` in the metadata).
        # 3. Call ``self.embeddings.embed(documents)`` to get vectors.
        # 4. Upsert into the collection: ``collection.upsert(ids=..., documents=...,
        #    embeddings=..., metadatas=...)``.
        if not chunks:
            return
        ids = [c.id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas: list[dict[str, str | int | float | bool]] = [
            {k: _scalarize(v) for k, v in {**c.metadata, "source": c.source}.items()}
            for c in chunks
        ]
        vectors = self.embeddings.embed(documents)
        if len(vectors) != len(chunks):
            raise RuntimeError(
                f"Embeddings provider returned {len(vectors)} vectors for {len(chunks)} chunks."
            )
        self._collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=vectors,
            metadatas=metadatas,
        )
        log.info("Upserted %d chunks into collection %r.", len(chunks), self.collection_name)

    def query(
        self,
        query_text: str,
        k: int = 4,
        where: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        """Return the ``k`` chunks most similar to ``query_text``.

        Args:
            query_text: Natural-language query string.
            k: Number of results to return.
            where: Optional Chroma metadata filter (e.g. ``{"source": "manual.pdf"}``).

        Returns:
            List of matching :class:`Chunk` objects, ordered most-similar first.
        """
        if not query_text or not query_text.strip():
            return []
        query_vec = self.embeddings.embed([query_text])[0]
        result = self._collection.query(
            query_embeddings=[query_vec],
            n_results=k,
            where=where,
        )
        chunks: list[Chunk] = []
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        for chunk_id, document, metadata in zip(ids, documents, metadatas, strict=False):
            meta = dict(metadata or {})
            source = str(meta.pop("source", chunk_id.split("::", 1)[0]))
            chunk_index = int(meta.get("chunk_index", 0))
            chunks.append(
                Chunk(text=document, source=source, chunk_index=chunk_index, metadata=meta)
            )
        return chunks

    def count(self) -> int:
        """Return the number of vectors currently stored."""
        # TODO Workshop 3 (small):
        # - Return ``int(self._collection.count())``.
        return int(self._collection.count())

    def clear(self) -> None:
        """Drop and recreate the collection. Wipes all stored vectors."""
        # TODO Workshop 3 (small):
        # - Call ``self._client.delete_collection(self.collection_name)``.
        # - Re-create the collection with the same cosine-space metadata.
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        log.info("Cleared collection %r.", self.collection_name)