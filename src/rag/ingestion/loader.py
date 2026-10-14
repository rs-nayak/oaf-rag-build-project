"""Document loading from disk: PDF, plain text, and Markdown.

Workshop 2 deliverable. Students implement :meth:`DocumentLoader.load` and
the per-format helpers.
"""

# ruff: noqa: F401  -- imports become used once Workshop 2 is implemented.

from __future__ import annotations

from pathlib import Path
from pypdf import PdfReader # added for PDF reading

from rag.ingestion.models import Document
from rag.utils.logging import get_logger

log = get_logger(__name__)

_TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst"}
_PDF_SUFFIXES = {".pdf"}
_SUPPORTED_SUFFIXES = _TEXT_SUFFIXES | _PDF_SUFFIXES


class DocumentLoader:
    """Loads documents from a single file or recursively from a directory."""

    def load(self, path: Path) -> list[Document]:
        """Load documents from ``path`` (file or directory).

        - If ``path`` is a single file, returns a list with one Document.
        - If ``path`` is a directory, recursively loads every supported file.
        - Unsupported extensions are skipped with a warning (not an error).

        Args:
            path: File or directory path.

        Returns:
            List of Documents (possibly empty if no supported files found).

        Raises:
            FileNotFoundError: If ``path`` does not exist.
        """
        # TODO Workshop 2:
        # 1. Coerce ``path`` to ``pathlib.Path`` and raise ``FileNotFoundError`` if it
        #    does not exist.
        # 2. If it's a single file, dispatch to ``_load_one`` and return the result
        #    wrapped in a list (or empty list if loading fails).
        # 3. If it's a directory, walk it recursively (``path.rglob("*")``) sorted for
        #    determinism. For every file whose extension is in ``_SUPPORTED_SUFFIXES``,
        #    call ``_load_one`` and collect the results.
        # 4. Return the collected list of Documents.
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

        documents = []

        if path.is_file():
            doc = self._load_one(path)
            if doc is not None:
                documents.append(doc)

        elif path.is_dir():
            for file in sorted(path.rglob("*")):
                if file.is_file() and file.suffix.lower() in _SUPPORTED_SUFFIXES:
                    doc = self._load_one(file)
                    if doc is not None:
                        documents.append(doc)

        return documents

        
    def _load_one(self, path: Path) -> Document | None:
        """Dispatch to the right format-specific loader. Returns None on failure."""
        # TODO Workshop 2:
        # - Look at ``path.suffix.lower()``.
        # - If it's a PDF suffix, call ``_load_pdf``.
        # - If it's a text/markdown suffix, call ``_load_text``.
        # - Anything else: log a warning and return None.
        # - Wrap the dispatch in try/except so a single broken file doesn't sink the
        #   whole batch (log the error, return None).
        try:
            suffix = path.suffix.lower()

            if suffix in _PDF_SUFFIXES:
                return self._load_pdf(path)

            elif suffix in _TEXT_SUFFIXES:
                return self._load_text(path)

            else:
                log.warning("Unsupported file type: %s", path)
                return None

        except Exception as e:
            log.error("Failed to load %s: %s", path, e)
            return None

    @staticmethod
    def _load_text(path: Path) -> Document:
        """Load a plain text or Markdown file as a single Document."""
        # TODO Workshop 2:
        # - Read the file as UTF-8 (use ``errors="replace"`` so weird bytes don't crash).
        # - Build a ``Document`` whose ``source`` is ``str(path)`` and whose metadata
        #   includes the file extension and size in bytes.
        text = path.read_text(
            encoding="utf-8",
            errors="replace"
        )

        return Document(
            text=text,
            source=str(path),
            metadata={
                "extension": path.suffix.lower(),
                "size_bytes": path.stat().st_size
            }
        )

    @staticmethod
    def _load_pdf(path: Path) -> Document:
        """Load a PDF as a single Document, joining pages with form-feeds."""
        # TODO Workshop 2:
        # - Use ``pypdf.PdfReader`` to open the file.
        # - Extract text from each page; join pages with "\n\f\n".
        # - Build a Document whose metadata records ``page_count`` and ``size_bytes``.
        reader = PdfReader(path)

        pages = [
            page.extract_text() or ""
            for page in reader.pages
        ]

        text = "\n\f\n".join(pages)

        return Document(
            text=text,
            source=str(path),
            metadata={
                "page_count": len(reader.pages),
                "size_bytes": path.stat().st_size
            }
        )