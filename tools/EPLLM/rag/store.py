# RAG store: indexes module version docs and optional error->fix examples for retrieval.
# Used by analyzer and error_handler agents to augment context.
import os
from pathlib import Path
from typing import List, Optional

try:
    from langchain_community.vectorstores import DocArrayInMemorySearch
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_core.documents import Document
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False


class RAGStore:
    """In-memory vector store over module version files and optional error examples."""

    def __init__(
        self,
        base_modules_dir: str,
        base_url: str = "http://localhost:11434",
        embedding_model: str = "nomic-embed-text",
    ):
        self.base_modules_dir = Path(base_modules_dir)
        self.base_url = base_url
        self.embedding_model = embedding_model
        self._store: Optional[DocArrayInMemorySearch] = None
        self._docs: List[Document] = []

    def _get_embeddings(self):
        if not RAG_AVAILABLE:
            return None
        return OllamaEmbeddings(
            base_url=self.base_url,
            model=self.embedding_model,
        )

    def build_from_modules_dir(self) -> bool:
        """Index all {module}_{python_version}.txt files under base_modules_dir."""
        if not RAG_AVAILABLE:
            return False
        self._docs = []
        if not self.base_modules_dir.exists():
            return False
        for path in self.base_modules_dir.glob("*_*.txt"):
            name = path.stem
            # e.g. requests_3.8.txt -> module=requests, py=3.8
            if "_" in name:
                parts = name.rsplit("_", 1)
                if len(parts) == 2:
                    module, py_ver = parts
                    try:
                        text = path.read_text()
                    except Exception:
                        text = ""
                    self._docs.append(
                        Document(
                            page_content=f"Module: {module}\nPython: {py_ver}\nVersions: {text[:8000]}",
                            metadata={"module": module, "python_version": py_ver, "source": str(path)},
                        )
                    )
        if not self._docs:
            return False
        emb = self._get_embeddings()
        if emb is None:
            return False
        self._store = DocArrayInMemorySearch.from_documents(self._docs, emb)
        return True

    def add_error_examples(self, examples: List[dict]) -> None:
        """Add error->fix example documents for retrieval. Each item: {error_snippet, fix_snippet, error_type}."""
        if not RAG_AVAILABLE or not examples:
            return
        for ex in examples:
            content = f"Error type: {ex.get('error_type', 'Unknown')}\nError:\n{ex.get('error_snippet', '')}\nFix:\n{ex.get('fix_snippet', '')}"
            self._docs.append(
                Document(page_content=content, metadata=ex)
            )
        emb = self._get_embeddings()
        if emb is not None and self._store is not None:
            self._store.add_documents(self._docs[-len(examples):])
        elif emb is not None:
            self._store = DocArrayInMemorySearch.from_documents(self._docs, emb)

    def retrieve_for_module_version(self, module: str, python_version: str, k: int = 3) -> str:
        """Return concatenated relevant chunks for this module and Python version."""
        if not RAG_AVAILABLE or self._store is None:
            return ""
        query = f"Python package {module} compatible with Python {python_version} available versions"
        try:
            docs = self._store.similarity_search(query, k=k)
            return "\n\n".join(d.page_content for d in docs)
        except Exception:
            return ""

    def retrieve_for_error(self, error_message: str, k: int = 2) -> str:
        """Return similar error->fix context."""
        if not RAG_AVAILABLE or self._store is None:
            return ""
        try:
            docs = self._store.similarity_search(error_message[:2000], k=k)
            return "\n\n".join(d.page_content for d in docs)
        except Exception:
            return ""

    def is_available(self) -> bool:
        return RAG_AVAILABLE and self._store is not None
