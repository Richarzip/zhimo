"""Knowledge retrieval boundary."""

from __future__ import annotations

class KnowledgeRepository:
    def __init__(self, vectorstore=None):
        self._vectorstore = vectorstore

    @property
    def vectorstore(self):
        if self._vectorstore is None:
            from .chroma import build_vectorstore

            self._vectorstore = build_vectorstore()
        return self._vectorstore

    def search(self, query: str, *, limit: int = 2) -> list[dict[str, str]]:
        return [
            {"name": doc.metadata.get("name", "未知"), "text": doc.page_content}
            for doc in self.vectorstore.similarity_search(query, k=limit)
        ]
