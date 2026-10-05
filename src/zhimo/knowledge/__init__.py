from .repository import KnowledgeRepository
from .data import CALLIGRAPHER_KNOWLEDGE
def build_vectorstore(*args, **kwargs):
    from .chroma import build_vectorstore as _build_vectorstore
    return _build_vectorstore(*args, **kwargs)


def init_knowledge(*args, **kwargs):
    from .chroma import init_knowledge as _init_knowledge
    return _init_knowledge(*args, **kwargs)

__all__ = ["KnowledgeRepository", "CALLIGRAPHER_KNOWLEDGE", "build_vectorstore", "init_knowledge"]
