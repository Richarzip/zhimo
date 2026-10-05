from .contracts import AnalysisRequest, AnalysisResponse
from .service import AnalysisService


def resolve_model_paths():
    from .factory import resolve_model_paths as _resolve_model_paths
    return _resolve_model_paths()


def get_recognizer():
    from .factory import get_recognizer as _get_recognizer
    return _get_recognizer()


def run_single(*args, **kwargs):
    from .pipeline import run_single as _run_single
    return _run_single(*args, **kwargs)


def run_multi(*args, **kwargs):
    from .pipeline import run_multi as _run_multi
    return _run_multi(*args, **kwargs)


def run_chat(*args, **kwargs):
    from .pipeline import run_chat as _run_chat
    return _run_chat(*args, **kwargs)

def get_agent():
    from .agent_runtime import get_agent as _get_agent
    return _get_agent()


__all__ = [
    "AnalysisRequest",
    "AnalysisResponse",
    "get_recognizer",
    "resolve_model_paths",
    "AnalysisService",
    "run_single",
    "run_multi",
    "run_chat",
    "get_agent",
]
