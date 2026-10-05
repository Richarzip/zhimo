"""Framework-neutral service boundary for future API adapters.

The legacy HTTP implementation still owns the detailed pipeline during this
migration. New integrations should depend on this request/response boundary;
the adapter can be replaced without changing the public contract.
"""

from __future__ import annotations

from typing import Any, Callable

from .contracts import AnalysisRequest


class AnalysisService:
    """Delegate an analysis request to an injected pipeline implementation."""

    def __init__(self, pipeline: Callable[..., dict[str, Any]]):
        self.pipeline = pipeline

    def analyze(self, request: AnalysisRequest) -> dict[str, Any]:
        """Run one request through the injected adapter.

        The adapter owns framework-specific keyword names. This keeps the
        public request contract stable while legacy pipelines are migrated.
        """
        return self.pipeline(request)
