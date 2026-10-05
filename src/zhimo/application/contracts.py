"""Small, dependency-light contracts shared by adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class AnalysisRequest:
    image: Any
    mode: Literal["single", "multi", "chat"] = "single"
    tta: bool = False
    cam: bool = False
    rag: bool = False
    denoise: bool = False
    prompt: str = ""
    session_id: str | None = None


@dataclass
class AnalysisResponse:
    mode: str
    recognition: dict[str, Any] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)
    segmentation: dict[str, Any] = field(default_factory=dict)
    knowledge: Any = None
    evidence: dict[str, Any] = field(default_factory=dict)
    steps: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "recognition": self.recognition,
            "quality": self.quality,
            "segmentation": self.segmentation,
            "knowledge": self.knowledge,
            "evidence": self.evidence,
            "steps": self.steps,
            "diagnostics": self.diagnostics,
        }
