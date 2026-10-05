"""Runtime configuration shared by inference, RAG and web adapters."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _paths_from_env() -> tuple[Path, ...] | None:
    raw = os.getenv("ZHIMO_MODEL_PATHS")
    if raw:
        return tuple(
            (Path(value.strip()) if Path(value.strip()).is_absolute() else PROJECT_ROOT / value.strip())
            for value in raw.split(os.pathsep)
            if value.strip()
        )
    single = os.getenv("ZHIMO_MODEL_PATH")
    if single:
        path = Path(single)
        return (path if path.is_absolute() else PROJECT_ROOT / path,)
    return None


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    model_paths: tuple[Path, ...] = (
        PROJECT_ROOT / "checkpoints" / "convnext.pth",
        PROJECT_ROOT / "checkpoints" / "swin.pth",
    )
    chroma_dir: Path = PROJECT_ROOT / "chroma_db"
    max_image_bytes: int = 32 * 1024 * 1024
    max_image_pixels: int = 25_000_000
    web_host: str = "127.0.0.1"
    web_port: int = 8765

    @property
    def ensemble_enabled(self) -> bool:
        return len(self.model_paths) > 1

    @property
    def models_exist(self) -> bool:
        return all(path.exists() for path in self.model_paths)


def get_settings() -> Settings:
    paths = _paths_from_env()
    return Settings(
        model_paths=paths or Settings.model_paths,
        chroma_dir=Path(os.getenv("ZHIMO_CHROMA_DIR", str(Settings.chroma_dir))),
        web_host=os.getenv("ZHIMO_HOST", Settings.web_host),
        web_port=int(os.getenv("ZHIMO_PORT", str(Settings.web_port))),
    )
