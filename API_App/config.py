"""Application configuration and filesystem path handling."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable


class Settings:
    """Resolve runtime paths from environment variables with safe defaults."""

    def __init__(self) -> None:
        self.api_dir = Path(__file__).resolve().parent
        self.project_root = self.api_dir.parent

        self.data_raw_dir = self._env_path(
            "DDXPLUS_DATA_DIR",
            self.project_root / "data" / "raw",
        )
        self.model_dir = self._env_path("MODEL_DIR", self.api_dir / "artifacts")

        self.metadata_dir = self.api_dir / "metadata"
        self.evidences_json_path = self._metadata_path(
            env_name="EVIDENCES_JSON_PATH",
            default_path=self.data_raw_dir / "release_evidences.json",
            fallback_path=self.metadata_dir / "release_evidences.json",
        )
        self.conditions_json_path = self._metadata_path(
            env_name="CONDITIONS_JSON_PATH",
            default_path=self.data_raw_dir / "release_conditions.json",
            fallback_path=self.metadata_dir / "release_conditions.json",
        )

        self.model_path = self.model_dir / "best_model.pkl"
        self.preprocessor_path = self.model_dir / "preprocessor.pkl"
        self.label_encoder_path = self.model_dir / "label_encoder.pkl"
        self.metrics_path = self.model_dir / "model_metrics.json"

    @staticmethod
    def _env_path(env_name: str, default_path: Path) -> Path:
        value = os.getenv(env_name)
        if value:
            return Path(value).expanduser().resolve()
        return default_path.resolve()

    def _metadata_path(self, env_name: str, default_path: Path, fallback_path: Path) -> Path:
        env_value = os.getenv(env_name)
        if env_value:
            return Path(env_value).expanduser().resolve()
        if default_path.exists():
            return default_path.resolve()
        return fallback_path.resolve()

    def detect_data_raw_files(self) -> list[str]:
        """Return detected files in data/raw without reading large datasets."""
        if not self.data_raw_dir.exists():
            return []
        return sorted(path.name for path in self.data_raw_dir.iterdir() if path.is_file())

    def display_path(self, path: Path) -> str:
        """Show project-relative paths when possible for clearer API responses."""
        try:
            return path.resolve().relative_to(self.project_root).as_posix()
        except ValueError:
            return path.resolve().as_posix()

    def missing_paths(self, paths: Iterable[Path]) -> list[str]:
        return [self.display_path(path) for path in paths if not path.exists()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
