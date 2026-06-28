"""Loading and lookup helpers for DDXPlus metadata JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import Settings
from utils import parse_evidence_token, validate_evidence_ids


class MetadataService:
    """Load metadata without making application import depend on file presence."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.evidences: dict[str, dict[str, Any]] = {}
        self.conditions: dict[str, dict[str, Any]] = {}
        self.load_errors: list[str] = []
        self.load()

    @staticmethod
    def _load_json(path: Path) -> dict[str, dict[str, Any]]:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, dict):
            raise ValueError(f"{path.name} must contain a JSON object.")
        return data

    def load(self) -> None:
        self.load_errors = []
        self.evidences = {}
        self.conditions = {}

        if self.settings.evidences_json_path.exists():
            try:
                self.evidences = self._load_json(self.settings.evidences_json_path)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                self.load_errors.append(f"Could not load evidences metadata: {exc}")
        else:
            self.load_errors.append(
                f"Missing evidences metadata: {self.settings.display_path(self.settings.evidences_json_path)}"
            )

        if self.settings.conditions_json_path.exists():
            try:
                self.conditions = self._load_json(self.settings.conditions_json_path)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                self.load_errors.append(f"Could not load conditions metadata: {exc}")
        else:
            self.load_errors.append(
                f"Missing conditions metadata: {self.settings.display_path(self.settings.conditions_json_path)}"
            )

    @property
    def evidences_loaded(self) -> bool:
        return bool(self.evidences)

    @property
    def conditions_loaded(self) -> bool:
        return bool(self.conditions)

    def missing_metadata_files(self) -> list[str]:
        return self.settings.missing_paths(
            [self.settings.evidences_json_path, self.settings.conditions_json_path]
        )

    def list_evidences(self, limit: int = 100) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for evidence_id, metadata in list(self.evidences.items())[:limit]:
            records.append(
                {
                    "evidence_id": evidence_id,
                    "question_en": metadata.get("question_en"),
                    "data_type": metadata.get("data_type"),
                    "is_antecedent": metadata.get("is_antecedent"),
                }
            )
        return records

    def get_evidence(self, evidence_id: str) -> dict[str, Any] | None:
        return self.evidences.get(evidence_id)

    def list_conditions(self, limit: int = 100) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for condition_name, metadata in list(self.conditions.items())[:limit]:
            records.append(
                {
                    "condition_name": metadata.get("condition_name", condition_name),
                    "condition_name_eng": metadata.get("cond-name-eng"),
                    "icd10_id": metadata.get("icd10-id"),
                    "severity": metadata.get("severity"),
                }
            )
        return records

    def get_condition(self, condition_name: str) -> dict[str, Any] | None:
        if condition_name in self.conditions:
            return self.conditions[condition_name]

        normalized = condition_name.casefold()
        for key, metadata in self.conditions.items():
            aliases = {
                key.casefold(),
                str(metadata.get("condition_name", "")).casefold(),
                str(metadata.get("cond-name-eng", "")).casefold(),
            }
            if normalized in aliases:
                return metadata
        return None

    def condition_summary(self, condition_name: str) -> dict[str, Any]:
        metadata = self.get_condition(condition_name)
        if not metadata:
            return {"icd10_id": None, "severity": None}
        return {
            "icd10_id": metadata.get("icd10-id"),
            "severity": metadata.get("severity"),
        }

    def validate_evidences(self, evidences: list[str]) -> list[str]:
        if not self.evidences_loaded:
            return []
        return validate_evidence_ids(evidences, self.evidences)

    def interpret_evidences(self, evidences: list[str]) -> list[dict[str, Any]]:
        interpreted: list[dict[str, Any]] = []
        for token in evidences:
            parsed = parse_evidence_token(token)
            evidence_id = parsed["evidence_id"]
            metadata = self.evidences.get(evidence_id, {})
            value = parsed["value"]
            value_meaning = metadata.get("value_meaning", {})
            value_key = str(value)

            record = {
                "raw": parsed["raw"],
                "evidence_id": evidence_id,
                "value": value,
                "question_en": metadata.get("question_en"),
                "data_type": metadata.get("data_type"),
                "is_antecedent": metadata.get("is_antecedent"),
            }
            if isinstance(value_meaning, dict) and value_key in value_meaning:
                record["value_meaning_en"] = value_meaning[value_key].get("en")
            interpreted.append(record)
        return interpreted
