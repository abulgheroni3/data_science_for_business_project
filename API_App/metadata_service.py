"""Loading and lookup helpers for DDXPlus metadata JSON files."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from config import Settings
from utils import parse_evidence_token, safe_parse_evidence_list, validate_evidence_ids


class MetadataService:
    """Load metadata without making application import depend on file presence."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.evidences: dict[str, dict[str, Any]] = {}
        self.conditions: dict[str, dict[str, Any]] = {}
        self.evidence_options: list[dict[str, Any]] = []
        self.evidence_options_source = "unavailable"
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

        self.evidence_options = self._build_evidence_options()

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
    
    """""
    # 28062026 - AB - Aggiunto per mostrare le descrizioni delle evidenze, pur mantenendo
    # -- -- la funzione list_evidence_options() per la UI
    def list_evidence_options(self, limit: int | None = None) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []

        for evidence_id, metadata in self.evidences.items():
            question = metadata.get("question_en") or "Metadata label unavailable"
            value_meaning = metadata.get("value_meaning", {})

            if isinstance(value_meaning, dict) and value_meaning:
                for value_key, value_metadata in value_meaning.items():
                    if isinstance(value_metadata, dict):
                        value_label = value_metadata.get("en") or str(value_key)
                    else:
                        value_label = str(value_metadata)

                    records.append(
                        {
                            "token": f"{evidence_id}_@_{value_key}",
                            "evidence_id": evidence_id,
                            "label": f"{question} — {value_label}",
                        }
                    )
            else:
                records.append(
                    {
                        "token": evidence_id,
                        "evidence_id": evidence_id,
                        "label": question,
                    }
                )

        if limit is not None:
            return records[:limit]

        return records
    """
    
    def list_evidence_options(self) -> list[dict[str, Any]]:
        """Return dropdown-ready evidence options for the web UI.

        Small dataset CSVs are scanned once to prefer tokens actually present in
        data/raw. Large local DDXPlus CSVs are intentionally not scanned at app
        startup; in that case the UI falls back to release_evidences.json.
        """
        return self.evidence_options

    def _build_evidence_options(self) -> list[dict[str, Any]]:
        if not self.evidences_loaded:
            self.evidence_options_source = "unavailable"
            return []

        dataset_tokens = self._scan_dataset_evidence_tokens_if_feasible()
        if dataset_tokens:
            self.evidence_options_source = "dataset"
            return self._options_from_tokens(dataset_tokens)

        self.evidence_options_source = "metadata"
        return self._options_from_metadata()

    def _scan_dataset_evidence_tokens_if_feasible(self) -> set[str]:
        data_dir = self.settings.data_raw_dir
        if not data_dir.exists():
            return set()

        csv_paths = sorted(data_dir.glob("*.csv"))
        if not csv_paths:
            return set()

        max_total_bytes = 50 * 1024 * 1024
        total_size = sum(path.stat().st_size for path in csv_paths if path.exists())
        if total_size > max_total_bytes:
            return set()

        tokens: set[str] = set()
        for csv_path in csv_paths:
            try:
                with csv_path.open("r", encoding="utf-8", newline="") as file:
                    reader = csv.DictReader(file)
                    for row in reader:
                        for token in safe_parse_evidence_list(row.get("EVIDENCES")):
                            tokens.add(token)
                        initial_evidence = (row.get("INITIAL_EVIDENCE") or "").strip()
                        if initial_evidence:
                            tokens.add(initial_evidence)
            except (OSError, csv.Error, ValueError):
                return set()
        return tokens

    def _options_from_metadata(self) -> list[dict[str, Any]]:
        options: list[dict[str, Any]] = []
        for evidence_id, metadata in self.evidences.items():
            possible_values = metadata.get("possible-values") or []
            if possible_values:
                for value in possible_values:
                    token = f"{evidence_id}_@_{value}"
                    options.append(self._option_from_token(token))
            else:
                options.append(self._option_from_token(evidence_id))
        return self._sort_options(options)

    def _options_from_tokens(self, tokens: set[str]) -> list[dict[str, Any]]:
        options: list[dict[str, Any]] = []
        for token in tokens:
            try:
                parsed = parse_evidence_token(token)
            except ValueError:
                continue
            if parsed["evidence_id"] in self.evidences:
                options.append(self._option_from_token(token))
        return self._sort_options(options)

    def _option_from_token(self, token: str) -> dict[str, Any]:
        parsed = parse_evidence_token(token)
        evidence_id = parsed["evidence_id"]
        value = parsed["value"]
        metadata = self.evidences.get(evidence_id, {})
        question = metadata.get("question_en") or evidence_id

        if parsed["raw"] == evidence_id:
            label = question
        else:
            label = f"{question} -> {self._value_label(metadata, value)}"

        return {
            "value": parsed["raw"],
            "label": label,
            "title": label,
            "evidence_id": evidence_id,
            "data_type": metadata.get("data_type"),
            "is_antecedent": metadata.get("is_antecedent"),
        }

    @staticmethod
    def _value_label(metadata: dict[str, Any], value: int | str) -> str:
        value_key = str(value)
        value_meaning = metadata.get("value_meaning")
        if isinstance(value_meaning, dict):
            meaning = value_meaning.get(value_key)
            if isinstance(meaning, dict):
                english_label = meaning.get("en")
                if english_label and english_label != "NA":
                    return str(english_label)
        return value_key

    @staticmethod
    def _sort_options(options: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(options, key=lambda item: (str(item["label"]).casefold(), item["value"]))

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
