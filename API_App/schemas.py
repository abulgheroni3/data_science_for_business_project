"""Pydantic schemas for API requests and responses."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, validator


class PredictionRequest(BaseModel):
    age: int = Field(..., ge=0, le=130)
    sex: str
    evidences: list[str]
    initial_evidence: str | None = None

    @validator("sex")
    def validate_sex(cls, value: str) -> str:
        normalized = value.strip().upper()
        if normalized not in {"M", "F"}:
            raise ValueError("sex must be 'M' or 'F'.")
        return normalized

    @validator("evidences")
    def validate_evidences(cls, value: list[str]) -> list[str]:
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        if not cleaned:
            raise ValueError("evidences must contain at least one evidence token.")
        return cleaned

    @validator("initial_evidence")
    def clean_initial_evidence(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class TopKPredictionRequest(PredictionRequest):
    k: int = Field(5, ge=1, le=20)


class TopKDiagnosis(BaseModel):
    diagnosis: str
    probability: float | None = None
    icd10_id: str | None = None
    severity: int | None = None


class InputSummary(BaseModel):
    age: int
    sex: str
    n_evidences: int
    initial_evidence: str | None = None


class InterpretedEvidence(BaseModel):
    raw: str
    evidence_id: str
    value: int | str
    question_en: str | None = None
    data_type: str | None = None
    is_antecedent: bool | None = None
    value_meaning_en: str | None = None


class PredictionResponse(BaseModel):
    predicted_diagnosis: str
    confidence: float | None = None
    top_k_diagnoses: list[TopKDiagnosis]
    input_summary: InputSummary
    interpreted_evidences: list[InterpretedEvidence]
    disclaimer: str


class MessageResponse(BaseModel):
    message: str
    details: dict[str, Any] | None = None
