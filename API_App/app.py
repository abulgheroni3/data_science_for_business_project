"""FastAPI routes for the DDXPlus differential diagnosis assistant."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Form, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError

from config import get_settings
from metadata_service import MetadataService
from model_service import (
    ModelNotReadyError,
    ModelService,
    PredictionError,
    PreprocessingError,
)
from schemas import PredictionRequest, PredictionResponse, TopKPredictionRequest
from utils import EvidenceParseError, parse_evidence_token, safe_parse_evidence_list


settings = get_settings()
metadata_service = MetadataService(settings)
model_service = ModelService(settings, metadata_service)

app = FastAPI(
    title="Explainable Differential Diagnosis Assistant",
    description="Educational FastAPI application layer for DDXPlus telemedicine triage experiments.",
    version="0.1.0",
)

app.mount("/static", StaticFiles(directory=settings.api_dir / "static"), name="static")
templates = Jinja2Templates(directory=settings.api_dir / "templates")


def _metadata_unavailable(detail: str) -> HTTPException:
    return HTTPException(
        status_code=503,
        detail={
            "message": detail,
            "load_errors": metadata_service.load_errors,
            "missing_files": metadata_service.missing_metadata_files(),
        },
    )


def _validate_request_evidences(request: PredictionRequest) -> None:
    for token in request.evidences:
        parse_evidence_token(token)
    if request.initial_evidence:
        parse_evidence_token(request.initial_evidence)

    invalid_ids = metadata_service.validate_evidences(request.evidences)
    if request.initial_evidence and metadata_service.evidences_loaded:
        initial_id = parse_evidence_token(request.initial_evidence)["evidence_id"]
        if initial_id not in metadata_service.evidences:
            invalid_ids.append(initial_id)

    if invalid_ids:
        unique_invalid = list(dict.fromkeys(invalid_ids))
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Unknown evidence ID found in request.",
                "invalid_evidence_ids": unique_invalid,
            },
        )


def _prediction_payload(request: PredictionRequest, k: int) -> dict[str, Any]:
    try:
        _validate_request_evidences(request)
        return model_service.predict(request, k=k)
    except EvidenceParseError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ModelNotReadyError as exc:
        raise HTTPException(
            status_code=503,
            detail={"message": exc.message, "missing_files": exc.missing_files},
        ) from exc
    except PreprocessingError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except PredictionError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _health_payload() -> dict[str, Any]:
    missing_files = (
        metadata_service.missing_metadata_files()
        + model_service.missing_artifacts()
    )
    return {
        "status": "ok",
        "metadata_loaded": metadata_service.evidences_loaded,
        "conditions_loaded": metadata_service.conditions_loaded,
        "model_loaded": model_service.model_loaded and model_service.preprocessor_loaded,
        "missing_files": missing_files,
        "data_raw_files_detected": settings.detect_data_raw_files(),
        "metadata_paths": {
            "evidences": settings.display_path(settings.evidences_json_path),
            "conditions": settings.display_path(settings.conditions_json_path),
        },
    }


def _template_context(
    request: Request,
    result: dict[str, Any] | None = None,
    error: str | None = None,
    form_values: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "request": request,
        "health": _health_payload(),
        "model_info": model_service.info(),
        "sample_evidences": metadata_service.list_evidences(limit=8)
        if metadata_service.evidences_loaded
        else [],
        "evidence_options": metadata_service.list_evidence_options(),
        "evidence_options_source": metadata_service.evidence_options_source,
        "evidence_option_values": [
            option["value"] for option in metadata_service.list_evidence_options()
        ],
        "result": result,
        "error": error,
        "form": form_values
        or {
            "age": "",
            "sex": "M",
            "evidences": "E_91, E_201, E_66, E_56_@_4",
            "initial_evidence": "E_91",
            "k": 5,
        },
    }


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Unexpected server error. Check server logs for details.",
            "error_type": type(exc).__name__,
        },
    )


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html", _template_context(request))


@app.post("/web-predict", response_class=HTMLResponse)
async def web_predict(
    request: Request,
    age: str = Form(""),
    sex: str = Form(""),
    evidences: str = Form(""),
    initial_evidence: str = Form(""),
    k: str = Form("5"),
) -> HTMLResponse:
    form_values = {
        "age": age,
        "sex": sex,
        "evidences": evidences,
        "initial_evidence": initial_evidence,
        "k": k,
    }
    try:
        evidence_list = safe_parse_evidence_list(evidences)
        payload = TopKPredictionRequest(
            age=int(age),
            sex=sex,
            evidences=evidence_list,
            initial_evidence=initial_evidence or None,
            k=int(k),
        )
        result = _prediction_payload(payload, payload.k)
        return templates.TemplateResponse(
            request,
            "index.html",
            _template_context(request, result=jsonable_encoder(result), form_values=form_values),
        )
    except (ValueError, ValidationError, HTTPException, EvidenceParseError) as exc:
        if isinstance(exc, HTTPException):
            detail = exc.detail
            if isinstance(detail, dict):
                error = detail.get("message", str(detail))
                if detail.get("missing_files"):
                    error = f"{error} Missing files: {', '.join(detail['missing_files'])}"
            else:
                error = str(detail)
        else:
            error = str(exc)
        return templates.TemplateResponse(
            request,
            "index.html",
            _template_context(request, error=error, form_values=form_values),
            status_code=200,
        )


@app.get("/health")
async def health() -> dict[str, Any]:
    return _health_payload()


@app.get("/metadata/evidences")
async def evidences(limit: int = Query(100, ge=1, le=1000)) -> list[dict[str, Any]]:
    if not metadata_service.evidences_loaded:
        raise _metadata_unavailable("Evidence metadata is not available.")
    return metadata_service.list_evidences(limit=limit)


@app.get("/metadata/evidences/{evidence_id}")
async def evidence_detail(evidence_id: str) -> dict[str, Any]:
    if not metadata_service.evidences_loaded:
        raise _metadata_unavailable("Evidence metadata is not available.")
    evidence = metadata_service.get_evidence(evidence_id)
    if not evidence:
        raise HTTPException(status_code=404, detail=f"Evidence ID '{evidence_id}' was not found.")
    return evidence


@app.get("/metadata/conditions")
async def conditions(limit: int = Query(100, ge=1, le=1000)) -> list[dict[str, Any]]:
    if not metadata_service.conditions_loaded:
        raise _metadata_unavailable("Condition metadata is not available.")
    return metadata_service.list_conditions(limit=limit)


@app.get("/metadata/conditions/{condition_name}")
async def condition_detail(condition_name: str) -> dict[str, Any]:
    if not metadata_service.conditions_loaded:
        raise _metadata_unavailable("Condition metadata is not available.")
    condition = metadata_service.get_condition(condition_name)
    if not condition:
        raise HTTPException(status_code=404, detail=f"Condition '{condition_name}' was not found.")
    return condition


@app.get("/model-info")
async def model_info() -> dict[str, Any]:
    return model_service.info()


@app.get("/metrics")
async def metrics() -> dict[str, Any]:
    try:
        return model_service.metrics()
    except Exception as exc:  # noqa: BLE001 - malformed metrics should be a clear API response
        raise HTTPException(status_code=500, detail=f"Could not read metrics file: {exc}") from exc


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> dict[str, Any]:
    return _prediction_payload(request, k=5)


@app.post("/predict-topk", response_model=PredictionResponse)
async def predict_topk(request: TopKPredictionRequest) -> dict[str, Any]:
    return _prediction_payload(request, k=request.k)
