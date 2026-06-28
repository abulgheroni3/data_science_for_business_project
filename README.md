# Explainable Differential Diagnosis Assistant for Telemedicine Triage

This project is a Data Science for Business exam prototype built around the DDXPlus dataset. The current implementation adds the application layer only: a FastAPI REST API, a simple Jinja2 web interface, metadata loading, model artifact integration points, and Docker support.

## API and Web Application

### Project purpose

The application exposes DDXPlus evidence and condition metadata and provides a prediction endpoint that can later use trained sklearn-compatible artifacts. It is designed for explainability and reproducibility, while keeping EDA, preprocessing, feature engineering, model training, model evaluation, and `summary.ipynb` separate for future work.

This tool is for educational purposes only and is not a real medical diagnosis.

### Application structure

```text
API_App/
  app.py
  config.py
  schemas.py
  utils.py
  metadata_service.py
  model_service.py
  requirements.txt
  Dockerfile
  templates/index.html
  static/style.css
  metadata/release_evidences.json
  metadata/release_conditions.json
  artifacts/
```

The raw DDXPlus files remain in `data/raw/`. The app does not move or depend on large raw CSV files at runtime.

### Expected files

Detected raw files:

- `data/raw/train.csv`
- `data/raw/validate.csv`
- `data/raw/test.csv`
- `data/raw/release_evidences.json`
- `data/raw/release_conditions.json`

Future model artifacts should be exported to:

- `API_App/artifacts/best_model.pkl`
- `API_App/artifacts/preprocessor.pkl`
- `API_App/artifacts/label_encoder.pkl`
- `API_App/artifacts/model_metrics.json`

### Runtime metadata loading

By default, metadata is resolved from the project root:

- `data/raw/release_evidences.json`
- `data/raw/release_conditions.json`

For a simpler Docker demo, the two JSON metadata files are also copied to `API_App/metadata/`. The app falls back to that folder when the project-root metadata path is not available, such as inside the Docker image.

Supported environment variables:

- `EVIDENCES_JSON_PATH`
- `CONDITIONS_JSON_PATH`
- `MODEL_DIR`
- `DDXPLUS_DATA_DIR`

### Run locally

```bash
cd API_App
pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Web app:

```text
http://localhost:8000
```

API docs:

```text
http://localhost:8000/docs
```

### Run with Docker

```bash
cd API_App
docker build -t ddxplus-diagnosis-api .
docker run -p 8000:8000 ddxplus-diagnosis-api
```

Docker uses Option A: only `release_evidences.json` and `release_conditions.json` are copied into `API_App/metadata/`. The large raw dataset CSV files are not copied into the Docker image.

### API endpoints

- `GET /`
- `GET /health`
- `GET /metadata/evidences?limit=100`
- `GET /metadata/evidences/{evidence_id}`
- `GET /metadata/conditions?limit=100`
- `GET /metadata/conditions/{condition_name}`
- `GET /model-info`
- `GET /metrics`
- `POST /predict`
- `POST /predict-topk`

### Example prediction request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 45,
    "sex": "M",
    "evidences": ["E_91", "E_201", "E_66", "E_56_@_4"],
    "initial_evidence": "E_91"
  }'
```

Use `/predict-topk` to specify `k`:

```json
{
  "age": 45,
  "sex": "M",
  "evidences": ["E_91", "E_201", "E_66"],
  "initial_evidence": "E_91",
  "k": 5
}
```

### Missing model artifacts

The API starts even when trained artifacts are absent. In that state:

- `/health` still works.
- `/metadata/evidences` and `/metadata/conditions` still work if JSON metadata is present.
- `/model-info` returns a clear missing-artifacts message.
- `/metrics` returns a clear message if `model_metrics.json` is missing.
- `/predict` and `/predict-topk` return a controlled HTTP error instead of crashing.
- The web page displays a user-friendly message.

### Leakage note

The DDXPlus CSV includes `DIFFERENTIAL_DIAGNOSIS` in the detected header. That field contains candidate diagnoses and probabilities, so it must not be used as model input. The API feature frame only contains `AGE`, `SEX`, `EVIDENCES`, and `INITIAL_EVIDENCE`; it does not include `PATHOLOGY`, `DIFFERENTIAL_DIAGNOSIS`, or `DIFFERENTIAL_DIGNOSIS`.

### Future artifact assumptions

The future `preprocessor.pkl` should accept a one-row pandas `DataFrame` with these non-leaking columns:

- `AGE`
- `SEX`
- `EVIDENCES`
- `INITIAL_EVIDENCE`

The future `best_model.pkl` should be sklearn-compatible and provide `predict`. If it also provides `predict_proba`, the API returns probabilities and top-k diagnoses. The future `label_encoder.pkl` should expose `classes_` and, ideally, `inverse_transform` so class labels can be decoded cleanly.
