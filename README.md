# Inflation Lab

Inflation Lab is a small Flask application that exposes a saved scikit-learn regression model through a responsive web interface. Enter ten economic indicators, run the preserved K-nearest-neighbors model, and compare its saved predictions with the test values on the evaluation page.

The project began as a 2017 classroom experiment. This pass gives it a conventional Flask entry point, a clear application boundary, shared templates, input validation, a compatibility loader for the old model artifact, and documentation that reflects what actually runs.

## Quick start

The application expects Python 3.9+ and a local checkout of the serialized model files.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:5000>. Set `FLASK_DEBUG=1` only during local development if you want Flask’s reloader and debugger.
For a deployed instance, set a strong random `SECRET_KEY` in the environment.

The production-style Flask command is also available with Flask 2.2+:

```bash
flask --app app:app run
```

## Application structure

```text
app.py                     Canonical development entry point
inflation_app/
  __init__.py               App factory and routes
  model_service.py          Lazy model loading and prediction contract
templates/                  Shared responsive UI
static/css/app.css          Application styles
static/js/app.js            Small progressive-enhancement script
*.pkl, pred, ytest          Preserved model and evaluation artifacts
legacy/                     Original training scripts and unused UI assets
```

`_init_.py` remains as a compatibility shim for anyone using the original filename. New code should import `app` or `create_app` from `app.py`.

## Routes

| Route | Method | Purpose |
| --- | --- | --- |
| `/` | GET | Prediction form and model overview |
| `/predict` | POST | Validates inputs and renders a prediction |
| `/result` | GET/POST | Backward-compatible alias for prediction |
| `/graph` | GET | Saved predicted-vs-actual evaluation |
| `/about` | GET | Project and artifact notes |
| `/contact` | GET/POST | Demo feedback form |
| `/health` | GET | Lightweight health check |

## Prediction contract

The live model receives exactly these ten values, in this order:

1. GDP deflator (`gdp`)
2. Rural population (`rural`)
3. Credit to public-sector banks (`psb`)
4. Total reserves (`reserves`)
5. Foreign portfolio investment (`fpi`)
6. Agricultural land (`agri`)
7. Exports and imports (`ex`)
8. Official development assistance (`oda`)
9. CO₂ emissions (`co2`)
10. Public-sector capital (`cap`)

The original training script stored a normalized target and later converted the model output with `prediction * 36.23 - 7.63`. The application preserves that transformation for compatibility with the shipped artifact. The `year`/month field is displayed for context and is not passed to the model because the serialized estimator was trained with ten features.

## Model artifacts

The model files were serialized with an older scikit-learn release. `inflation_app/model_service.py` remaps the old module paths and rebuilds the saved KNN search tree from the training arrays when the artifact is loaded. This keeps the demo usable on current supported environments, but it does not make the historical model production-grade.

The original scripts refer to a machine-specific `pro.csv` path and use removed scikit-learn APIs. They are preserved under `legacy/training/` for historical reference rather than imported by the web app. To create a trustworthy production model, supply a versioned dataset, persist the preprocessing pipeline with the estimator, retrain with a supported scikit-learn version, and add model-quality tests.

## Verification

Run the lightweight checks from the project root:

```bash
python -m compileall -q app.py _init_.py inflation_app
python - <<'PY'
from app import app

client = app.test_client()
assert client.get('/').status_code == 200
assert client.get('/health').json == {'status': 'ok'}
assert client.get('/graph').status_code == 200
PY
```

## Limitations

- The application is an educational demo, not an investment or policy tool.
- The serialized models and evaluation arrays are committed binary artifacts; inspect their provenance before using them for decisions.
- The contact form displays a confirmation but does not send email or persist messages.
- The evaluation chart loads Chart.js from a CDN; the data table remains available if JavaScript is disabled.
