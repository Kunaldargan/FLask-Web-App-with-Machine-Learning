"""Flask application factory and HTTP routes."""

from __future__ import annotations

import math
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, flash, redirect, render_template, request, url_for

from .model_service import FEATURE_FIELDS, ModelLoadError, PredictionService


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_evaluation_data() -> tuple[np.ndarray, np.ndarray]:
    """Read the saved prediction comparison arrays used by the graph page."""

    with (PROJECT_ROOT / "pred").open("rb") as predictions_file:
        predictions = np.asarray(pickle.load(predictions_file), dtype=float).reshape(-1)
    with (PROJECT_ROOT / "ytest").open("rb") as actual_file:
        actual = np.asarray(pickle.load(actual_file), dtype=float).reshape(-1)
    return predictions, actual


def create_app(test_config: dict[str, Any] | None = None) -> Flask:
    """Create and configure the Flask application."""

    app = Flask(
        __name__,
        template_folder=str(PROJECT_ROOT / "templates"),
        static_folder=str(PROJECT_ROOT / "static"),
    )
    app.config.from_mapping(
        SECRET_KEY=os.getenv("SECRET_KEY", "development-only-change-me"),
        PROJECT_ROOT=PROJECT_ROOT,
    )
    if test_config:
        app.config.update(test_config)

    prediction_service = PredictionService(PROJECT_ROOT / "logmodel.pkl")

    @app.get("/")
    def home():
        return render_template(
            "home.html",
            feature_fields=FEATURE_FIELDS,
            models=[
                {
                    "name": "K-nearest neighbors",
                    "short_name": "KNN",
                    "description": "The live model used by the prediction form.",
                    "status": "Available",
                },
                {
                    "name": "Random forest",
                    "short_name": "RF",
                    "description": "A saved comparison model from the original project.",
                    "status": "Reference",
                },
                {
                    "name": "Decision tree",
                    "short_name": "DT",
                    "description": "A compact tree-based regression baseline.",
                    "status": "Reference",
                },
                {
                    "name": "Neural network",
                    "short_name": "NN",
                    "description": "A saved multilayer perceptron comparison model.",
                    "status": "Reference",
                },
            ],
        )

    @app.route("/predict", methods=["POST"])
    @app.route("/result", methods=["POST"])
    def predict():
        values: dict[str, float] = {}
        try:
            for field, label in FEATURE_FIELDS:
                raw_value = request.form.get(field, "").strip()
                if not raw_value:
                    raise ValueError(f"{label} is required.")
                value = float(raw_value)
                if not math.isfinite(value):
                    raise ValueError(f"{label} must be a finite number.")
                values[field] = value
        except ValueError as exc:
            flash(str(exc), "error")
            return redirect(url_for("home"))

        try:
            prediction = prediction_service.predict(list(values.values()))
        except ModelLoadError as exc:
            app.logger.exception("Unable to load the prediction model")
            flash(str(exc), "error")
            return redirect(url_for("home"))
        except ValueError as exc:
            flash(str(exc), "error")
            return redirect(url_for("home"))

        return render_template(
            "result.html",
            prediction=prediction,
            values=values,
            year=request.form.get("year", "") or "Not provided",
        )

    @app.get("/result")
    def result_get():
        flash("Enter the indicators on the home page to generate a prediction.", "info")
        return redirect(url_for("home"))

    @app.get("/graph")
    def graph():
        try:
            predictions, actual = _load_evaluation_data()
            count = min(predictions.size, actual.size)
            points = [
                {
                    "sample": index + 1,
                    "predicted": round(float(predictions[index]), 4),
                    "actual": round(float(actual[index]), 4),
                }
                for index in range(count)
            ]
            mse = float(np.mean((predictions[:count] - actual[:count]) ** 2)) if count else None
        except (OSError, pickle.PickleError, ValueError) as exc:
            app.logger.warning("Evaluation data is unavailable: %s", exc)
            points, mse = [], None

        return render_template("graph.html", points=points, mse=mse)

    @app.route("/contact", methods=["GET", "POST"])
    def contact():
        if request.method == "POST":
            name = request.form.get("name", "").strip()
            if not name:
                flash("Please add your name before sending feedback.", "error")
            else:
                flash(f"Thanks, {name}. Your feedback has been recorded for this demo.", "success")
                return redirect(url_for("contact"))
        return render_template("contact.html")

    @app.get("/about")
    def about():
        return render_template("about.html")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.errorhandler(404)
    def not_found(_error):
        return render_template("404.html"), 404

    return app


app = create_app()

