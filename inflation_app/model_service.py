"""Loading and prediction helpers for the legacy scikit-learn artifacts."""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.neighbors import KNeighborsRegressor


# The order is part of the model contract. Keep it aligned with the original
# training script and the form rendered on the home page.
FEATURE_FIELDS: tuple[tuple[str, str], ...] = (
    ("gdp", "GDP deflator"),
    ("rural", "Rural population"),
    ("psb", "Credit to public-sector banks"),
    ("reserves", "Total reserves"),
    ("fpi", "Foreign portfolio investment"),
    ("agri", "Agricultural land"),
    ("ex", "Exports and imports"),
    ("oda", "Official development assistance"),
    ("co2", "CO₂ emissions"),
    ("cap", "Public-sector capital"),
)


class ModelLoadError(RuntimeError):
    """Raised when a saved model cannot be loaded or used."""


class _LegacyTree:
    """Placeholder for the old Cython tree object embedded in the KNN pickle."""

    def __new__(cls, *_args: Any, **_kwargs: Any):
        return super().__new__(cls)

    def __setstate__(self, state: Any) -> None:
        self.state = state


class _LegacyMetric:
    """Placeholder for the removed ``EuclideanDistance`` Cython class."""

    def __new__(cls, *_args: Any, **_kwargs: Any):
        return super().__new__(cls)

    def __setstate__(self, state: Any) -> None:
        self.state = state


def _legacy_new_object(cls: type) -> Any:
    return cls.__new__(cls)


class _LegacyKNN(KNeighborsRegressor):
    """Read a pre-0.20 KNN pickle, then rebuild its tree with current sklearn."""

    def __setstate__(self, state: dict[str, Any]) -> None:
        # The old tree cannot be used by current sklearn. The training arrays
        # are still valid and are enough to rebuild the index below.
        state.pop("_tree", None)
        state.pop("_fit_method", None)
        super().__setstate__(state)


class _CompatibleUnpickler(pickle.Unpickler):
    """Resolve module paths renamed by scikit-learn after the artifacts shipped."""

    MODULES = {
        "sklearn.neighbors.regression": "sklearn.neighbors._regression",
        "sklearn.ensemble.forest": "sklearn.ensemble._forest",
        "sklearn.tree.tree": "sklearn.tree._classes",
        "sklearn.neural_network.multilayer_perceptron": "sklearn.neural_network._multilayer_perceptron",
    }

    def find_class(self, module: str, name: str) -> Any:
        if module == "sklearn.neighbors.regression" and name == "KNeighborsRegressor":
            return _LegacyKNN
        if module in {"sklearn.neighbors.kd_tree", "sklearn.neighbors.dist_metrics"}:
            if name == "newObj":
                return _legacy_new_object
            if name == "BinaryTree":
                return _LegacyTree
            if name == "EuclideanDistance":
                return _LegacyMetric
        return super().find_class(self.MODULES.get(module, module), name)


class PredictionService:
    """Lazy, cached prediction service for the KNN inflation model."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self._model: Any | None = None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        if not self.model_path.is_file():
            raise ModelLoadError(f"The model artifact was not found: {self.model_path.name}")

        try:
            with self.model_path.open("rb") as model_file, warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                model = _CompatibleUnpickler(model_file).load()
            if isinstance(model, _LegacyKNN):
                model.fit(model._fit_X, np.asarray(model._y).reshape(-1))
            self._model = model
            return model
        except (OSError, pickle.PickleError, AttributeError, ImportError, ValueError) as exc:
            raise ModelLoadError("The saved model could not be loaded by this environment.") from exc

    def predict(self, values: list[float]) -> float:
        if len(values) != len(FEATURE_FIELDS):
            raise ValueError(f"Expected {len(FEATURE_FIELDS)} indicators.")
        features = np.asarray(values, dtype=float).reshape(1, -1)
        if not np.isfinite(features).all():
            raise ValueError("All indicators must be finite numbers.")

        model = self._load_model()
        try:
            raw_prediction = float(np.asarray(model.predict(features)).reshape(-1)[0])
        except (AttributeError, TypeError, ValueError) as exc:
            raise ModelLoadError("The saved model does not accept this feature set.") from exc

        # This is the inverse transform used by the original training script.
        return round(raw_prediction * 36.23 - 7.63, 4)

