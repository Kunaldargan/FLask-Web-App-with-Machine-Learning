"""Train a small 1D CNN over tabular time-series windows.

This is an optional experiment for the original economic time-series data. A
window of consecutive observations is treated as a one-dimensional signal;
Conv1D filters learn short-term temporal patterns before a regression head
predicts the next target value.

TensorFlow is imported only when training starts so the Flask application does
not require a deep-learning runtime just to serve the existing model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler


def load_series(data_path: Path, target: str, time_column: str | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load numeric features and the target in chronological order."""

    frame = pd.read_csv(data_path)
    if time_column:
        if time_column not in frame.columns:
            raise ValueError(f"Time column not found: {time_column}")
        frame = frame.sort_values(time_column).reset_index(drop=True)
    if target not in frame.columns:
        raise ValueError(f"Target column not found: {target}")

    numeric = frame.select_dtypes(include=[np.number])
    if time_column and time_column in numeric.columns:
        numeric = numeric.drop(columns=[time_column])
    if target not in numeric.columns:
        raise ValueError(f"Target column must be numeric: {target}")
    feature_columns = list(numeric.columns)
    features = numeric.to_numpy(dtype=np.float32)
    labels = numeric[target].to_numpy(dtype=np.float32).reshape(-1, 1)
    return features, labels, feature_columns


def make_windows(features: np.ndarray, labels: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create ``[samples, time, features]`` windows and their target indices."""

    if window_size < 2:
        raise ValueError("window_size must be at least 2")
    if len(features) <= window_size:
        raise ValueError("The time series must contain more rows than window_size")

    windows, targets, target_indices = [], [], []
    for end in range(window_size, len(features)):
        windows.append(features[end - window_size : end])
        targets.append(labels[end])
        target_indices.append(end)
    return np.asarray(windows), np.asarray(targets), np.asarray(target_indices)


def build_model(window_size: int, feature_count: int) -> Any:
    """Build the temporal CNN lazily so TensorFlow remains optional."""

    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required for the CNN experiment. Install requirements-cnn.txt first."
        ) from exc

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(window_size, feature_count)),
            tf.keras.layers.Conv1D(32, kernel_size=3, padding="causal", activation="relu"),
            tf.keras.layers.Dropout(0.15),
            tf.keras.layers.Conv1D(16, kernel_size=3, padding="causal", activation="relu"),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1),
        ],
        name="economic_time_series_cnn",
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse", metrics=["mae"])
    return model


def train(
    data_path: Path,
    target: str,
    time_column: str | None,
    output_dir: Path,
    window_size: int,
    epochs: int,
    batch_size: int,
) -> dict[str, float | int | str]:
    """Fit the CNN chronologically and save its model, predictions, and metrics."""

    features, labels, feature_columns = load_series(data_path, target, time_column)
    split_index = max(window_size + 1, int(len(features) * 0.8))
    if split_index >= len(features):
        raise ValueError("Not enough rows for a chronological validation split")

    feature_scaler = MinMaxScaler().fit(features[:split_index])
    target_scaler = MinMaxScaler().fit(labels[:split_index])
    scaled_features = feature_scaler.transform(features)
    scaled_labels = target_scaler.transform(labels)
    windows, targets, target_indices = make_windows(scaled_features, scaled_labels, window_size)

    train_mask = target_indices < split_index
    if not train_mask.any() or train_mask.all():
        raise ValueError("The chosen window and dataset do not leave train and validation samples")
    x_train, y_train = windows[train_mask], targets[train_mask]
    x_test, y_test = windows[~train_mask], targets[~train_mask]

    model = build_model(window_size, len(feature_columns))
    import tensorflow as tf

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
        verbose=1,
    )
    predictions = model.predict(x_test, verbose=0)
    actual = target_scaler.inverse_transform(y_test)
    predicted = target_scaler.inverse_transform(predictions)

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / "time_series_cnn.keras")
    pd.DataFrame({"actual": actual.reshape(-1), "predicted": predicted.reshape(-1)}).to_csv(
        output_dir / "predictions.csv", index=False
    )
    metrics = {
        "data": str(data_path),
        "target": target,
        "features": ",".join(feature_columns),
        "window_size": window_size,
        "train_samples": int(len(x_train)),
        "validation_samples": int(len(x_test)),
        "epochs_run": len(history.history.get("loss", [])),
        "mae": round(float(mean_absolute_error(actual, predicted)), 6),
        "rmse": round(float(np.sqrt(mean_squared_error(actual, predicted))), 6),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a 1D CNN over chronological economic time-series windows.")
    parser.add_argument("--data", type=Path, required=True, help="CSV containing the time series")
    parser.add_argument("--target", required=True, help="Numeric target column to predict")
    parser.add_argument("--time-column", help="Optional column used to sort rows chronologically")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/time-series-cnn"))
    parser.add_argument("--window-size", type=int, default=5, help="Number of past rows in each CNN window")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    metrics = train(
        args.data,
        args.target,
        args.time_column,
        args.output_dir,
        args.window_size,
        args.epochs,
        args.batch_size,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
