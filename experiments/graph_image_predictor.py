"""Extract and extrapolate an inflation curve from a graph image.

This experiment is intentionally explicit about its assumptions. It detects
the orange line in the checked-in KNN plot, smooths the pixel trace, converts
pixels back to the plot's y-axis range, and fits a small linear extrapolation
over the last few points. It demonstrates image-based graph digitization; it
is not a replacement for forecasting from the original numeric dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .computer_vision import _fit, _font, _panel


# Calibration for static/images/knnPlot1.png. The values correspond to the
# visible plot rectangle and its y-axis labels (0.25 through 0.60).
DEFAULT_CALIBRATION = {
    "x_start": 66,
    "x_end": 369,
    "y_top": 16,
    "y_bottom": 224,
    "y_min": 0.25,
    "y_max": 0.60,
    "sample_count": 22,
}


def color_mask(image: Image.Image, target: tuple[int, int, int], tolerance: float) -> np.ndarray:
    """Return pixels close to an RGB line color as a boolean mask."""

    pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    distance = np.linalg.norm(pixels - np.asarray(target, dtype=np.float32), axis=2)
    return distance <= tolerance


def extract_trace(
    image: Image.Image,
    target: tuple[int, int, int] = (255, 127, 14),
    tolerance: float = 70,
    calibration: dict[str, float | int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract per-column y pixels, resampled values, and the filtered mask."""

    settings = {**DEFAULT_CALIBRATION, **(calibration or {})}
    mask = color_mask(image, target, tolerance)
    x_start, x_end = int(settings["x_start"]), int(settings["x_end"])
    y_top, y_bottom = int(settings["y_top"]), int(settings["y_bottom"])
    region = mask[y_top : y_bottom + 1, x_start : x_end + 1]
    column_y = np.full(region.shape[1], np.nan, dtype=np.float32)
    for column in range(region.shape[1]):
        rows = np.where(region[:, column])[0]
        if rows.size:
            column_y[column] = float(np.median(rows) + y_top)

    valid = np.isfinite(column_y)
    if valid.sum() < 10:
        raise ValueError("Could not find enough pixels for the selected graph line")

    pixel_x = np.arange(x_start, x_end + 1, dtype=np.float32)
    filtered_y = np.interp(pixel_x, pixel_x[valid], column_y[valid])
    sample_x = np.linspace(x_start, x_end, int(settings["sample_count"]))
    sample_y = np.interp(sample_x, pixel_x, filtered_y)
    return sample_x, sample_y, mask


def smooth_trace(values: np.ndarray, window: int = 3) -> np.ndarray:
    """Reduce anti-aliasing noise with a median and moving-average filter."""

    if window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd number")
    if window == 1:
        return values.astype(np.float32)
    radius = window // 2
    padded = np.pad(values, radius, mode="edge")
    median = np.array([np.median(padded[index : index + window]) for index in range(len(values))])
    averaged = np.convolve(np.pad(median, radius, mode="edge"), np.ones(window) / window, mode="valid")
    return averaged.astype(np.float32)


def pixels_to_values(pixel_y: np.ndarray, calibration: dict[str, float | int] | None = None) -> np.ndarray:
    """Map image-space y coordinates to the graph's calibrated y values."""

    settings = {**DEFAULT_CALIBRATION, **(calibration or {})}
    y_top, y_bottom = float(settings["y_top"]), float(settings["y_bottom"])
    y_min, y_max = float(settings["y_min"]), float(settings["y_max"])
    return y_max - (pixel_y - y_top) / (y_bottom - y_top) * (y_max - y_min)


def predict_next(values: np.ndarray, last_points: int = 5) -> float:
    """Fit a linear trend to the final values and extrapolate one sample."""

    if last_points < 2 or last_points > len(values):
        raise ValueError("last_points must be between 2 and the number of samples")
    tail = values[-last_points:]
    slope, intercept = np.polyfit(np.arange(last_points, dtype=np.float32), tail, 1)
    return float(np.polyval((slope, intercept), last_points))


def _series_chart(values: np.ndarray, smoothed: np.ndarray, forecast: float, y_min: float, y_max: float) -> Image.Image:
    size = (560, 340)
    chart = Image.new("RGB", size, "#102f3b")
    draw = ImageDraw.Draw(chart)
    left, top, right, bottom = 58, 28, 530, 290
    draw.rectangle((left, top, right, bottom), outline="#8eafb0", width=1)
    for fraction in (0, 0.25, 0.5, 0.75, 1):
        y = bottom - fraction * (bottom - top)
        draw.line((left, y, right, y), fill="#31535d", width=1)
        label = y_min + fraction * (y_max - y_min)
        draw.text((8, y - 8), f"{label:.2f}", fill="#d8e6e1", font=_font(14))

    def point(index: float, value: float) -> tuple[float, float]:
        x = left + index / len(values) * (right - left)
        y = bottom - (value - y_min) / (y_max - y_min) * (bottom - top)
        return x, y

    actual_points = [point(index, float(value)) for index, value in enumerate(values)]
    smooth_points = [point(index, float(value)) for index, value in enumerate(smoothed)]
    draw.line(actual_points, fill="#ff8a5c", width=3)
    draw.line(smooth_points, fill="#68d0bf", width=3)
    forecast_x, forecast_y = point(len(values), forecast)
    draw.line((actual_points[-1][0], actual_points[-1][1], forecast_x, forecast_y), fill="#f4d35e", width=2)
    draw.ellipse((forecast_x - 6, forecast_y - 6, forecast_x + 6, forecast_y + 6), fill="#f4d35e")
    draw.text((left, 306), "orange: extracted  ·  teal: filtered  ·  yellow: next", fill="#d8e6e1", font=_font(14))
    return chart


def _summary_panel(forecast: float, last_points: int, sample_count: int) -> Image.Image:
    size = (560, 340)
    panel = Image.new("RGB", size, "#176b87")
    draw = ImageDraw.Draw(panel)
    draw.text((28, 28), "Image-based estimate", fill="#a8d7d9", font=_font(18))
    draw.text((28, 75), f"{forecast:.4f}", fill="#ffffff", font=_font(66))
    draw.text((30, 150), "next plotted value", fill="#d8e6e1", font=_font(18))
    lines = [
        f"Samples digitized: {sample_count}",
        f"Trend window: {last_points} points",
        "Filter: orange RGB segmentation + smoothing",
        "Forecast: linear one-step extrapolation",
    ]
    for index, line in enumerate(lines):
        draw.text((30, 205 + index * 25), line, fill="#d8e6e1", font=_font(15))
    return panel


def create_report(
    input_path: Path,
    output_path: Path,
    last_points: int = 5,
    tolerance: float = 70,
    calibration: dict[str, float | int] | None = None,
) -> dict[str, Any]:
    """Create a graph-filtering report and return the extracted forecast."""

    settings = {**DEFAULT_CALIBRATION, **(calibration or {})}
    with Image.open(input_path) as source:
        image = source.convert("RGB")

    sample_x, pixel_y, mask = extract_trace(image, tolerance=tolerance, calibration=settings)
    values = pixels_to_values(pixel_y, settings)
    smoothed = smooth_trace(values)
    forecast = predict_next(smoothed, last_points)
    y_min, y_max = float(settings["y_min"]), float(settings["y_max"])

    mask_image = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    panels = [
        _panel("Original inflation graph", image, (560, 340)),
        _panel("Orange-line filter", mask_image, (560, 340)),
        _panel(
            "Filtered trace + forecast",
            _series_chart(values, smoothed, forecast, y_min, y_max),
            (560, 340),
        ),
        _summary_panel(forecast, last_points, len(values)),
    ]
    margin, gap = 24, 18
    canvas = Image.new("RGB", (margin * 2 + 560 * 2 + gap, margin * 2 + 340 * 2 + gap), "#f7f5ef")
    for index, panel in enumerate(panels):
        x = margin + (index % 2) * (560 + gap)
        y = margin + (index // 2) * (340 + gap)
        canvas.paste(panel, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, optimize=True)
    return {
        "input": str(input_path),
        "output": str(output_path),
        "samples": len(values),
        "last_points": last_points,
        "forecast": round(forecast, 4),
        "min_observed": round(float(values.min()), 4),
        "max_observed": round(float(values.max()), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter an inflation graph image and extrapolate its next plotted value.")
    parser.add_argument("--input", type=Path, required=True, help="Graph image to digitize")
    parser.add_argument("--output", type=Path, required=True, help="Generated analysis report PNG")
    parser.add_argument("--last-points", type=int, default=5, help="Number of final samples used for extrapolation")
    parser.add_argument("--tolerance", type=float, default=70, help="RGB distance tolerance for the orange line")
    args = parser.parse_args()

    summary = create_report(args.input, args.output, args.last_points, args.tolerance)
    print("Inflation graph image prediction report created")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
