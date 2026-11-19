"""Dependency-light computer-vision preprocessing experiment.

The experiment turns one image into an inspection sheet containing:

* the original RGB image;
* grayscale luminance;
* a configurable binary threshold;
* Sobel gradient magnitude; and
* RGB channel histograms.

It is deliberately deterministic and uses Pillow + NumPy so it can run on the
same small environment as the web application without requiring OpenCV.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


RESAMPLE = getattr(getattr(Image, "Resampling", Image), "LANCZOS")


def grayscale_array(image: Image.Image) -> np.ndarray:
    """Return an image's luminance channel as a float array in [0, 255]."""

    red, green, blue = np.asarray(image.convert("RGB"), dtype=np.float32).transpose(2, 0, 1)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def threshold_array(gray: np.ndarray, threshold: int) -> np.ndarray:
    """Create a binary image from a grayscale array."""

    return np.where(gray >= threshold, 255, 0).astype(np.uint8)


def sobel_array(gray: np.ndarray) -> np.ndarray:
    """Compute normalized Sobel gradient magnitude without a CV framework."""

    padded = np.pad(gray, 1, mode="reflect")
    horizontal = (
        padded[:-2, 2:]
        + 2 * padded[1:-1, 2:]
        + padded[2:, 2:]
        - padded[:-2, :-2]
        - 2 * padded[1:-1, :-2]
        - padded[2:, :-2]
    )
    vertical = (
        padded[2:, :-2]
        + 2 * padded[2:, 1:-1]
        + padded[2:, 2:]
        - padded[:-2, :-2]
        - 2 * padded[:-2, 1:-1]
        - padded[:-2, 2:]
    )
    magnitude = np.hypot(horizontal, vertical)
    maximum = float(magnitude.max())
    if maximum == 0:
        return np.zeros_like(magnitude, dtype=np.uint8)
    return np.clip(magnitude / maximum * 255, 0, 255).astype(np.uint8)


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return ImageOps.contain(image.convert("RGB"), size, RESAMPLE)


def _panel(title: str, image: Image.Image, size: tuple[int, int]) -> Image.Image:
    panel = Image.new("RGB", size, "#ffffff")
    draw = ImageDraw.Draw(panel)
    draw.rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), radius=14, outline="#d8e2df", width=2)
    draw.text((20, 16), title, fill="#173238", font=_font(22))
    fitted = _fit(image, (size[0] - 40, size[1] - 72))
    panel.paste(fitted, ((size[0] - fitted.width) // 2, 58 + (size[1] - 58 - fitted.height) // 2))
    return panel


def _histogram_panel(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    panel = Image.new("RGB", size, "#ffffff")
    draw = ImageDraw.Draw(panel)
    draw.rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), radius=14, outline="#d8e2df", width=2)
    draw.text((20, 16), "RGB channel histogram", fill="#173238", font=_font(22))

    pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    left, top = 30, 62
    right, bottom = size[0] - 24, size[1] - 28
    draw.rectangle((left, top, right, bottom), outline="#d8e2df", width=1)
    colors = ((0, "#d85b4f"), (1, "#45a879"), (2, "#3f83bd"))
    for channel, color in colors:
        histogram, _ = np.histogram(pixels[:, :, channel], bins=64, range=(0, 256))
        peak = max(int(histogram.max()), 1)
        points = []
        for index, value in enumerate(histogram):
            x = left + index * (right - left) / (len(histogram) - 1)
            y = bottom - int(value / peak * (bottom - top - 4))
            points.append((x, y))
        draw.line(points, fill=color, width=3)
    return panel


def create_report(input_path: Path, output_path: Path, threshold: int = 128) -> dict[str, float | int | str]:
    """Create a six-panel analysis sheet and return summary statistics."""

    if not 0 <= threshold <= 255:
        raise ValueError("threshold must be between 0 and 255")

    with Image.open(input_path) as source:
        image = source.convert("RGB")

    gray = grayscale_array(image)
    thresholded = threshold_array(gray, threshold)
    edges = sobel_array(gray)
    tile_size = (390, 260)
    panels: Iterable[Image.Image] = (
        _panel("Original", image, tile_size),
        _panel("Grayscale", Image.fromarray(gray.astype(np.uint8), mode="L"), tile_size),
        _panel(f"Threshold ≥ {threshold}", Image.fromarray(thresholded, mode="L"), tile_size),
        _panel("Sobel edges", Image.fromarray(edges, mode="L"), tile_size),
        _histogram_panel(image, tile_size),
        _panel("Edge mask", Image.fromarray(np.where(edges > 40, 255, 0).astype(np.uint8), mode="L"), tile_size),
    )

    margin, gap = 24, 18
    canvas = Image.new("RGB", (margin * 2 + tile_size[0] * 3 + gap * 2, margin * 2 + tile_size[1] * 2 + gap), "#f7f5ef")
    for index, panel in enumerate(panels):
        x = margin + (index % 3) * (tile_size[0] + gap)
        y = margin + (index // 3) * (tile_size[1] + gap)
        canvas.paste(panel, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, optimize=True)
    return {
        "input": str(input_path),
        "output": str(output_path),
        "width": image.width,
        "height": image.height,
        "mean_luminance": round(float(gray.mean()), 2),
        "edge_density": round(float((edges > 40).mean()), 4),
        "threshold": threshold,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a computer-vision inspection sheet from one image.")
    parser.add_argument("--input", type=Path, required=True, help="Path to an input image")
    parser.add_argument("--output", type=Path, required=True, help="Path for the generated PNG report")
    parser.add_argument("--threshold", type=int, default=128, help="Binary threshold from 0 to 255 (default: 128)")
    args = parser.parse_args()

    summary = create_report(args.input, args.output, args.threshold)
    print("Computer-vision report created")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
