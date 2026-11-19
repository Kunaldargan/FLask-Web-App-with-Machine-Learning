# Computer-vision experiments

This folder contains small, reproducible image-processing experiments that complement the inflation-regression demo without becoming part of the Flask request path.

## Inspection sheet

The first experiment produces six views of an image:

1. Original RGB image
2. Luminance/grayscale conversion
3. Binary threshold mask
4. Sobel gradient magnitude
5. RGB channel histograms
6. A high-contrast edge mask

Run it from the project root:

```bash
python -m experiments.computer_vision \
  --input static/images/cv-fruit-public-domain.jpg \
  --output docs/images/computer-vision-report.png
```

Try another threshold to see how segmentation changes:

```bash
python -m experiments.computer_vision \
  --input static/images/cv-fruit-public-domain.jpg \
  --output /tmp/fruit-threshold-180.png \
  --threshold 180
```

The implementation uses Pillow and NumPy only. It is intended for learning and visual inspection, not as a replacement for a production image-processing pipeline.

## Filter the inflation graph and predict

The original KNN comparison image, `static/images/knnPlot1.png`, contains an orange inflation trace. `graph_image_predictor.py` applies a color-distance filter to isolate that line, interpolates gaps caused by anti-aliasing, smooths the pixel trace, converts it through the chart calibration, and extrapolates one next value from the final five samples.

```bash
python -m experiments.graph_image_predictor \
  --input static/images/knnPlot1.png \
  --output docs/images/inflation-graph-filter-prediction.png
```

The output report shows the original graph, filtered line mask, extracted/filtered trace, and forecast summary. The default run produces an estimate of approximately `0.3141` for this static image. Because the input is a screenshot, the result is sensitive to axis bounds, line color, resolution, and crop; treat it as a graph-digitization experiment, not a production forecast.

## 1D CNN over time-series data

The project also includes an optional temporal convolutional network for the original economic time-series shape. Each training example is a sliding window of consecutive rows with shape `[window, features]`; causal `Conv1D` layers learn local temporal patterns and a regression head predicts the next target value. The split is chronological and `shuffle=False` is used during training to avoid leaking future ordering into the experiment.

Install the optional deep-learning dependencies:

```bash
pip install -r requirements-cnn.txt
```

Then point the experiment at the original CSV when it is available:

```bash
python -m experiments.time_series_cnn \
  --data /path/to/pro.csv \
  --time-column Year \
  --target INFLATION \
  --window-size 5 \
  --output-dir artifacts/time-series-cnn
```

The run writes a `.keras` model, `predictions.csv`, and `metrics.json`. TensorFlow is optional and is not imported by the Flask application.
