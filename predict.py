"""
SalesCode AI - Spot the Fake Photo
Usage:
    python predict.py path/to/image.jpg
Output:
    One number between 0 and 1
    0 = real photo
    1 = photo of screen / recapture
"""

import sys
import os
import json
import math
import numpy as np
import cv2

MODEL_PATH = os.path.join(os.path.dirname(__file__), "recapture_model.json")
IMG_SIZE = 512


def sigmoid(x):
    x = max(min(float(x), 50), -50)
    return 1.0 / (1.0 + math.exp(-x))


def safe_float(x, default=0.0):
    try:
        if np.isfinite(x):
            return float(x)
    except Exception:
        pass
    return default


def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def resize_keep_aspect(img, max_side=IMG_SIZE):
    h, w = img.shape[:2]
    scale = max_side / float(max(h, w))
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def dominant_frequency_features(gray):
    """Frequency peaks / grid-like periodicity caused by screen pixels and refresh patterns."""
    g = gray.astype(np.float32) / 255.0
    h, w = g.shape
    g = g - np.mean(g)
    win_y = np.hanning(h).reshape(-1, 1)
    win_x = np.hanning(w).reshape(1, -1)
    f = np.fft.fftshift(np.fft.fft2(g * win_y * win_x))
    mag = np.log1p(np.abs(f))

    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    mag[r < min(h, w) * 0.04] = 0  # remove low-frequency content

    total_energy = np.sum(mag) + 1e-8
    peak_energy = np.percentile(mag, 99.7)
    peak_ratio = peak_energy / (np.mean(mag) + 1e-8)

    vertical_band = mag[:, max(0, cx - 2):min(w, cx + 3)]
    horizontal_band = mag[max(0, cy - 2):min(h, cy + 3), :]
    axis_energy_ratio = (np.sum(vertical_band) + np.sum(horizontal_band)) / total_energy

    high_freq_ratio = np.sum(mag[r > min(h, w) * 0.30]) / total_energy
    return safe_float(peak_ratio), safe_float(axis_energy_ratio), safe_float(high_freq_ratio)


def color_quantization_features(img):
    """Screens often introduce posterization/quantization and unusual channel behavior."""
    small = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    unique_counts = []
    entropy_vals = []
    for c in cv2.split(small):
        hist = cv2.calcHist([c], [0], None, [64], [0, 256]).flatten()
        p = hist / (np.sum(hist) + 1e-8)
        entropy_vals.append(-np.sum(p * np.log2(p + 1e-8)))
        unique_counts.append(len(np.unique((c // 8).astype(np.uint8))))

    saturation_mean = np.mean(hsv[:, :, 1]) / 255.0
    saturation_std = np.std(hsv[:, :, 1]) / 255.0
    quant_score = 1.0 - (np.mean(unique_counts) / 32.0)
    entropy_mean = np.mean(entropy_vals) / 6.0
    return safe_float(quant_score), safe_float(entropy_mean), safe_float(saturation_mean), safe_float(saturation_std)


def glare_and_clipping_features(img):
    """Screen photos often contain glare, bright clipped regions, or dark borders."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bright_ratio = np.mean(gray > 245)
    dark_ratio = np.mean(gray < 10)

    # specular glare: bright pixels with low saturation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    glare_ratio = np.mean((hsv[:, :, 2] > 235) & (hsv[:, :, 1] < 45))

    # rectangular border/edge energy can appear when a screen is visible
    edges = cv2.Canny(gray, 80, 160)
    h, w = gray.shape
    border = np.zeros_like(edges, dtype=bool)
    pad_y, pad_x = max(2, h // 25), max(2, w // 25)
    border[:pad_y, :] = True
    border[-pad_y:, :] = True
    border[:, :pad_x] = True
    border[:, -pad_x:] = True
    border_edge_ratio = np.mean(edges[border] > 0) / (np.mean(edges > 0) + 1e-8)
    return safe_float(bright_ratio), safe_float(dark_ratio), safe_float(glare_ratio), safe_float(border_edge_ratio)


def sharpness_texture_features(gray):
    """Texture statistics. Recaptured photos often have abnormal high-frequency/noise patterns."""
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    sharpness = np.var(lap) / 10000.0

    # local binary-like contrast using gradients
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    grad_mean = np.mean(grad) / 255.0
    grad_std = np.std(grad) / 255.0

    # banding: repeated row/column intensity changes
    row_mean = gray.mean(axis=1)
    col_mean = gray.mean(axis=0)
    row_banding = np.std(np.diff(row_mean)) / 255.0
    col_banding = np.std(np.diff(col_mean)) / 255.0
    return safe_float(sharpness), safe_float(grad_mean), safe_float(grad_std), safe_float(row_banding), safe_float(col_banding)


def extract_features(image_path):
    img = resize_keep_aspect(load_image(image_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    features = []
    features.extend(dominant_frequency_features(gray))
    features.extend(color_quantization_features(img))
    features.extend(glare_and_clipping_features(img))
    features.extend(sharpness_texture_features(gray))
    return np.array(features, dtype=np.float32)


FEATURE_NAMES = [
    "freq_peak_ratio", "axis_energy_ratio", "high_freq_ratio",
    "quant_score", "entropy_mean", "saturation_mean", "saturation_std",
    "bright_ratio", "dark_ratio", "glare_ratio", "border_edge_ratio",
    "sharpness", "grad_mean", "grad_std", "row_banding", "col_banding"
]


def heuristic_score(x):
    """Fallback if recapture_model.json is missing. Works as a baseline only."""
    d = dict(zip(FEATURE_NAMES, x))
    score = 0.0
    score += 0.85 * min(d["freq_peak_ratio"] / 35.0, 2.0)
    score += 1.20 * min(d["axis_energy_ratio"] / 0.11, 2.0)
    score += 0.90 * min(d["high_freq_ratio"] / 0.45, 2.0)
    score += 0.65 * min(d["quant_score"] / 0.30, 2.0)
    score += 0.65 * min(d["glare_ratio"] / 0.05, 2.0)
    score += 0.35 * min((d["row_banding"] + d["col_banding"]) / 0.030, 2.0)
    score += 0.25 * min(d["border_edge_ratio"] / 2.5, 2.0)
    # Normalize around empirical midpoint
    return sigmoid((score - 3.2) * 1.3)


def model_score(x):
    if not os.path.exists(MODEL_PATH):
        return heuristic_score(x)
    try:
        with open(MODEL_PATH, "r", encoding="utf-8") as f:
            model = json.load(f)
        mean = np.array(model["mean"], dtype=np.float32)
        scale = np.array(model["scale"], dtype=np.float32)
        coef = np.array(model["coef"], dtype=np.float32)
        intercept = float(model["intercept"])
        z = (x - mean) / (scale + 1e-8)
        return sigmoid(np.dot(z, coef) + intercept)
    except Exception:
        return heuristic_score(x)


def predict(image_path):
    features = extract_features(image_path)
    score = model_score(features)
    return max(0.0, min(1.0, float(score)))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py image.jpg", file=sys.stderr)
        sys.exit(1)
    try:
        print(f"{predict(sys.argv[1]):.6f}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)
