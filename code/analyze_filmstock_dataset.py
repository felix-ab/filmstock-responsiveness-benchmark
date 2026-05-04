#!/usr/bin/env python3
"""Extract statistical analysis data from the filmstock benchmark image run.

The script reads the generation manifest, validates every normalized PNG, turns
each image into numeric features, and writes analysis-ready CSV files.
"""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image
from scipy.stats import chi2, t, ttest_1samp, wilcoxon
from skimage import color as skcolor
from skimage.measure import shannon_entropy
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = Path(
    os.environ.get(
        "FILMSTOCK_RUN_DIR",
        REPO_ROOT / "data" / "generated" / "filmstock_20260503_005006_f42a60",
    )
)
MANIFEST_PATH = RUN_DIR / "manifest.csv"
ANALYSIS_DIR = Path(os.environ.get("FILMSTOCK_ANALYSIS_DIR", RUN_DIR / "analysis"))
PLOTS_DIR = ANALYSIS_DIR / "plots"

EXPECTED_WIDTH = 1024
EXPECTED_HEIGHT = 1024

ID_COLUMNS = [
    "status",
    "model_key",
    "provider",
    "api_model",
    "scene_key",
    "film_key",
    "light_key",
    "scan_key",
    "replicate",
    "variant_id",
    "condition_id",
    "prompt_id",
    "image_path",
    "width",
    "height",
    "mode",
    "prompt",
    "started_at",
    "completed_at",
]

FACTOR_COLUMNS = [
    "model_xai_grok_imagine",
    "film_cinestill800t",
    "light_warm_practical",
    "scan_pushed",
    "scene_apartment",
    "scene_backstage",
    "scene_corner_store",
]

MAIN_EFFECTS = {
    "filmstock_cinestill_minus_portra": {
        "vary": "film_key",
        "low": "portra400",
        "high": "cinestill800t",
        "match": ["model_key", "scene_key", "light_key", "scan_key", "replicate"],
    },
    "lighting_warm_minus_cool": {
        "vary": "light_key",
        "low": "cool_ambient",
        "high": "warm_practical",
        "match": ["model_key", "scene_key", "film_key", "scan_key", "replicate"],
    },
    "scan_pushed_minus_clean": {
        "vary": "scan_key",
        "low": "clean_scan",
        "high": "pushed_scan",
        "match": ["model_key", "scene_key", "film_key", "light_key", "replicate"],
    },
}

EFFECT_FEATURES = [
    "rgb_mean_r",
    "rgb_mean_g",
    "rgb_mean_b",
    "lab_mean_l",
    "lab_mean_a",
    "lab_mean_b",
    "saturation_mean",
    "warmth_rgb_mean_r_minus_b",
    "luma_mean",
    "luma_sd",
    "rms_contrast",
    "shadow_fraction",
    "highlight_fraction",
    "edge_density_canny",
    "gradient_mag_mean",
    "texture_highpass_std",
    "texture_highpass_mad",
    "laplacian_variance",
    "image_entropy_gray",
    "color_spatial_var_lab_l",
    "color_spatial_var_lab_a",
    "color_spatial_var_lab_b",
    "halation_ring_red_excess_mean",
    "halation_index_ring_minus_background",
    "red_highlight_excess_mean",
    "local_lab_cov_l_a_abs_mean",
    "local_lab_cov_l_b_abs_mean",
    "local_lab_cov_a_b_abs_mean",
    "hue_circular_variance",
    "hue_resultant_length",
    "hue_warm_share",
    "hue_teal_cyan_share",
    "hue_red_orange_highlight_share",
    "shadow_warmth_rgb_r_minus_b",
    "highlight_warmth_rgb_r_minus_b",
    "highlight_minus_shadow_warmth",
    "highlight_minus_shadow_lab_b",
    "split_tone_distance_lab_ab",
    "luma_skewness",
    "luma_kurtosis_excess",
    "highlight_rolloff_ratio",
    "shadow_compression_ratio",
    "fft_high_freq_power_share",
    "fft_high_mid_power_ratio",
    "fft_radial_loglog_slope",
    "palette_entropy_6",
    "palette_effective_clusters_6",
    "palette_top2_lab_distance",
    "palette_mean_pairwise_lab_distance",
]

COLOR_RESPONSE_FEATURES = [
    "rgb_mean_r",
    "rgb_mean_g",
    "rgb_mean_b",
    "lab_mean_a",
    "lab_mean_b",
    "saturation_mean",
    "warmth_rgb_mean_r_minus_b",
    "color_spatial_var_lab_a",
    "color_spatial_var_lab_b",
    "halation_ring_red_excess_mean",
    "halation_index_ring_minus_background",
    "red_highlight_excess_mean",
    "local_lab_cov_l_a_abs_mean",
    "local_lab_cov_l_b_abs_mean",
    "local_lab_cov_a_b_abs_mean",
    "hue_circular_variance",
    "hue_resultant_length",
    "hue_warm_share",
    "hue_teal_cyan_share",
    "hue_red_orange_highlight_share",
    "shadow_warmth_rgb_r_minus_b",
    "highlight_warmth_rgb_r_minus_b",
    "highlight_minus_shadow_warmth",
    "highlight_minus_shadow_lab_a",
    "highlight_minus_shadow_lab_b",
    "split_tone_distance_lab_ab",
    "palette_entropy_6",
    "palette_effective_clusters_6",
    "palette_dominant_lab_a",
    "palette_dominant_lab_b",
    "palette_top2_lab_distance",
    "palette_mean_pairwise_lab_distance",
]

CONTRAST_TEXTURE_FEATURES = [
    "lab_mean_l",
    "luma_mean",
    "luma_sd",
    "rms_contrast",
    "dynamic_range_p95_p05",
    "shadow_fraction",
    "highlight_fraction",
    "texture_highpass_std",
    "texture_highpass_mad",
    "local_contrast_mean",
    "laplacian_variance",
    "luma_skewness",
    "luma_kurtosis_excess",
    "highlight_rolloff_ratio",
    "shadow_compression_ratio",
    "upper_lower_luma_spread_ratio",
    "fft_low_freq_power_share",
    "fft_mid_freq_power_share",
    "fft_high_freq_power_share",
    "fft_high_mid_power_ratio",
    "fft_high_low_power_ratio",
    "fft_radial_loglog_slope",
    "fft_high_freq_log_power_mean",
]

STRUCTURE_FEATURES = [
    "edge_density_canny",
    "gradient_mag_mean",
    "gradient_mag_p95",
    "orientation_entropy",
    "image_entropy_gray",
]


def safe_float(value: float) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return float("nan")
    return float(value)


def array_stats(prefix: str, values: np.ndarray) -> dict[str, float]:
    flat = values.reshape(-1).astype(np.float64)
    return {
        f"{prefix}_mean": safe_float(np.mean(flat)),
        f"{prefix}_sd": safe_float(np.std(flat, ddof=1)),
        f"{prefix}_p01": safe_float(np.percentile(flat, 1)),
        f"{prefix}_p05": safe_float(np.percentile(flat, 5)),
        f"{prefix}_p50": safe_float(np.percentile(flat, 50)),
        f"{prefix}_p95": safe_float(np.percentile(flat, 95)),
        f"{prefix}_p99": safe_float(np.percentile(flat, 99)),
    }


def covariance_features(prefix: str, channels: np.ndarray, names: list[str]) -> dict[str, float]:
    flat = channels.reshape(-1, len(names)).astype(np.float64)
    cov = np.cov(flat, rowvar=False, ddof=1)
    corr = np.corrcoef(flat, rowvar=False)
    out: dict[str, float] = {}
    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            if j <= i:
                continue
            out[f"{prefix}_cov_{name_i}_{name_j}"] = safe_float(cov[i, j])
            out[f"{prefix}_corr_{name_i}_{name_j}"] = safe_float(corr[i, j])
    return out


def mean_or_nan(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return safe_float(np.mean(values))


def spatial_grid_features(rgb: np.ndarray, lab: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    h, w, _ = rgb.shape
    rgb_cells: list[np.ndarray] = []
    lab_cells: list[np.ndarray] = []
    for y0 in np.linspace(0, h, 5, dtype=int)[:-1]:
        y1 = y0 + h // 4
        if y0 == h - h // 4:
            y1 = h
        for x0 in np.linspace(0, w, 5, dtype=int)[:-1]:
            x1 = x0 + w // 4
            if x0 == w - w // 4:
                x1 = w
            rgb_cells.append(np.mean(rgb[y0:y1, x0:x1], axis=(0, 1)))
            lab_cells.append(np.mean(lab[y0:y1, x0:x1], axis=(0, 1)))

    rgb_means = np.vstack(rgb_cells)
    lab_means = np.vstack(lab_cells)
    for i, ch in enumerate(["r", "g", "b"]):
        out[f"color_spatial_var_rgb_{ch}"] = safe_float(np.var(rgb_means[:, i], ddof=1))
        out[f"color_spatial_range_rgb_{ch}"] = safe_float(np.max(rgb_means[:, i]) - np.min(rgb_means[:, i]))
    for i, ch in enumerate(["l", "a", "b"]):
        out[f"color_spatial_var_lab_{ch}"] = safe_float(np.var(lab_means[:, i], ddof=1))
        out[f"color_spatial_range_lab_{ch}"] = safe_float(np.max(lab_means[:, i]) - np.min(lab_means[:, i]))
    return out


def halation_features(rgb: np.ndarray, luma: np.ndarray) -> dict[str, float]:
    bright_threshold = np.percentile(luma, 98.5)
    bright = luma >= bright_threshold
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    dilated = cv2.dilate(bright.astype(np.uint8), kernel, iterations=1).astype(bool)
    ring = dilated & ~bright
    background = ~dilated

    red_excess = rgb[:, :, 0] - np.maximum(rgb[:, :, 1], rgb[:, :, 2])
    warm_excess = rgb[:, :, 0] - rgb[:, :, 2]
    ring_red = red_excess[ring]
    bg_red = red_excess[background]

    ring_red_mean = mean_or_nan(ring_red)
    bg_red_mean = mean_or_nan(bg_red)
    return {
        "halation_bright_threshold_luma": safe_float(bright_threshold),
        "halation_bright_fraction": safe_float(np.mean(bright)),
        "halation_ring_fraction": safe_float(np.mean(ring)),
        "red_highlight_excess_mean": mean_or_nan(red_excess[bright]),
        "warm_highlight_excess_mean": mean_or_nan(warm_excess[bright]),
        "halation_ring_red_excess_mean": ring_red_mean,
        "halation_ring_red_excess_p95": safe_float(np.percentile(ring_red, 95)) if ring_red.size else float("nan"),
        "halation_background_red_excess_mean": bg_red_mean,
        "halation_index_ring_minus_background": safe_float(ring_red_mean - bg_red_mean),
    }


def summarize_vector(prefix: str, values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_sd": float("nan"),
            f"{prefix}_abs_mean": float("nan"),
            f"{prefix}_p90_abs": float("nan"),
        }
    return {
        f"{prefix}_mean": safe_float(np.mean(arr)),
        f"{prefix}_sd": safe_float(np.std(arr, ddof=1)) if arr.size > 1 else float("nan"),
        f"{prefix}_abs_mean": safe_float(np.mean(np.abs(arr))),
        f"{prefix}_p90_abs": safe_float(np.percentile(np.abs(arr), 90)),
    }


def local_color_covariance_features(lab: np.ndarray, grid: int = 8) -> dict[str, float]:
    """Summarize patchwise Lab covariance/correlation across an 8 by 8 grid."""
    h, w, _ = lab.shape
    ys = np.array_split(np.arange(h), grid)
    xs = np.array_split(np.arange(w), grid)
    cov_values: dict[str, list[float]] = {"l_a": [], "l_b": [], "a_b": []}
    corr_values: dict[str, list[float]] = {"l_a": [], "l_b": [], "a_b": []}

    for y_idx in ys:
        for x_idx in xs:
            patch = lab[np.ix_(y_idx, x_idx)].reshape(-1, 3).astype(np.float64)
            if patch.shape[0] < 3:
                continue
            cov = np.cov(patch, rowvar=False, ddof=1)
            sds = np.sqrt(np.maximum(np.diag(cov), 0.0))
            denom = np.outer(sds, sds)
            corr = np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 1e-12)
            cov_values["l_a"].append(float(cov[0, 1]))
            cov_values["l_b"].append(float(cov[0, 2]))
            cov_values["a_b"].append(float(cov[1, 2]))
            corr_values["l_a"].append(float(corr[0, 1]))
            corr_values["l_b"].append(float(corr[0, 2]))
            corr_values["a_b"].append(float(corr[1, 2]))

    out: dict[str, float] = {}
    for pair in ["l_a", "l_b", "a_b"]:
        out.update(summarize_vector(f"local_lab_cov_{pair}", cov_values[pair]))
        out.update(summarize_vector(f"local_lab_corr_{pair}", corr_values[pair]))
    return out


def circular_hue_features(hsv: np.ndarray, luma: np.ndarray) -> dict[str, float]:
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    saturated = (sat > 0.14) & (val > 0.08)
    if not np.any(saturated):
        saturated = val > np.percentile(val, 5)

    h = hue[saturated].reshape(-1)
    weights = sat[saturated].reshape(-1).astype(np.float64) + 1e-6
    angles = 2 * np.pi * h
    sin_mean = np.sum(np.sin(angles) * weights) / np.sum(weights)
    cos_mean = np.sum(np.cos(angles) * weights) / np.sum(weights)
    resultant = math.sqrt(sin_mean * sin_mean + cos_mean * cos_mean)
    mean_angle = math.atan2(sin_mean, cos_mean)
    if mean_angle < 0:
        mean_angle += 2 * np.pi

    def share(mask: np.ndarray) -> float:
        denom = np.sum(saturated)
        return safe_float(np.sum(mask & saturated) / denom) if denom else float("nan")

    warm = (hue <= 0.17) | (hue >= 0.94)
    red_orange = (hue <= 0.11) | (hue >= 0.96)
    yellow = (hue > 0.11) & (hue <= 0.19)
    green = (hue > 0.22) & (hue <= 0.42)
    teal_cyan = (hue > 0.42) & (hue <= 0.56)
    blue = (hue > 0.56) & (hue <= 0.74)
    magenta = (hue > 0.78) & (hue <= 0.94)
    highlights = saturated & (luma >= np.percentile(luma, 90))
    highlight_denom = np.sum(highlights)

    return {
        "hue_circular_mean_turns": safe_float(mean_angle / (2 * np.pi)),
        "hue_resultant_length": safe_float(resultant),
        "hue_circular_variance": safe_float(1.0 - resultant),
        "hue_warm_share": share(warm),
        "hue_red_orange_share": share(red_orange),
        "hue_yellow_share": share(yellow),
        "hue_green_share": share(green),
        "hue_teal_cyan_share": share(teal_cyan),
        "hue_blue_share": share(blue),
        "hue_magenta_share": share(magenta),
        "hue_red_orange_highlight_share": safe_float(np.sum(highlights & red_orange) / highlight_denom) if highlight_denom else 0.0,
        "hue_teal_cyan_shadow_share": share(teal_cyan & (luma <= np.percentile(luma, 25))),
    }


def tone_region_features(rgb: np.ndarray, lab: np.ndarray, hsv: np.ndarray, luma: np.ndarray) -> dict[str, float]:
    p25, p40, p60, p75 = np.percentile(luma, [25, 40, 60, 75])
    masks = {
        "shadow": luma <= p25,
        "midtone": (luma >= p40) & (luma <= p60),
        "highlight": luma >= p75,
    }
    out: dict[str, float] = {}
    summaries: dict[str, dict[str, float]] = {}
    for name, mask in masks.items():
        r = rgb[:, :, 0][mask]
        g = rgb[:, :, 1][mask]
        b = rgb[:, :, 2][mask]
        l_chan = lab[:, :, 0][mask]
        a_chan = lab[:, :, 1][mask]
        b_chan = lab[:, :, 2][mask]
        sat = hsv[:, :, 1][mask]
        summaries[name] = {
            "lab_l": mean_or_nan(l_chan),
            "lab_a": mean_or_nan(a_chan),
            "lab_b": mean_or_nan(b_chan),
            "warmth": safe_float(np.mean(r) - np.mean(b)) if r.size else float("nan"),
            "saturation": mean_or_nan(sat),
            "red_excess": mean_or_nan(r - np.maximum(g, b)) if r.size else float("nan"),
        }
        for key, value in summaries[name].items():
            out[f"{name}_{key if key != 'warmth' else 'warmth_rgb_r_minus_b'}"] = value
        out[f"{name}_fraction"] = safe_float(np.mean(mask))

    out["highlight_minus_shadow_lab_a"] = safe_float(summaries["highlight"]["lab_a"] - summaries["shadow"]["lab_a"])
    out["highlight_minus_shadow_lab_b"] = safe_float(summaries["highlight"]["lab_b"] - summaries["shadow"]["lab_b"])
    out["highlight_minus_shadow_warmth"] = safe_float(summaries["highlight"]["warmth"] - summaries["shadow"]["warmth"])
    out["highlight_minus_shadow_saturation"] = safe_float(summaries["highlight"]["saturation"] - summaries["shadow"]["saturation"])
    out["split_tone_distance_lab_ab"] = safe_float(
        math.sqrt(
            (summaries["highlight"]["lab_a"] - summaries["shadow"]["lab_a"]) ** 2
            + (summaries["highlight"]["lab_b"] - summaries["shadow"]["lab_b"]) ** 2
        )
    )
    out["midtone_shadow_lab_b_delta"] = safe_float(summaries["midtone"]["lab_b"] - summaries["shadow"]["lab_b"])
    out["highlight_midtone_lab_b_delta"] = safe_float(summaries["highlight"]["lab_b"] - summaries["midtone"]["lab_b"])
    return out


def tonal_curve_features(luma: np.ndarray) -> dict[str, float]:
    flat = luma.reshape(-1).astype(np.float64)
    p01, p05, p10, p25, p50, p75, p90, p95, p99 = np.percentile(flat, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    sd = np.std(flat, ddof=1)
    z = (flat - np.mean(flat)) / max(sd, 1e-12)
    full = max(p95 - p05, 1e-12)
    return {
        "luma_skewness": safe_float(np.mean(z ** 3)),
        "luma_kurtosis_excess": safe_float(np.mean(z ** 4) - 3.0),
        "shadow_compression_ratio": safe_float((p25 - p05) / full),
        "highlight_compression_ratio": safe_float((p95 - p75) / full),
        "highlight_rolloff_ratio": safe_float((p99 - p95) / max(p95 - p50, 1e-12)),
        "shadow_rolloff_ratio": safe_float((p05 - p01) / max(p50 - p05, 1e-12)),
        "upper_lower_luma_spread_ratio": safe_float((p95 - p50) / max(p50 - p05, 1e-12)),
        "midtone_luma_width_p75_p25": safe_float(p75 - p25),
        "extreme_tail_luma_width": safe_float(p99 - p01),
        "lower_tail_luma_width_p10_p01": safe_float(p10 - p01),
        "upper_tail_luma_width_p99_p90": safe_float(p99 - p90),
    }


def frequency_domain_features(gray: np.ndarray) -> dict[str, float]:
    small = cv2.resize(gray, (512, 512), interpolation=cv2.INTER_AREA).astype(np.float64)
    small = small - np.mean(small)
    window = np.outer(np.hanning(small.shape[0]), np.hanning(small.shape[1]))
    spectrum = np.fft.fftshift(np.fft.fft2(small * window))
    power = np.abs(spectrum) ** 2
    h, w = small.shape
    yy, xx = np.indices((h, w))
    radius = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2) / (min(h, w) / 2)
    valid = (radius > 0.015) & (radius <= 1.0)
    total = float(np.sum(power[valid])) + 1e-12
    bands = {
        "low": (radius > 0.02) & (radius <= 0.10),
        "mid": (radius > 0.10) & (radius <= 0.28),
        "high": (radius > 0.28) & (radius <= 0.65),
    }
    band_power = {name: float(np.sum(power[mask])) + 1e-12 for name, mask in bands.items()}
    centers = []
    radial_means = []
    for lo, hi in zip(np.linspace(0.035, 0.65, 18)[:-1], np.linspace(0.035, 0.65, 18)[1:]):
        mask = (radius > lo) & (radius <= hi)
        if np.any(mask):
            centers.append((lo + hi) / 2)
            radial_means.append(float(np.mean(power[mask])) + 1e-12)
    if len(centers) >= 3:
        slope = float(np.polyfit(np.log(centers), np.log(radial_means), 1)[0])
    else:
        slope = float("nan")
    return {
        "fft_low_freq_power_share": safe_float(band_power["low"] / total),
        "fft_mid_freq_power_share": safe_float(band_power["mid"] / total),
        "fft_high_freq_power_share": safe_float(band_power["high"] / total),
        "fft_high_mid_power_ratio": safe_float(band_power["high"] / band_power["mid"]),
        "fft_high_low_power_ratio": safe_float(band_power["high"] / band_power["low"]),
        "fft_radial_loglog_slope": safe_float(slope),
        "fft_low_freq_log_power_mean": safe_float(np.log(np.mean(power[bands["low"]]) + 1e-12)),
        "fft_mid_freq_log_power_mean": safe_float(np.log(np.mean(power[bands["mid"]]) + 1e-12)),
        "fft_high_freq_log_power_mean": safe_float(np.log(np.mean(power[bands["high"]]) + 1e-12)),
    }


def palette_features(lab: np.ndarray, n_clusters: int = 6) -> dict[str, float]:
    sample = lab[::16, ::16].reshape(-1, 3).astype(np.float64)
    if sample.shape[0] < n_clusters:
        return {}
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=216,
        n_init=3,
        batch_size=1024,
        reassignment_ratio=0.0,
    )
    labels = kmeans.fit_predict(sample)
    centers = kmeans.cluster_centers_
    counts = np.bincount(labels, minlength=n_clusters).astype(np.float64)
    shares = counts / counts.sum()
    order = np.argsort(shares)[::-1]
    shares = shares[order]
    centers = centers[order]
    entropy = -np.sum(shares * np.log2(shares + 1e-12))
    distances = []
    weighted_distances = []
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            dist = float(np.linalg.norm(centers[i] - centers[j]))
            distances.append(dist)
            weighted_distances.append(dist * shares[i] * shares[j])
    dominant = centers[0]
    dominant_chroma = math.sqrt(dominant[1] ** 2 + dominant[2] ** 2)
    return {
        "palette_entropy_6": safe_float(entropy),
        "palette_effective_clusters_6": safe_float(2 ** entropy),
        "palette_top1_share": safe_float(shares[0]),
        "palette_top2_share_gap": safe_float(shares[0] - shares[1]),
        "palette_top2_lab_distance": safe_float(np.linalg.norm(centers[0] - centers[1])),
        "palette_mean_pairwise_lab_distance": safe_float(np.mean(distances)),
        "palette_weighted_pairwise_lab_distance": safe_float(np.sum(weighted_distances)),
        "palette_within_cluster_lab_rmse": safe_float(math.sqrt(max(kmeans.inertia_ / sample.shape[0], 0.0))),
        "palette_dominant_lab_l": safe_float(dominant[0]),
        "palette_dominant_lab_a": safe_float(dominant[1]),
        "palette_dominant_lab_b": safe_float(dominant[2]),
        "palette_dominant_lab_chroma": safe_float(dominant_chroma),
    }


def extract_image_features(image_path: Path) -> tuple[dict[str, float], dict[str, object]]:
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        rgb8 = np.asarray(im)
        width, height = im.size
        mode = im.mode

    rgb = rgb8.astype(np.float32) / 255.0
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    gray8 = cv2.cvtColor(rgb8, cv2.COLOR_RGB2GRAY)
    gray = gray8.astype(np.float32) / 255.0
    luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
    lab = skcolor.rgb2lab(rgb)
    hsv = skcolor.rgb2hsv(rgb)

    feats: dict[str, float] = {
        "rgb_mean_r": safe_float(np.mean(r)),
        "rgb_mean_g": safe_float(np.mean(g)),
        "rgb_mean_b": safe_float(np.mean(b)),
        "rgb_sd_r": safe_float(np.std(r, ddof=1)),
        "rgb_sd_g": safe_float(np.std(g, ddof=1)),
        "rgb_sd_b": safe_float(np.std(b, ddof=1)),
        "warmth_rgb_mean_r_minus_b": safe_float(np.mean(r) - np.mean(b)),
        "warmth_rgb_mean_rg_minus_b": safe_float(((np.mean(r) + np.mean(g)) / 2.0) - np.mean(b)),
        "lab_mean_l": safe_float(np.mean(lab[:, :, 0])),
        "lab_mean_a": safe_float(np.mean(lab[:, :, 1])),
        "lab_mean_b": safe_float(np.mean(lab[:, :, 2])),
        "lab_sd_l": safe_float(np.std(lab[:, :, 0], ddof=1)),
        "lab_sd_a": safe_float(np.std(lab[:, :, 1], ddof=1)),
        "lab_sd_b": safe_float(np.std(lab[:, :, 2], ddof=1)),
        "lab_chroma_mean": safe_float(np.mean(np.sqrt(lab[:, :, 1] ** 2 + lab[:, :, 2] ** 2))),
        "saturation_mean": safe_float(np.mean(hsv[:, :, 1])),
        "saturation_sd": safe_float(np.std(hsv[:, :, 1], ddof=1)),
        "value_mean": safe_float(np.mean(hsv[:, :, 2])),
        "value_sd": safe_float(np.std(hsv[:, :, 2], ddof=1)),
    }

    feats.update(array_stats("luma", luma))
    feats["dynamic_range_p95_p05"] = safe_float(feats["luma_p95"] - feats["luma_p05"])
    feats["luma_iqr_p75_p25"] = safe_float(np.percentile(luma, 75) - np.percentile(luma, 25))
    feats["rms_contrast"] = safe_float(np.std(luma, ddof=1))
    feats["michelson_contrast_p95_p05"] = safe_float(
        (feats["luma_p95"] - feats["luma_p05"]) / max(feats["luma_p95"] + feats["luma_p05"], 1e-9)
    )
    feats["shadow_fraction"] = safe_float(np.mean(luma < 0.20))
    feats["deep_shadow_fraction"] = safe_float(np.mean(luma < 0.10))
    feats["highlight_fraction"] = safe_float(np.mean(luma > 0.85))
    feats["clipped_shadow_fraction"] = safe_float(np.mean(luma < 0.01))
    feats["clipped_highlight_fraction"] = safe_float(np.mean(luma > 0.99))

    feats.update(covariance_features("rgb", rgb, ["r", "g", "b"]))
    feats.update(covariance_features("lab", lab, ["l", "a", "b"]))

    hue = hsv[:, :, 0].reshape(-1)
    hue_hist, _ = np.histogram(hue, bins=np.linspace(0, 1, 13), density=False)
    hue_hist = hue_hist.astype(np.float64) / hue_hist.sum()
    for i, value in enumerate(hue_hist):
        feats[f"hue_hist_12bin_{i:02d}"] = safe_float(value)
    feats["hue_entropy_12bin"] = safe_float(-np.sum(hue_hist * np.log2(hue_hist + 1e-12)))

    edges = cv2.Canny(gray8, 60, 120)
    feats["edge_density_canny"] = safe_float(np.mean(edges > 0))
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    feats["gradient_mag_mean"] = safe_float(np.mean(grad))
    feats["gradient_mag_sd"] = safe_float(np.std(grad, ddof=1))
    feats["gradient_mag_p95"] = safe_float(np.percentile(grad, 95))

    edge_angle_mask = grad > np.percentile(grad, 75)
    angles = np.mod(np.arctan2(gy[edge_angle_mask], gx[edge_angle_mask]), np.pi)
    if angles.size:
        orient_hist, _ = np.histogram(angles, bins=np.linspace(0, np.pi, 10), density=False)
        orient_hist = orient_hist.astype(np.float64) / orient_hist.sum()
        feats["orientation_entropy"] = safe_float(-np.sum(orient_hist * np.log2(orient_hist + 1e-12)))
    else:
        feats["orientation_entropy"] = float("nan")

    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=2.0)
    highpass = gray - blur
    feats["texture_highpass_std"] = safe_float(np.std(highpass, ddof=1))
    feats["texture_highpass_mad"] = safe_float(np.median(np.abs(highpass - np.median(highpass))))
    feats["laplacian_variance"] = safe_float(cv2.Laplacian(gray, cv2.CV_32F).var())
    local_mean = cv2.blur(gray, (15, 15))
    local_mean_sq = cv2.blur(gray * gray, (15, 15))
    local_var = np.maximum(local_mean_sq - local_mean * local_mean, 0)
    local_sd = np.sqrt(local_var)
    feats["local_contrast_mean"] = safe_float(np.mean(local_sd))
    feats["local_contrast_p95"] = safe_float(np.percentile(local_sd, 95))
    feats["image_entropy_gray"] = safe_float(shannon_entropy(gray8))

    h, w = gray.shape
    top = slice(0, h // 3)
    bottom = slice(2 * h // 3, h)
    left = slice(0, w // 3)
    right = slice(2 * w // 3, w)
    center_y = slice(h // 3, 2 * h // 3)
    center_x = slice(w // 3, 2 * w // 3)
    edge_mask = np.ones((h, w), dtype=bool)
    edge_mask[center_y, center_x] = False
    feats["top_luma_mean"] = safe_float(np.mean(luma[top, :]))
    feats["bottom_luma_mean"] = safe_float(np.mean(luma[bottom, :]))
    feats["top_bottom_luma_delta"] = safe_float(feats["top_luma_mean"] - feats["bottom_luma_mean"])
    feats["left_warmth_mean"] = safe_float(np.mean((r - b)[:, left]))
    feats["right_warmth_mean"] = safe_float(np.mean((r - b)[:, right]))
    feats["left_right_warmth_delta"] = safe_float(feats["left_warmth_mean"] - feats["right_warmth_mean"])
    feats["center_luma_mean"] = safe_float(np.mean(luma[center_y, center_x]))
    feats["edge_luma_mean"] = safe_float(np.mean(luma[edge_mask]))
    feats["center_edge_luma_delta"] = safe_float(feats["center_luma_mean"] - feats["edge_luma_mean"])

    feats.update(spatial_grid_features(rgb, lab))
    feats.update(halation_features(rgb, luma))
    feats.update(local_color_covariance_features(lab))
    feats.update(circular_hue_features(hsv, luma))
    feats.update(tone_region_features(rgb, lab, hsv, luma))
    feats.update(tonal_curve_features(luma))
    feats.update(frequency_domain_features(gray))
    feats.update(palette_features(lab))

    qc = {
        "width_actual": width,
        "height_actual": height,
        "mode_actual": mode,
        "is_square": width == height,
        "is_expected_size": width == EXPECTED_WIDTH and height == EXPECTED_HEIGHT,
        "has_nan_features": any(math.isnan(v) for v in feats.values()),
    }
    return feats, qc


def read_manifest() -> list[dict[str, str]]:
    with MANIFEST_PATH.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        if fieldnames is None:
            fieldnames = []
    elif fieldnames is None:
        seen: list[str] = []
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.append(key)
        fieldnames = seen
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def add_factor_columns(row: dict[str, object]) -> None:
    row["model_xai_grok_imagine"] = int(row["model_key"] == "xai_grok_imagine")
    row["film_cinestill800t"] = int(row["film_key"] == "cinestill800t")
    row["light_warm_practical"] = int(row["light_key"] == "warm_practical")
    row["scan_pushed"] = int(row["scan_key"] == "pushed_scan")
    row["scene_apartment"] = int(row["scene_key"] == "apartment")
    row["scene_backstage"] = int(row["scene_key"] == "backstage")
    row["scene_corner_store"] = int(row["scene_key"] == "corner_store")


def clean_and_extract() -> tuple[list[dict[str, object]], list[str], dict[str, object]]:
    manifest_rows = read_manifest()
    feature_rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []
    feature_names: list[str] = []

    for i, src in enumerate(manifest_rows, start=1):
        image_path = Path(src["image_path"])
        qc_record: dict[str, object] = {
            "row_number": i,
            "variant_id": src.get("variant_id", ""),
            "image_path": str(image_path),
            "manifest_status": src.get("status", ""),
            "file_exists": image_path.exists(),
            "readable": False,
        }
        if src.get("status") != "success" or not image_path.exists():
            qc_rows.append(qc_record)
            continue

        feats, qc = extract_image_features(image_path)
        if not feature_names:
            feature_names = list(feats.keys())

        row: dict[str, object] = {col: src.get(col, "") for col in ID_COLUMNS}
        add_factor_columns(row)
        row["width"] = int(src["width"])
        row["height"] = int(src["height"])
        row["replicate"] = int(src["replicate"])
        row.update(feats)
        feature_rows.append(row)

        qc_record.update(qc)
        qc_record["readable"] = True
        qc_rows.append(qc_record)

    write_csv(ANALYSIS_DIR / "qc_images.csv", qc_rows)
    qc_report = {
        "run_dir": str(RUN_DIR),
        "manifest_rows": len(manifest_rows),
        "feature_rows": len(feature_rows),
        "missing_files": sum(1 for row in qc_rows if not row["file_exists"]),
        "unreadable_files": sum(1 for row in qc_rows if row["file_exists"] and not row["readable"]),
        "non_success_manifest_rows": sum(1 for row in manifest_rows if row.get("status") != "success"),
        "unexpected_size_count": sum(1 for row in qc_rows if row.get("readable") and not row.get("is_expected_size")),
        "non_square_count": sum(1 for row in qc_rows if row.get("readable") and not row.get("is_square")),
        "images_with_nan_features": sum(1 for row in qc_rows if row.get("has_nan_features")),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    (ANALYSIS_DIR / "qc_report.json").write_text(json.dumps(qc_report, indent=2))
    return feature_rows, feature_names, qc_report


def numeric_matrix(rows: list[dict[str, object]], feature_names: list[str]) -> np.ndarray:
    mat = []
    for row in rows:
        mat.append([float(row[name]) for name in feature_names])
    arr = np.array(mat, dtype=np.float64)
    col_means = np.nanmean(arr, axis=0)
    inds = np.where(np.isnan(arr))
    arr[inds] = np.take(col_means, inds[1])
    return arr


def summarize_conditions(rows: list[dict[str, object]], feature_names: list[str]) -> list[dict[str, object]]:
    group_keys = ["model_key", "provider", "api_model", "scene_key", "film_key", "light_key", "scan_key", "condition_id"]
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in group_keys)].append(row)

    out: list[dict[str, object]] = []
    for key, items in sorted(groups.items()):
        record: dict[str, object] = {group_keys[i]: key[i] for i in range(len(group_keys))}
        record["n_images"] = len(items)
        for feature in feature_names:
            vals = [float(item[feature]) for item in items if not math.isnan(float(item[feature]))]
            record[f"{feature}_mean"] = safe_float(statistics.fmean(vals)) if vals else float("nan")
            record[f"{feature}_sd"] = safe_float(statistics.stdev(vals)) if len(vals) > 1 else float("nan")
        out.append(record)
    return out


def row_index(rows: list[dict[str, object]], keys: Iterable[str]) -> dict[tuple[object, ...], dict[str, object]]:
    index: dict[tuple[object, ...], dict[str, object]] = {}
    for row in rows:
        index[tuple(row[key] for key in keys)] = row
    return index


def paired_main_effects(rows: list[dict[str, object]], feature_names: list[str]) -> list[dict[str, object]]:
    effects: list[dict[str, object]] = []
    for effect_type, spec in MAIN_EFFECTS.items():
        vary = spec["vary"]
        low = spec["low"]
        high = spec["high"]
        match = spec["match"]
        idx = row_index(rows, [*match, vary])
        match_values = sorted({tuple(row[key] for key in match) for row in rows})
        for match_value in match_values:
            low_row = idx.get((*match_value, low))
            high_row = idx.get((*match_value, high))
            if not low_row or not high_row:
                continue
            base: dict[str, object] = {
                "effect_type": effect_type,
                "effect_family": "main_effect",
                "factor": vary,
                "low_level": low,
                "high_level": high,
                "model_key": low_row["model_key"],
                "scene_key": low_row["scene_key"],
                "film_key": low_row.get("film_key", ""),
                "light_key": low_row.get("light_key", ""),
                "scan_key": low_row.get("scan_key", ""),
                "replicate": low_row["replicate"],
                "low_variant_id": low_row["variant_id"],
                "high_variant_id": high_row["variant_id"],
            }
            for feature in feature_names:
                effects.append(
                    {
                        **base,
                        "feature": feature,
                        "effect_value": safe_float(float(high_row[feature]) - float(low_row[feature])),
                    }
                )
    return effects


def condition_mean_map(rows: list[dict[str, object]], feature_names: list[str]) -> dict[tuple[str, str, str, str, str], dict[str, float]]:
    groups: dict[tuple[str, str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["model_key"]),
            str(row["scene_key"]),
            str(row["film_key"]),
            str(row["light_key"]),
            str(row["scan_key"]),
        )
        groups[key].append(row)
    means: dict[tuple[str, str, str, str, str], dict[str, float]] = {}
    for key, items in groups.items():
        means[key] = {
            feature: safe_float(np.mean([float(item[feature]) for item in items]))
            for feature in feature_names
        }
    return means


def interaction_effects(rows: list[dict[str, object]], feature_names: list[str]) -> list[dict[str, object]]:
    means = condition_mean_map(rows, feature_names)
    models = sorted({str(row["model_key"]) for row in rows})
    scenes = sorted({str(row["scene_key"]) for row in rows})
    films = ["portra400", "cinestill800t"]
    lights = ["cool_ambient", "warm_practical"]
    scans = ["clean_scan", "pushed_scan"]
    out: list[dict[str, object]] = []

    for model in models:
        for scene in scenes:
            for scan in scans:
                keys = [(model, scene, film, light, scan) for film in films for light in lights]
                if not all(key in means for key in keys):
                    continue
                for feature in feature_names:
                    value = (
                        means[(model, scene, "cinestill800t", "warm_practical", scan)][feature]
                        - means[(model, scene, "portra400", "warm_practical", scan)][feature]
                        - means[(model, scene, "cinestill800t", "cool_ambient", scan)][feature]
                        + means[(model, scene, "portra400", "cool_ambient", scan)][feature]
                    )
                    out.append(
                        {
                            "effect_type": "filmstock_by_lighting",
                            "effect_family": "two_factor_interaction",
                            "model_key": model,
                            "scene_key": scene,
                            "scan_key": scan,
                            "feature": feature,
                            "effect_value": safe_float(value),
                            "n_condition_means": 4,
                            "n_images_used": 8,
                        }
                    )

            for light in lights:
                keys = [(model, scene, film, light, scan) for film in films for scan in scans]
                if not all(key in means for key in keys):
                    continue
                for feature in feature_names:
                    value = (
                        means[(model, scene, "cinestill800t", light, "pushed_scan")][feature]
                        - means[(model, scene, "portra400", light, "pushed_scan")][feature]
                        - means[(model, scene, "cinestill800t", light, "clean_scan")][feature]
                        + means[(model, scene, "portra400", light, "clean_scan")][feature]
                    )
                    out.append(
                        {
                            "effect_type": "filmstock_by_scan",
                            "effect_family": "two_factor_interaction",
                            "model_key": model,
                            "scene_key": scene,
                            "light_key": light,
                            "feature": feature,
                            "effect_value": safe_float(value),
                            "n_condition_means": 4,
                            "n_images_used": 8,
                        }
                    )

            for film in films:
                keys = [(model, scene, film, light, scan) for light in lights for scan in scans]
                if not all(key in means for key in keys):
                    continue
                for feature in feature_names:
                    value = (
                        means[(model, scene, film, "warm_practical", "pushed_scan")][feature]
                        - means[(model, scene, film, "cool_ambient", "pushed_scan")][feature]
                        - means[(model, scene, film, "warm_practical", "clean_scan")][feature]
                        + means[(model, scene, film, "cool_ambient", "clean_scan")][feature]
                    )
                    out.append(
                        {
                            "effect_type": "lighting_by_scan",
                            "effect_family": "two_factor_interaction",
                            "model_key": model,
                            "scene_key": scene,
                            "film_key": film,
                            "feature": feature,
                            "effect_value": safe_float(value),
                            "n_condition_means": 4,
                            "n_images_used": 8,
                        }
                    )

            keys = [(model, scene, film, light, scan) for film in films for light in lights for scan in scans]
            if all(key in means for key in keys):
                for feature in feature_names:
                    value = 0.0
                    for film in films:
                        for light in lights:
                            for scan in scans:
                                sign = 1.0
                                sign *= 1.0 if film == "cinestill800t" else -1.0
                                sign *= 1.0 if light == "warm_practical" else -1.0
                                sign *= 1.0 if scan == "pushed_scan" else -1.0
                                value += sign * means[(model, scene, film, light, scan)][feature]
                    out.append(
                        {
                            "effect_type": "filmstock_by_lighting_by_scan",
                            "effect_family": "three_factor_interaction",
                            "model_key": model,
                            "scene_key": scene,
                            "feature": feature,
                            "effect_value": safe_float(value),
                            "n_condition_means": 8,
                            "n_images_used": 16,
                        }
                    )
    return out


def summarize_effects(effect_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in effect_rows:
        groups[(str(row["effect_type"]), str(row["model_key"]), str(row["feature"]))].append(float(row["effect_value"]))

    out: list[dict[str, object]] = []
    for (effect_type, model_key, feature), values in sorted(groups.items()):
        arr = np.array(values, dtype=np.float64)
        out.append(
            {
                "effect_type": effect_type,
                "model_key": model_key,
                "feature": feature,
                "n_effects": int(arr.size),
                "mean_effect": safe_float(np.mean(arr)),
                "mean_abs_effect": safe_float(np.mean(np.abs(arr))),
                "median_abs_effect": safe_float(np.median(np.abs(arr))),
                "sd_effect": safe_float(np.std(arr, ddof=1)) if arr.size > 1 else float("nan"),
                "se_effect": safe_float(np.std(arr, ddof=1) / math.sqrt(arr.size)) if arr.size > 1 else float("nan"),
            }
        )
    return out


def zscore_rows(rows: list[dict[str, object]], feature_names: list[str]) -> tuple[dict[str, dict[str, float]], dict[str, tuple[float, float]]]:
    mat = numeric_matrix(rows, feature_names)
    means = np.mean(mat, axis=0)
    sds = np.std(mat, axis=0, ddof=1)
    sds[sds == 0] = 1.0
    zmat = (mat - means) / sds
    z_by_variant = {
        str(row["variant_id"]): {feature_names[j]: safe_float(zmat[i, j]) for j in range(len(feature_names))}
        for i, row in enumerate(rows)
    }
    params = {feature_names[j]: (safe_float(means[j]), safe_float(sds[j])) for j in range(len(feature_names))}
    return z_by_variant, params


def responsiveness_distances(rows: list[dict[str, object]], feature_names: list[str]) -> list[dict[str, object]]:
    used_features = [feature for feature in EFFECT_FEATURES if feature in feature_names]
    color_features = [feature for feature in COLOR_RESPONSE_FEATURES if feature in feature_names]
    contrast_features = [feature for feature in CONTRAST_TEXTURE_FEATURES if feature in feature_names]
    structure_features = [feature for feature in STRUCTURE_FEATURES if feature in feature_names]
    z_features: list[str] = []
    for feature in [*used_features, *color_features, *contrast_features, *structure_features]:
        if feature not in z_features:
            z_features.append(feature)
    z_by_variant, _ = zscore_rows(rows, z_features)
    out: list[dict[str, object]] = []

    for effect_type, spec in MAIN_EFFECTS.items():
        vary = spec["vary"]
        low = spec["low"]
        high = spec["high"]
        match = spec["match"]
        idx = row_index(rows, [*match, vary])
        match_values = sorted({tuple(row[key] for key in match) for row in rows})
        for match_value in match_values:
            low_row = idx.get((*match_value, low))
            high_row = idx.get((*match_value, high))
            if not low_row or not high_row:
                continue
            low_z = z_by_variant[str(low_row["variant_id"])]
            high_z = z_by_variant[str(high_row["variant_id"])]

            def distance(features: list[str]) -> float:
                diff = np.array([high_z[feature] - low_z[feature] for feature in features], dtype=np.float64)
                return safe_float(np.sqrt(np.sum(diff * diff)))

            out.append(
                {
                    "effect_type": effect_type,
                    "factor": vary,
                    "low_level": low,
                    "high_level": high,
                    "model_key": low_row["model_key"],
                    "scene_key": low_row["scene_key"],
                    "film_key": low_row.get("film_key", ""),
                    "light_key": low_row.get("light_key", ""),
                    "scan_key": low_row.get("scan_key", ""),
                    "replicate": low_row["replicate"],
                    "low_variant_id": low_row["variant_id"],
                    "high_variant_id": high_row["variant_id"],
                    "distance_all_selected_features": distance(used_features),
                    "distance_color_features": distance(color_features),
                    "distance_contrast_texture_features": distance(contrast_features),
                    "distance_structure_features": distance(structure_features),
                    "n_all_features": len(used_features),
                    "n_color_features": len(color_features),
                    "n_contrast_texture_features": len(contrast_features),
                    "n_structure_features": len(structure_features),
                }
            )
    return out


def summarize_distances(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    distance_cols = [
        "distance_all_selected_features",
        "distance_color_features",
        "distance_contrast_texture_features",
        "distance_structure_features",
    ]
    groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        for col in distance_cols:
            groups[(str(row["effect_type"]), str(row["model_key"]), col)].append(float(row[col]))

    out: list[dict[str, object]] = []
    for (effect_type, model_key, distance_metric), values in sorted(groups.items()):
        arr = np.array(values, dtype=np.float64)
        out.append(
            {
                "effect_type": effect_type,
                "model_key": model_key,
                "distance_metric": distance_metric,
                "n_pairs": int(arr.size),
                "mean_distance": safe_float(np.mean(arr)),
                "median_distance": safe_float(np.median(arr)),
                "sd_distance": safe_float(np.std(arr, ddof=1)) if arr.size > 1 else float("nan"),
                "se_distance": safe_float(np.std(arr, ddof=1) / math.sqrt(arr.size)) if arr.size > 1 else float("nan"),
            }
        )
    return out


def model_comparison_tests(distance_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    distance_cols = [
        "distance_all_selected_features",
        "distance_color_features",
        "distance_contrast_texture_features",
        "distance_structure_features",
    ]
    out: list[dict[str, object]] = []
    effect_types = sorted({str(row["effect_type"]) for row in distance_rows})

    for effect_type in effect_types:
        effect_rows = [row for row in distance_rows if row["effect_type"] == effect_type]
        match_keys = sorted(
            {
                (
                    str(row["scene_key"]),
                    str(row.get("film_key", "")),
                    str(row.get("light_key", "")),
                    str(row.get("scan_key", "")),
                    str(row["replicate"]),
                )
                for row in effect_rows
            }
        )
        indexed = {
            (
                str(row["model_key"]),
                str(row["scene_key"]),
                str(row.get("film_key", "")),
                str(row.get("light_key", "")),
                str(row.get("scan_key", "")),
                str(row["replicate"]),
            ): row
            for row in effect_rows
        }
        for metric in distance_cols:
            diffs: list[float] = []
            for key in match_keys:
                gpt = indexed.get(("chatgpt_image_2", *key))
                xai = indexed.get(("xai_grok_imagine", *key))
                if gpt and xai:
                    diffs.append(float(xai[metric]) - float(gpt[metric]))
            arr = np.array(diffs, dtype=np.float64)
            if arr.size < 2:
                continue
            mean_diff = float(np.mean(arr))
            sd_diff = float(np.std(arr, ddof=1))
            se_diff = sd_diff / math.sqrt(arr.size)
            t_res = ttest_1samp(arr, popmean=0.0)
            try:
                w_res = wilcoxon(arr)
                wilcoxon_stat = safe_float(w_res.statistic)
                wilcoxon_p = safe_float(w_res.pvalue)
            except ValueError:
                wilcoxon_stat = float("nan")
                wilcoxon_p = float("nan")
            t_crit = t.ppf(0.975, df=arr.size - 1)
            out.append(
                {
                    "effect_type": effect_type,
                    "distance_metric": metric,
                    "comparison": "xai_grok_imagine_minus_chatgpt_image_2",
                    "n_paired_conditions": int(arr.size),
                    "mean_difference": safe_float(mean_diff),
                    "median_difference": safe_float(np.median(arr)),
                    "sd_difference": safe_float(sd_diff),
                    "se_difference": safe_float(se_diff),
                    "ci95_low": safe_float(mean_diff - t_crit * se_diff),
                    "ci95_high": safe_float(mean_diff + t_crit * se_diff),
                    "paired_t_statistic": safe_float(t_res.statistic),
                    "paired_t_p_value": safe_float(t_res.pvalue),
                    "cohens_dz": safe_float(mean_diff / sd_diff) if sd_diff > 0 else float("nan"),
                    "wilcoxon_statistic": wilcoxon_stat,
                    "wilcoxon_p_value": wilcoxon_p,
                }
            )
    return out


def run_pca(rows: list[dict[str, object]], feature_names: list[str]) -> None:
    pca_features = [
        feature
        for feature in feature_names
        if not feature.startswith("hue_hist_12bin_")
    ]
    mat = numeric_matrix(rows, pca_features)
    scaler = StandardScaler()
    z = scaler.fit_transform(mat)
    pca = PCA(n_components=min(10, z.shape[0], z.shape[1]))
    scores = pca.fit_transform(z)

    score_rows: list[dict[str, object]] = []
    for i, row in enumerate(rows):
        record = {col: row[col] for col in ID_COLUMNS if col in row and col != "prompt"}
        for j in range(scores.shape[1]):
            record[f"PC{j + 1}"] = safe_float(scores[i, j])
        score_rows.append(record)
    write_csv(ANALYSIS_DIR / "pca_image_scores.csv", score_rows)

    loadings_rows: list[dict[str, object]] = []
    for j in range(scores.shape[1]):
        for i, feature in enumerate(pca_features):
            loadings_rows.append(
                {
                    "component": f"PC{j + 1}",
                    "feature": feature,
                    "loading": safe_float(pca.components_[j, i]),
                }
            )
    write_csv(ANALYSIS_DIR / "pca_loadings.csv", loadings_rows)

    explained_rows = [
        {
            "component": f"PC{i + 1}",
            "explained_variance_ratio": safe_float(ratio),
            "cumulative_explained_variance_ratio": safe_float(np.sum(pca.explained_variance_ratio_[: i + 1])),
        }
        for i, ratio in enumerate(pca.explained_variance_ratio_)
    ]
    write_csv(ANALYSIS_DIR / "pca_explained_variance.csv", explained_rows)
    ellipse_rows = pca_confidence_ellipses(score_rows, ["model_key", "model_key|film_key", "model_key|light_key"])
    write_csv(ANALYSIS_DIR / "pca_confidence_ellipses.csv", ellipse_rows)
    make_pca_plot(score_rows, ellipse_rows)


def pca_confidence_ellipses(score_rows: list[dict[str, object]], group_specs: list[str]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    theta = np.linspace(0, 2 * np.pi, 160)
    circle = np.column_stack([np.cos(theta), np.sin(theta)])
    chi = chi2.ppf(0.95, df=2)

    for spec in group_specs:
        parts = spec.split("|")
        groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
        for row in score_rows:
            groups[tuple(row[p] for p in parts)].append(row)
        for key, items in sorted(groups.items()):
            if len(items) < 3:
                continue
            points = np.array([[float(row["PC1"]), float(row["PC2"])] for row in items], dtype=np.float64)
            center = points.mean(axis=0)
            cov = np.cov(points, rowvar=False, ddof=1) / len(items)
            vals, vecs = np.linalg.eigh(cov)
            vals = np.maximum(vals, 0)
            transform = vecs @ np.diag(np.sqrt(vals * chi))
            ellipse = circle @ transform.T + center
            group_label = "_".join(str(part) for part in key)
            for i, point in enumerate(ellipse):
                out.append(
                    {
                        "group_type": spec,
                        "group": group_label,
                        "n": len(items),
                        "point_index": i,
                        "PC1": safe_float(point[0]),
                        "PC2": safe_float(point[1]),
                    }
                )
    return out


def make_pca_plot(score_rows: list[dict[str, object]], ellipse_rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    colors = {"chatgpt_image_2": "#2A6FBB", "xai_grok_imagine": "#C43C39"}
    markers = {"portra400": "o", "cinestill800t": "s"}
    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
    for row in score_rows:
        ax.scatter(
            float(row["PC1"]),
            float(row["PC2"]),
            c=colors.get(str(row["model_key"]), "#555555"),
            marker=markers.get(str(row["film_key"]), "o"),
            s=32,
            alpha=0.72,
            edgecolor="white",
            linewidth=0.4,
        )
    for group in sorted({row["group"] for row in ellipse_rows if row["group_type"] == "model_key"}):
        pts = [row for row in ellipse_rows if row["group_type"] == "model_key" and row["group"] == group]
        pts = sorted(pts, key=lambda row: int(row["point_index"]))
        ax.plot(
            [float(row["PC1"]) for row in pts],
            [float(row["PC2"]) for row in pts],
            color=colors.get(group, "#333333"),
            linewidth=2,
            label=f"{group} 95% mean ellipse",
        )
    ax.axhline(0, color="#dddddd", linewidth=0.8)
    ax.axvline(0, color="#dddddd", linewidth=0.8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA of Image Feature Space")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "pca_model_confidence_ellipses.png")
    plt.close(fig)


def feature_dictionary(feature_names: list[str]) -> list[dict[str, object]]:
    descriptions = {
        "edge_density_canny": "Proportion of pixels marked as edges by Canny detection; higher values indicate more fine structural complexity.",
        "texture_highpass_std": "Standard deviation after subtracting a Gaussian blur; proxy for grain, fine texture, and high-frequency detail.",
        "texture_highpass_mad": "Median absolute deviation of high-pass texture; robust proxy for grain and fine texture.",
        "halation_index_ring_minus_background": "Red-excess difference between rings around bright regions and the background; proxy for red halation.",
        "halation_ring_red_excess_mean": "Mean red-excess near bright highlights; useful for CineStill-style halation response.",
        "red_highlight_excess_mean": "Mean red dominance inside the brightest pixels.",
        "warmth_rgb_mean_r_minus_b": "Mean red channel minus mean blue channel; positive values indicate warmer color balance.",
        "lab_mean_b": "CIELAB yellow-blue axis mean; positive values are more yellow, negative values more blue.",
        "lab_mean_a": "CIELAB red-green axis mean; positive values are more red/magenta, negative values more green.",
        "saturation_mean": "Mean HSV saturation.",
        "rms_contrast": "Standard deviation of luminance; standard RMS contrast measure.",
        "dynamic_range_p95_p05": "Difference between 95th and 5th luminance percentiles.",
        "shadow_fraction": "Proportion of pixels with luminance below 0.20.",
        "highlight_fraction": "Proportion of pixels with luminance above 0.85.",
        "color_spatial_var_lab_l": "Variance of 4 by 4 grid-cell mean CIELAB lightness; captures spatial unevenness in brightness.",
        "color_spatial_var_lab_a": "Variance of 4 by 4 grid-cell mean CIELAB red-green channel; captures spatial color arrangement.",
        "color_spatial_var_lab_b": "Variance of 4 by 4 grid-cell mean CIELAB yellow-blue channel; captures spatial color arrangement.",
        "local_lab_cov_l_a_abs_mean": "Mean absolute local covariance between Lab lightness and red-green values across 8 by 8 patches.",
        "local_lab_cov_l_b_abs_mean": "Mean absolute local covariance between Lab lightness and yellow-blue values across 8 by 8 patches.",
        "local_lab_cov_a_b_abs_mean": "Mean absolute local covariance between Lab red-green and yellow-blue values across 8 by 8 patches.",
        "hue_circular_variance": "Circular variance of saturated-pixel hue; higher values indicate more dispersed hue structure.",
        "hue_resultant_length": "Circular concentration of saturated-pixel hue; higher values indicate a tighter dominant hue.",
        "hue_warm_share": "Share of saturated pixels in warm red, orange, or yellow hue ranges.",
        "hue_teal_cyan_share": "Share of saturated pixels in teal or cyan hue ranges.",
        "hue_red_orange_highlight_share": "Share of saturated highlight pixels in red-orange hue ranges.",
        "highlight_minus_shadow_warmth": "Difference in red-minus-blue warmth between highlight and shadow tonal regions.",
        "highlight_minus_shadow_lab_b": "Difference in Lab yellow-blue position between highlight and shadow tonal regions.",
        "split_tone_distance_lab_ab": "Euclidean distance between highlight and shadow means in Lab a-b chroma space.",
        "luma_skewness": "Skewness of the luminance distribution.",
        "luma_kurtosis_excess": "Excess kurtosis of the luminance distribution.",
        "highlight_rolloff_ratio": "Upper-tail luminance compression proxy comparing extreme highlights to the upper midtone spread.",
        "shadow_compression_ratio": "Lower-tonal spread proxy comparing shadows to the main 5th-to-95th percentile range.",
        "fft_high_freq_power_share": "Share of Fourier power in high spatial frequencies; proxy for grain and fine texture.",
        "fft_high_mid_power_ratio": "Ratio of high-frequency to mid-frequency Fourier power.",
        "fft_radial_loglog_slope": "Slope of the radial Fourier power spectrum on log-log axes.",
        "palette_entropy_6": "Entropy of a six-cluster Lab color palette.",
        "palette_effective_clusters_6": "Effective number of color clusters implied by the six-cluster palette entropy.",
        "palette_top2_lab_distance": "Lab distance between the two most common palette clusters.",
        "palette_mean_pairwise_lab_distance": "Mean Lab distance among all palette cluster centers.",
    }
    rows = []
    for feature in feature_names:
        rows.append(
            {
                "feature": feature,
                "description": descriptions.get(feature, "Image-derived numeric feature for color, contrast, texture, structure, or spatial distribution."),
            }
        )
    return rows


def write_analysis_readme(qc_report: dict[str, object]) -> None:
    text = f"""# Analysis Outputs

Created: `{qc_report['created_at']}`

This folder turns the 96 generated PNGs into statistical-analysis-ready data.

## Primary Tables

- `image_features.csv`: one row per image with manifest metadata, factor indicators, and numeric image features.
- `condition_features.csv`: condition-level means and standard deviations by model, scene, film stock, lighting, and scan condition.
- `effect_pairs_long.csv`: long-format high-minus-low main effects plus factorial interaction contrasts for every numeric feature.
- `effect_summary_by_model.csv`: model-level summaries of the feature effects.
- `responsiveness_distances.csv`: paired prompt-response distances in standardized feature space.
- `responsiveness_distance_summary.csv`: model-level summaries of those response distances.
- `model_comparison_tests.csv`: paired model-comparison tests on standardized prompt-response distances.
- `pca_image_scores.csv`: PCA coordinates for each image.
- `pca_loadings.csv`: feature loadings for each PCA component.
- `pca_explained_variance.csv`: explained variance ratios.
- `pca_confidence_ellipses.csv`: 95 percent confidence ellipse coordinates for PC1 and PC2 groups.
- `feature_dictionary.csv`: concise descriptions of the extracted image features.
- `qc_images.csv` and `qc_report.json`: data-cleaning checks.

## Cleaning Result

- Manifest rows: `{qc_report['manifest_rows']}`
- Feature rows: `{qc_report['feature_rows']}`
- Missing files: `{qc_report['missing_files']}`
- Unreadable files: `{qc_report['unreadable_files']}`
- Unexpected image size: `{qc_report['unexpected_size_count']}`
- Non-square images: `{qc_report['non_square_count']}`

## Modeling Notes

Use `image_features.csv` for image-level regression, MANOVA, PCA, repeated-measures, or robust regression. Use `responsiveness_distances.csv` when the response variable should be the magnitude of stylistic change induced by a prompt factor. Use `effect_pairs_long.csv` for explicit paired contrasts such as CineStill minus Portra, warm practical minus cool ambient, pushed scan minus clean scan, and interaction effects.
"""
    (ANALYSIS_DIR / "README_ANALYSIS.md").write_text(text)


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    rows, feature_names, qc_report = clean_and_extract()
    ordered_feature_fields = [*ID_COLUMNS, *FACTOR_COLUMNS, *feature_names]
    write_csv(ANALYSIS_DIR / "image_features.csv", rows, ordered_feature_fields)
    write_csv(ANALYSIS_DIR / "feature_dictionary.csv", feature_dictionary(feature_names))

    condition_rows = summarize_conditions(rows, feature_names)
    write_csv(ANALYSIS_DIR / "condition_features.csv", condition_rows)

    main_effect_rows = paired_main_effects(rows, feature_names)
    interaction_rows = interaction_effects(rows, feature_names)
    all_effect_rows = [*main_effect_rows, *interaction_rows]
    write_csv(ANALYSIS_DIR / "effect_pairs_long.csv", all_effect_rows)
    write_csv(ANALYSIS_DIR / "effect_summary_by_model.csv", summarize_effects(all_effect_rows))

    distance_rows = responsiveness_distances(rows, feature_names)
    write_csv(ANALYSIS_DIR / "responsiveness_distances.csv", distance_rows)
    write_csv(ANALYSIS_DIR / "responsiveness_distance_summary.csv", summarize_distances(distance_rows))
    write_csv(ANALYSIS_DIR / "model_comparison_tests.csv", model_comparison_tests(distance_rows))

    run_pca(rows, feature_names)
    write_analysis_readme(qc_report)

    print(json.dumps({
        "analysis_dir": str(ANALYSIS_DIR),
        "image_feature_rows": len(rows),
        "feature_count": len(feature_names),
        "condition_rows": len(condition_rows),
        "effect_rows": len(all_effect_rows),
        "responsiveness_distance_rows": len(distance_rows),
        "qc": qc_report,
    }, indent=2))


if __name__ == "__main__":
    main()
