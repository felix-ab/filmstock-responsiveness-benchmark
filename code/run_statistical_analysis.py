#!/usr/bin/env python3
"""Run the inferential statistics and writeup visuals for the filmstock study.

This script assumes `analyze_filmstock_dataset.py` has already created the
analysis-ready CSV files. It deliberately avoids pandas and statsmodels so the
analysis is reproducible with the packages currently installed on this machine.
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.linalg import pinv
from scipy.stats import f as f_dist
from scipy.stats import friedmanchisquare, kruskal, t as t_dist
from scipy.stats import ttest_1samp, wilcoxon


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = Path(os.environ.get("FILMSTOCK_RUN_DIR", REPO_ROOT / "data" / "generated"))
ANALYSIS_DIR = Path(os.environ.get("FILMSTOCK_ANALYSIS_DIR", REPO_ROOT / "data" / "analysis"))
STATS_DIR = ANALYSIS_DIR / "stats"
FIG_DIR = STATS_DIR / "figures"


MODEL_LABELS = {
    "chatgpt_image_2": "ChatGPT Image 2",
    "xai_grok_imagine": "Grok Imagine",
}

EFFECT_LABELS = {
    "filmstock_cinestill_minus_portra": "CineStill minus Portra",
    "lighting_warm_minus_cool": "Warm practical minus cool ambient",
    "scan_pushed_minus_clean": "Pushed scan minus clean scan",
    "filmstock_by_lighting": "Filmstock by lighting",
    "filmstock_by_scan": "Filmstock by scan",
    "lighting_by_scan": "Lighting by scan",
    "filmstock_by_lighting_by_scan": "Filmstock by lighting by scan",
}

METRIC_LABELS = {
    "distance_all_selected_features": "All selected features",
    "distance_color_features": "Color features",
    "distance_contrast_texture_features": "Contrast and texture",
    "distance_structure_features": "Structure features",
}

FEATURE_LABELS = {
    "halation_index_ring_minus_background": "Halation index",
    "halation_ring_red_excess_mean": "Red ring excess",
    "red_highlight_excess_mean": "Red highlight excess",
    "warmth_rgb_mean_r_minus_b": "Warmth, red minus blue",
    "lab_mean_b": "Lab yellow minus blue",
    "lab_mean_a": "Lab red minus green",
    "saturation_mean": "Saturation",
    "rms_contrast": "RMS contrast",
    "dynamic_range_p95_p05": "Dynamic range",
    "shadow_fraction": "Shadow fraction",
    "highlight_fraction": "Highlight fraction",
    "texture_highpass_std": "High-pass texture SD",
    "texture_highpass_mad": "High-pass texture MAD",
    "laplacian_variance": "Laplacian variance",
    "edge_density_canny": "Edge density",
    "gradient_mag_mean": "Gradient magnitude",
    "local_lab_cov_l_a_abs_mean": "Local L-a covariance",
    "local_lab_cov_l_b_abs_mean": "Local L-b covariance",
    "local_lab_cov_a_b_abs_mean": "Local a-b covariance",
    "hue_circular_variance": "Hue circular variance",
    "hue_resultant_length": "Hue concentration",
    "hue_warm_share": "Warm hue share",
    "hue_teal_cyan_share": "Teal-cyan hue share",
    "hue_red_orange_highlight_share": "Red-orange highlight share",
    "shadow_warmth_rgb_r_minus_b": "Shadow warmth",
    "highlight_warmth_rgb_r_minus_b": "Highlight warmth",
    "highlight_minus_shadow_warmth": "Highlight-shadow warmth split",
    "highlight_minus_shadow_lab_a": "Highlight-shadow Lab a split",
    "highlight_minus_shadow_lab_b": "Highlight-shadow Lab b split",
    "split_tone_distance_lab_ab": "Split-tone Lab distance",
    "luma_skewness": "Luma skewness",
    "luma_kurtosis_excess": "Luma excess kurtosis",
    "highlight_rolloff_ratio": "Highlight rolloff ratio",
    "shadow_compression_ratio": "Shadow compression ratio",
    "upper_lower_luma_spread_ratio": "Upper-lower luma spread",
    "fft_high_freq_power_share": "High-frequency power share",
    "fft_high_mid_power_ratio": "High-mid frequency ratio",
    "fft_radial_loglog_slope": "Fourier radial slope",
    "palette_entropy_6": "Palette entropy",
    "palette_effective_clusters_6": "Effective palette clusters",
    "palette_top2_lab_distance": "Top-two palette distance",
    "palette_mean_pairwise_lab_distance": "Mean palette distance",
}

TARGETED_TESTS = {
    "filmstock_cinestill_minus_portra": [
        "halation_index_ring_minus_background",
        "halation_ring_red_excess_mean",
        "red_highlight_excess_mean",
        "warmth_rgb_mean_r_minus_b",
        "lab_mean_b",
        "saturation_mean",
        "hue_red_orange_highlight_share",
        "local_lab_cov_a_b_abs_mean",
        "split_tone_distance_lab_ab",
        "palette_mean_pairwise_lab_distance",
        "texture_highpass_std",
        "fft_high_freq_power_share",
        "edge_density_canny",
    ],
    "lighting_warm_minus_cool": [
        "warmth_rgb_mean_r_minus_b",
        "lab_mean_b",
        "saturation_mean",
        "hue_warm_share",
        "hue_teal_cyan_share",
        "highlight_minus_shadow_warmth",
        "highlight_minus_shadow_lab_b",
        "split_tone_distance_lab_ab",
        "halation_index_ring_minus_background",
        "halation_ring_red_excess_mean",
        "red_highlight_excess_mean",
        "shadow_fraction",
        "highlight_fraction",
    ],
    "scan_pushed_minus_clean": [
        "texture_highpass_std",
        "texture_highpass_mad",
        "fft_high_freq_power_share",
        "fft_high_mid_power_ratio",
        "fft_radial_loglog_slope",
        "laplacian_variance",
        "rms_contrast",
        "dynamic_range_p95_p05",
        "shadow_compression_ratio",
        "highlight_rolloff_ratio",
        "luma_kurtosis_excess",
        "shadow_fraction",
        "lab_mean_b",
        "edge_density_canny",
    ],
}

SELECTED_CORR_FEATURES = [
    "warmth_rgb_mean_r_minus_b",
    "lab_mean_b",
    "saturation_mean",
    "halation_index_ring_minus_background",
    "red_highlight_excess_mean",
    "rms_contrast",
    "texture_highpass_std",
    "edge_density_canny",
    "color_spatial_var_lab_a",
    "color_spatial_var_lab_b",
    "local_lab_cov_a_b_abs_mean",
    "hue_circular_variance",
    "hue_warm_share",
    "highlight_minus_shadow_warmth",
    "split_tone_distance_lab_ab",
    "fft_high_freq_power_share",
    "palette_entropy_6",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def as_float(row: dict[str, object], key: str) -> float:
    return float(row[key])


def safe_float(value: float) -> float:
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return float("nan")
    return float(value)


def mean_sd(values: Iterable[float]) -> tuple[float, float, int]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), 0
    if arr.size == 1:
        return float(arr.mean()), float("nan"), 1
    return float(arr.mean()), float(arr.std(ddof=1)), int(arr.size)


def mean_ci(values: Iterable[float], alpha: float = 0.05) -> tuple[float, float, float, float, int]:
    mean, sd, n = mean_sd(values)
    if n < 2:
        return mean, float("nan"), float("nan"), float("nan"), n
    se = sd / math.sqrt(n)
    crit = t_dist.ppf(1 - alpha / 2, df=n - 1)
    return mean, se, mean - crit * se, mean + crit * se, n


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = 5000) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        return float(values[0]), float(values[0])
    samples = rng.choice(values, size=(n_boot, values.size), replace=True)
    means = samples.mean(axis=1)
    return safe_float(np.percentile(means, 2.5)), safe_float(np.percentile(means, 97.5))


def holm_adjust(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [1.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p_values[idx]
        running = max(running, val)
        adjusted[idx] = min(1.0, running)
    return adjusted


def bh_adjust(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i], reverse=True)
    adjusted = [1.0] * m
    running = 1.0
    for rank_from_top, idx in enumerate(order):
        rank = m - rank_from_top
        val = p_values[idx] * m / rank
        running = min(running, val)
        adjusted[idx] = min(1.0, running)
    return adjusted


def paired_test(
    gpt_values: list[float],
    xai_values: list[float],
    rng: np.random.Generator,
) -> dict[str, object]:
    gpt = np.asarray(gpt_values, dtype=np.float64)
    xai = np.asarray(xai_values, dtype=np.float64)
    diff = xai - gpt
    mean_diff, se, ci_low, ci_high, n = mean_ci(diff)
    boot_low, boot_high = bootstrap_ci(diff, rng)
    t_res = ttest_1samp(diff, popmean=0.0) if n > 1 else None
    try:
        w_res = wilcoxon(diff) if n > 1 else None
    except ValueError:
        w_res = None
    sd = float(diff.std(ddof=1)) if n > 1 else float("nan")
    return {
        "n_pairs": n,
        "mean_chatgpt": safe_float(gpt.mean()) if gpt.size else float("nan"),
        "mean_xai": safe_float(xai.mean()) if xai.size else float("nan"),
        "mean_difference_xai_minus_chatgpt": safe_float(mean_diff),
        "median_difference_xai_minus_chatgpt": safe_float(np.median(diff)) if diff.size else float("nan"),
        "sd_difference": safe_float(sd),
        "se_difference": safe_float(se),
        "ci95_low": safe_float(ci_low),
        "ci95_high": safe_float(ci_high),
        "bootstrap_ci95_low": boot_low,
        "bootstrap_ci95_high": boot_high,
        "paired_t_statistic": safe_float(t_res.statistic) if t_res is not None else float("nan"),
        "paired_t_p_value": safe_float(t_res.pvalue) if t_res is not None else float("nan"),
        "cohens_dz": safe_float(mean_diff / sd) if sd and np.isfinite(sd) and sd > 0 else float("nan"),
        "wilcoxon_statistic": safe_float(w_res.statistic) if w_res is not None else float("nan"),
        "wilcoxon_p_value": safe_float(w_res.pvalue) if w_res is not None else float("nan"),
    }


def global_feature_sds(image_rows: list[dict[str, str]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for feature in set().union(*TARGETED_TESTS.values(), SELECTED_CORR_FEATURES):
        vals = np.asarray([float(row[feature]) for row in image_rows if feature in row], dtype=np.float64)
        if vals.size > 1:
            sd = vals.std(ddof=1)
            out[feature] = float(sd) if sd > 0 else 1.0
    return out


def matched_key_for_effect(row: dict[str, str]) -> tuple[str, ...]:
    effect = row["effect_type"]
    if effect == "filmstock_cinestill_minus_portra":
        return (row["scene_key"], row["light_key"], row["scan_key"], row["replicate"])
    if effect == "lighting_warm_minus_cool":
        return (row["scene_key"], row["film_key"], row["scan_key"], row["replicate"])
    if effect == "scan_pushed_minus_clean":
        return (row["scene_key"], row["film_key"], row["light_key"], row["replicate"])
    return (row.get("scene_key", ""), row.get("replicate", ""))


def targeted_feature_tests(
    effect_rows: list[dict[str, str]],
    image_rows: list[dict[str, str]],
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    feature_sds = global_feature_sds(image_rows)
    indexed: dict[tuple[str, str, str, tuple[str, ...]], float] = {}
    main_rows = [row for row in effect_rows if row.get("effect_family") == "main_effect"]
    for row in main_rows:
        key = (row["effect_type"], row["feature"], row["model_key"], matched_key_for_effect(row))
        indexed[key] = float(row["effect_value"])

    out: list[dict[str, object]] = []
    for effect_type, features in TARGETED_TESTS.items():
        for feature in features:
            matched = sorted({matched_key_for_effect(row) for row in main_rows if row["effect_type"] == effect_type and row["feature"] == feature})
            gpt_vals: list[float] = []
            xai_vals: list[float] = []
            for key in matched:
                gpt = indexed.get((effect_type, feature, "chatgpt_image_2", key))
                xai = indexed.get((effect_type, feature, "xai_grok_imagine", key))
                if gpt is not None and xai is not None:
                    gpt_vals.append(gpt)
                    xai_vals.append(xai)
            if not gpt_vals:
                continue
            test = paired_test(gpt_vals, xai_vals, rng)
            sd = feature_sds.get(feature, 1.0)
            row = {
                "effect_type": effect_type,
                "effect_label": EFFECT_LABELS.get(effect_type, effect_type),
                "feature": feature,
                "feature_label": FEATURE_LABELS.get(feature, feature),
                "mean_chatgpt_raw": safe_float(np.mean(gpt_vals)),
                "mean_xai_raw": safe_float(np.mean(xai_vals)),
                "mean_chatgpt_standardized": safe_float(np.mean(np.asarray(gpt_vals) / sd)),
                "mean_xai_standardized": safe_float(np.mean(np.asarray(xai_vals) / sd)),
                **test,
            }
            out.append(row)

    pvals = [float(row["paired_t_p_value"]) for row in out]
    holm = holm_adjust(pvals)
    bh = bh_adjust(pvals)
    for row, holm_p, bh_p in zip(out, holm, bh):
        row["holm_adjusted_p"] = safe_float(holm_p)
        row["bh_fdr_p"] = safe_float(bh_p)
    return out


@dataclass
class RegressionResult:
    method: str
    metric: str
    terms: list[str]
    beta: np.ndarray
    se: np.ndarray
    p: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    residuals: np.ndarray
    weights: np.ndarray | None = None


def design_distance_model(rows: list[dict[str, str]], metric: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    y = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
    terms = [
        "Intercept",
        "xAI model",
        "Lighting effect",
        "Scan effect",
        "xAI by lighting effect",
        "xAI by scan effect",
        "Backstage scene",
        "Corner store scene",
        "Replicate 2",
    ]
    x = []
    for row in rows:
        model = 1.0 if row["model_key"] == "xai_grok_imagine" else 0.0
        lighting = 1.0 if row["effect_type"] == "lighting_warm_minus_cool" else 0.0
        scan = 1.0 if row["effect_type"] == "scan_pushed_minus_clean" else 0.0
        backstage = 1.0 if row["scene_key"] == "backstage" else 0.0
        corner = 1.0 if row["scene_key"] == "corner_store" else 0.0
        rep2 = 1.0 if str(row["replicate"]) == "2" else 0.0
        x.append([1.0, model, lighting, scan, model * lighting, model * scan, backstage, corner, rep2])
    return y, np.asarray(x, dtype=np.float64), terms


def fit_linear_model(
    y: np.ndarray,
    x: np.ndarray,
    method: str,
    weights: np.ndarray | None = None,
) -> RegressionResult:
    n, p = x.shape
    if weights is None:
        weights = np.ones(n)
    sw = np.sqrt(weights).reshape(-1, 1)
    xw = x * sw
    yw = y * sw.reshape(-1)
    xtx_inv = pinv(xw.T @ xw)
    beta = xtx_inv @ xw.T @ yw
    fitted = x @ beta
    residuals = y - fitted
    df = max(n - np.linalg.matrix_rank(x), 1)

    if method == "OLS HC3":
        hat = np.sum((x @ pinv(x.T @ x)) * x, axis=1)
        denom = np.maximum(1.0 - hat, 1e-8)
        meat = x.T @ (x * ((residuals / denom) ** 2).reshape(-1, 1))
        cov = pinv(x.T @ x) @ meat @ pinv(x.T @ x)
    else:
        sigma2 = float(np.sum(weights * residuals * residuals) / df)
        cov = xtx_inv * sigma2

    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    tvals = beta / np.maximum(se, 1e-12)
    pvals = 2 * (1 - t_dist.cdf(np.abs(tvals), df=df))
    crit = t_dist.ppf(0.975, df=df)
    return RegressionResult(
        method=method,
        metric="",
        terms=[],
        beta=beta,
        se=se,
        p=pvals,
        ci_low=beta - crit * se,
        ci_high=beta + crit * se,
        residuals=residuals,
        weights=weights,
    )


def fit_huber_irls(y: np.ndarray, x: np.ndarray, c: float = 1.345, max_iter: int = 100) -> RegressionResult:
    weights = np.ones_like(y)
    beta_prev = np.zeros(x.shape[1])
    for _ in range(max_iter):
        fit = fit_linear_model(y, x, "Huber IRLS", weights)
        resid = fit.residuals
        scale = np.median(np.abs(resid - np.median(resid))) / 0.6745
        if not np.isfinite(scale) or scale <= 1e-10:
            scale = np.std(resid, ddof=1) or 1.0
        u = resid / (scale * c)
        weights = np.where(np.abs(u) <= 1.0, 1.0, 1.0 / np.maximum(np.abs(u), 1e-12))
        if np.linalg.norm(fit.beta - beta_prev) < 1e-8:
            break
        beta_prev = fit.beta.copy()
    return fit_linear_model(y, x, "Huber IRLS", weights)


def regression_tables(distance_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    metrics = [
        "distance_all_selected_features",
        "distance_color_features",
        "distance_contrast_texture_features",
        "distance_structure_features",
    ]
    out: list[dict[str, object]] = []
    for metric in metrics:
        y, x, terms = design_distance_model(distance_rows, metric)
        ols = fit_linear_model(y, x, "OLS HC3")

        group_vars: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row, resid in zip(distance_rows, ols.residuals):
            group_vars[(row["effect_type"], row["model_key"])].append(float(resid))
        variances = {}
        for key, vals in group_vars.items():
            arr = np.asarray(vals, dtype=np.float64)
            variances[key] = float(arr.var(ddof=1)) if arr.size > 1 else 1.0
        weights = np.asarray([
            1.0 / max(variances[(row["effect_type"], row["model_key"])], 1e-8)
            for row in distance_rows
        ])
        wls = fit_linear_model(y, x, "Two-stage WLS", weights)
        huber = fit_huber_irls(y, x)
        for result in [ols, wls, huber]:
            for term, beta, se, pval, lo, hi in zip(terms, result.beta, result.se, result.p, result.ci_low, result.ci_high):
                out.append(
                    {
                        "metric": metric,
                        "metric_label": METRIC_LABELS.get(metric, metric),
                        "method": result.method,
                        "term": term,
                        "estimate": safe_float(beta),
                        "standard_error": safe_float(se),
                        "ci95_low": safe_float(lo),
                        "ci95_high": safe_float(hi),
                        "p_value": safe_float(pval),
                    }
                )
    return out


def design_matrix_for_manova(rows: list[dict[str, str]], terms: list[str]) -> tuple[np.ndarray, list[str]]:
    columns = [np.ones(len(rows))]
    names = ["Intercept"]
    binaries = {
        "model": np.asarray([1.0 if row["model_key"] == "xai_grok_imagine" else 0.0 for row in rows]),
        "film": np.asarray([1.0 if row["film_key"] == "cinestill800t" else 0.0 for row in rows]),
        "light": np.asarray([1.0 if row["light_key"] == "warm_practical" else 0.0 for row in rows]),
        "scan": np.asarray([1.0 if row["scan_key"] == "pushed_scan" else 0.0 for row in rows]),
    }

    for term in terms:
        if term == "scene":
            columns.append(np.asarray([1.0 if row["scene_key"] == "backstage" else 0.0 for row in rows]))
            names.append("scene_backstage")
            columns.append(np.asarray([1.0 if row["scene_key"] == "corner_store" else 0.0 for row in rows]))
            names.append("scene_corner_store")
        elif ":" in term:
            parts = term.split(":")
            col = np.ones(len(rows))
            for part in parts:
                col = col * binaries[part]
            columns.append(col)
            names.append(term)
        else:
            columns.append(binaries[term])
            names.append(term)
    return np.column_stack(columns), names


def residual_sscp(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    beta = pinv(x) @ y
    resid = y - x @ beta
    return resid.T @ resid


def pillai_trace(y: np.ndarray, x_full: np.ndarray, x_reduced: np.ndarray) -> float:
    e_full = residual_sscp(y, x_full)
    e_reduced = residual_sscp(y, x_reduced)
    h = e_reduced - e_full
    return safe_float(np.trace(h @ pinv(h + e_full)))


def manova_permutation_tests(
    pca_rows: list[dict[str, str]],
    n_perm: int = 999,
    seed: int = 216,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    y = np.asarray([[float(row[f"PC{i}"]) for i in range(1, 5)] for row in pca_rows], dtype=np.float64)
    tested_terms = [
        "model",
        "film",
        "light",
        "scan",
        "model:film",
        "model:light",
        "model:scan",
        "film:light",
        "film:scan",
        "light:scan",
    ]
    full_terms = ["scene", *tested_terms]
    x_full, _ = design_matrix_for_manova(pca_rows, full_terms)
    results: list[dict[str, object]] = []

    for term in tested_terms:
        reduced_terms = [t for t in full_terms if t != term]
        x_reduced, _ = design_matrix_for_manova(pca_rows, reduced_terms)
        observed = pillai_trace(y, x_full, x_reduced)

        beta_red = pinv(x_reduced) @ y
        fitted_red = x_reduced @ beta_red
        resid_red = y - fitted_red
        perm_stats = []
        for _ in range(n_perm):
            perm = rng.permutation(y.shape[0])
            y_perm = fitted_red + resid_red[perm, :]
            perm_stats.append(pillai_trace(y_perm, x_full, x_reduced))
        perm_stats_arr = np.asarray(perm_stats, dtype=np.float64)
        p_perm = (1 + np.sum(perm_stats_arr >= observed)) / (n_perm + 1)
        results.append(
            {
                "term": term,
                "term_label": term.replace(":", " by "),
                "pillai_trace": safe_float(observed),
                "permutation_p_value": safe_float(p_perm),
                "n_permutations": n_perm,
                "response_components": "PC1, PC2, PC3, PC4",
                "covariates": "scene fixed effects",
            }
        )
    pvals = [float(row["permutation_p_value"]) for row in results]
    for row, holm, bh in zip(results, holm_adjust(pvals), bh_adjust(pvals)):
        row["holm_adjusted_p"] = safe_float(holm)
        row["bh_fdr_p"] = safe_float(bh)
    return results


def paired_distance_tests(
    distance_rows: list[dict[str, str]],
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    metrics = list(METRIC_LABELS.keys())
    out: list[dict[str, object]] = []
    for effect_type in sorted({row["effect_type"] for row in distance_rows}):
        rows = [row for row in distance_rows if row["effect_type"] == effect_type]
        match_keys = sorted({
            (row["scene_key"], row.get("film_key", ""), row.get("light_key", ""), row.get("scan_key", ""), row["replicate"])
            for row in rows
        })
        for metric in metrics:
            gpt_vals: list[float] = []
            xai_vals: list[float] = []
            index = {
                (row["model_key"], row["scene_key"], row.get("film_key", ""), row.get("light_key", ""), row.get("scan_key", ""), row["replicate"]): float(row[metric])
                for row in rows
            }
            for key in match_keys:
                gpt = index.get(("chatgpt_image_2", *key))
                xai = index.get(("xai_grok_imagine", *key))
                if gpt is not None and xai is not None:
                    gpt_vals.append(gpt)
                    xai_vals.append(xai)
            test = paired_test(gpt_vals, xai_vals, rng)
            out.append(
                {
                    "effect_type": effect_type,
                    "effect_label": EFFECT_LABELS.get(effect_type, effect_type),
                    "distance_metric": metric,
                    "metric_label": METRIC_LABELS.get(metric, metric),
                    **test,
                }
            )
    pvals = [float(row["paired_t_p_value"]) for row in out]
    for row, holm, bh in zip(out, holm_adjust(pvals), bh_adjust(pvals)):
        row["holm_adjusted_p"] = safe_float(holm)
        row["bh_fdr_p"] = safe_float(bh)
    return out


def nonparametric_tests(distance_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    metrics = list(METRIC_LABELS.keys())
    out: list[dict[str, object]] = []
    for model in sorted({row["model_key"] for row in distance_rows}):
        model_rows = [row for row in distance_rows if row["model_key"] == model]
        for metric in metrics:
            grouped = defaultdict(list)
            for row in model_rows:
                grouped[row["effect_type"]].append(float(row[metric]))
            effects = sorted(grouped)
            if len(effects) >= 2:
                k = kruskal(*[grouped[effect] for effect in effects])
                out.append(
                    {
                        "test": "Kruskal-Wallis",
                        "model_key": model,
                        "distance_metric": metric,
                        "metric_label": METRIC_LABELS.get(metric, metric),
                        "grouping": "effect_type",
                        "statistic": safe_float(k.statistic),
                        "p_value": safe_float(k.pvalue),
                        "n_groups": len(effects),
                    }
                )

            # Friedman test on scene by replicate blocks after averaging over
            # other design variables within each effect type.
            blocks = sorted({(row["scene_key"], row["replicate"]) for row in model_rows})
            effect_values: dict[str, list[float]] = {effect: [] for effect in effects}
            for block in blocks:
                for effect in effects:
                    vals = [
                        float(row[metric])
                        for row in model_rows
                        if row["effect_type"] == effect and (row["scene_key"], row["replicate"]) == block
                    ]
                    if vals:
                        effect_values[effect].append(float(np.mean(vals)))
            if effects and all(len(effect_values[effect]) == len(blocks) for effect in effects):
                fr = friedmanchisquare(*[effect_values[effect] for effect in effects])
                out.append(
                    {
                        "test": "Friedman repeated-measures ANOVA",
                        "model_key": model,
                        "distance_metric": metric,
                        "metric_label": METRIC_LABELS.get(metric, metric),
                        "grouping": "effect_type within scene by replicate blocks",
                        "statistic": safe_float(fr.statistic),
                        "p_value": safe_float(fr.pvalue),
                        "n_groups": len(effects),
                        "n_blocks": len(blocks),
                    }
                )
    pvals = [float(row["p_value"]) for row in out]
    for row, holm, bh in zip(out, holm_adjust(pvals), bh_adjust(pvals)):
        row["holm_adjusted_p"] = safe_float(holm)
        row["bh_fdr_p"] = safe_float(bh)
    return out


def interaction_summary(effect_rows: list[dict[str, str]], image_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    feature_sds = global_feature_sds(image_rows)
    selected_features = sorted(feature_sds)
    rows = [
        row
        for row in effect_rows
        if row.get("effect_family") in {"two_factor_interaction", "three_factor_interaction"}
        and row["feature"] in selected_features
    ]
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        sd = feature_sds.get(row["feature"], 1.0)
        groups[(row["effect_type"], row["model_key"])].append(abs(float(row["effect_value"]) / sd))
    out = []
    for (effect_type, model), vals in sorted(groups.items()):
        mean, se, lo, hi, n = mean_ci(vals)
        out.append(
            {
                "interaction_type": effect_type,
                "interaction_label": EFFECT_LABELS.get(effect_type, effect_type),
                "model_key": model,
                "n_standardized_feature_contrasts": n,
                "mean_abs_standardized_interaction": safe_float(mean),
                "se": safe_float(se),
                "ci95_low": safe_float(lo),
                "ci95_high": safe_float(hi),
            }
        )
    return out


def plot_pca_scores() -> None:
    scores = read_csv(ANALYSIS_DIR / "pca_image_scores.csv")
    ellipses = read_csv(ANALYSIS_DIR / "pca_confidence_ellipses.csv")
    explained = read_csv(ANALYSIS_DIR / "pca_explained_variance.csv")
    pc1 = float(explained[0]["explained_variance_ratio"]) * 100
    pc2 = float(explained[1]["explained_variance_ratio"]) * 100

    colors = {"chatgpt_image_2": "#2A6FBB", "xai_grok_imagine": "#C43C39"}
    markers = {"portra400": "o", "cinestill800t": "s"}
    fig, ax = plt.subplots(figsize=(8.5, 6.2), dpi=180)
    for model in ["chatgpt_image_2", "xai_grok_imagine"]:
        for film in ["portra400", "cinestill800t"]:
            pts = [row for row in scores if row["model_key"] == model and row["film_key"] == film]
            ax.scatter(
                [float(row["PC1"]) for row in pts],
                [float(row["PC2"]) for row in pts],
                s=42,
                marker=markers[film],
                color=colors[model],
                edgecolor="white",
                linewidth=0.5,
                alpha=0.78,
                label=f"{MODEL_LABELS[model]}, {film}",
            )
    for model in ["chatgpt_image_2", "xai_grok_imagine"]:
        pts = [row for row in ellipses if row["group_type"] == "model_key" and row["group"] == model]
        pts.sort(key=lambda row: int(row["point_index"]))
        ax.plot(
            [float(row["PC1"]) for row in pts],
            [float(row["PC2"]) for row in pts],
            color=colors[model],
            linewidth=2.2,
            label=f"{MODEL_LABELS[model]} 95% mean ellipse",
        )
    ax.axhline(0, color="#D8D8D8", linewidth=0.9)
    ax.axvline(0, color="#D8D8D8", linewidth=0.9)
    ax.set_xlabel(f"PC1 ({pc1:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pc2:.1f}% variance)")
    ax.set_title("Image Feature Space: PCA Scores and 95% Mean Ellipses")
    ax.legend(fontsize=7.5, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig01_pca_scores_ellipses.png")
    plt.close(fig)


def plot_pca_loadings() -> None:
    rows = read_csv(ANALYSIS_DIR / "pca_loadings.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=180)
    for ax, pc in zip(axes, ["PC1", "PC2"]):
        pc_rows = [row for row in rows if row["component"] == pc]
        pc_rows.sort(key=lambda row: abs(float(row["loading"])), reverse=True)
        top = pc_rows[:12][::-1]
        labels = [FEATURE_LABELS.get(row["feature"], row["feature"]) for row in top]
        vals = [float(row["loading"]) for row in top]
        colors = ["#C43C39" if v > 0 else "#2A6FBB" for v in vals]
        ax.barh(labels, vals, color=colors, alpha=0.85)
        ax.axvline(0, color="#333333", linewidth=0.8)
        ax.set_title(f"Top Loadings for {pc}")
        ax.tick_params(axis="y", labelsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig02_pca_loadings.png")
    plt.close(fig)


def plot_distance_bars(summary_rows: list[dict[str, str]]) -> None:
    metrics = list(METRIC_LABELS.keys())
    effects = [
        "filmstock_cinestill_minus_portra",
        "lighting_warm_minus_cool",
        "scan_pushed_minus_clean",
    ]
    models = ["chatgpt_image_2", "xai_grok_imagine"]
    colors = {"chatgpt_image_2": "#2A6FBB", "xai_grok_imagine": "#C43C39"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=180, sharex=True)
    axes = axes.reshape(-1)
    for ax, metric in zip(axes, metrics):
        subset = [row for row in summary_rows if row["distance_metric"] == metric]
        x = np.arange(len(effects))
        width = 0.36
        for j, model in enumerate(models):
            means = []
            errors = []
            for effect in effects:
                row = next(r for r in subset if r["effect_type"] == effect and r["model_key"] == model)
                means.append(float(row["mean_distance"]))
                se = float(row["se_distance"])
                n = int(row["n_pairs"])
                errors.append(t_dist.ppf(0.975, n - 1) * se)
            offset = (j - 0.5) * width
            ax.bar(x + offset, means, width=width, color=colors[model], alpha=0.85, label=MODEL_LABELS[model])
            ax.errorbar(x + offset, means, yerr=errors, fmt="none", ecolor="#222222", elinewidth=0.9, capsize=3)
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xticks(x)
        ax.set_xticklabels(["Filmstock", "Lighting", "Scan"], rotation=0)
        ax.set_ylabel("Mean standardized distance")
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.8)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Prompt Responsiveness Distances by Model and Prompt Factor", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig03_responsiveness_distance_bars.png", bbox_inches="tight")
    plt.close(fig)


def plot_model_difference_forest(test_rows: list[dict[str, object]]) -> None:
    rows = sorted(
        test_rows,
        key=lambda row: (
            list(EFFECT_LABELS).index(row["effect_type"]) if row["effect_type"] in EFFECT_LABELS else 99,
            list(METRIC_LABELS).index(row["distance_metric"]),
        ),
    )
    fig, ax = plt.subplots(figsize=(9, 7), dpi=180)
    y = np.arange(len(rows))
    colors = {
        "distance_all_selected_features": "#222222",
        "distance_color_features": "#BD5A2A",
        "distance_contrast_texture_features": "#4A7D45",
        "distance_structure_features": "#756BB1",
    }
    for i, row in enumerate(rows):
        mean = float(row["mean_difference_xai_minus_chatgpt"])
        lo = float(row["ci95_low"])
        hi = float(row["ci95_high"])
        metric = row["distance_metric"]
        ax.plot([lo, hi], [i, i], color=colors[metric], linewidth=1.8)
        ax.scatter([mean], [i], color=colors[metric], s=34, zorder=3)
    labels = [
        f"{EFFECT_LABELS[row['effect_type']].split(' minus ')[0]} | {METRIC_LABELS[row['distance_metric']]}"
        for row in rows
    ]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_xlabel("Mean paired difference: Grok Imagine minus ChatGPT Image 2")
    ax.set_title("Paired Model Differences in Prompt Responsiveness")
    ax.grid(axis="x", color="#E6E6E6")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig04_model_difference_forest.png")
    plt.close(fig)


def plot_targeted_heatmap(target_rows: list[dict[str, object]]) -> None:
    rows = []
    values = []
    for effect_type, features in TARGETED_TESTS.items():
        for feature in features:
            found = next(row for row in target_rows if row["effect_type"] == effect_type and row["feature"] == feature)
            rows.append(f"{EFFECT_LABELS[effect_type].split(' minus ')[0]}: {FEATURE_LABELS.get(feature, feature)}")
            values.append([float(found["mean_chatgpt_standardized"]), float(found["mean_xai_standardized"])])
    mat = np.asarray(values, dtype=np.float64)
    vmax = max(0.2, float(np.nanmax(np.abs(mat))))
    fig, ax = plt.subplots(figsize=(7.2, 10.5), dpi=180)
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["ChatGPT Image 2", "Grok Imagine"])
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows, fontsize=7.2)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("Targeted Prompt Effects, Standardized by Feature SD")
    fig.colorbar(im, ax=ax, shrink=0.75, label="Mean high-minus-low effect in SD units")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig05_targeted_feature_effect_heatmap.png")
    plt.close(fig)


def plot_interaction_heatmap(interaction_rows: list[dict[str, object]]) -> None:
    interactions = [
        "filmstock_by_lighting",
        "filmstock_by_scan",
        "lighting_by_scan",
        "filmstock_by_lighting_by_scan",
    ]
    models = ["chatgpt_image_2", "xai_grok_imagine"]
    mat = np.zeros((len(interactions), len(models)))
    for i, interaction in enumerate(interactions):
        for j, model in enumerate(models):
            row = next(r for r in interaction_rows if r["interaction_type"] == interaction and r["model_key"] == model)
            mat[i, j] = float(row["mean_abs_standardized_interaction"])
    fig, ax = plt.subplots(figsize=(7.8, 4.7), dpi=180)
    im = ax.imshow(mat, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels([MODEL_LABELS[m] for m in models])
    ax.set_yticks(np.arange(len(interactions)))
    ax.set_yticklabels([EFFECT_LABELS[i] for i in interactions])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Average Absolute Standardized Interaction Contrasts")
    fig.colorbar(im, ax=ax, label="Mean absolute interaction effect, SD units")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig06_interaction_effect_heatmap.png")
    plt.close(fig)


def plot_manova(manova_rows: list[dict[str, object]]) -> None:
    rows = sorted(manova_rows, key=lambda row: float(row["pillai_trace"]), reverse=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=180)
    y = np.arange(len(rows))
    vals = [float(row["pillai_trace"]) for row in rows]
    ax.barh(y, vals, color="#4E79A7", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels([row["term_label"] for row in rows])
    ax.invert_yaxis()
    for i, row in enumerate(rows):
        p = float(row["permutation_p_value"])
        ax.text(vals[i] + max(vals) * 0.015, i, f"p={p:.3f}", va="center", fontsize=8)
    ax.set_xlabel("Pillai trace on PC1-PC4")
    ax.set_title("Permutation MANOVA: Multivariate Effects in PCA Space")
    ax.grid(axis="x", color="#E5E5E5")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig07_manova_pillai.png")
    plt.close(fig)


def plot_regression_coefficients(rows: list[dict[str, object]]) -> None:
    keep_terms = ["xAI model", "Lighting effect", "Scan effect", "xAI by lighting effect", "xAI by scan effect"]
    subset = [
        row
        for row in rows
        if row["metric"] == "distance_all_selected_features"
        and row["method"] in {"OLS HC3", "Two-stage WLS", "Huber IRLS"}
        and row["term"] in keep_terms
    ]
    methods = ["OLS HC3", "Two-stage WLS", "Huber IRLS"]
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=180)
    base_y = np.arange(len(keep_terms))
    offsets = {"OLS HC3": -0.22, "Two-stage WLS": 0.0, "Huber IRLS": 0.22}
    colors = {"OLS HC3": "#2A6FBB", "Two-stage WLS": "#6C8E3F", "Huber IRLS": "#C43C39"}
    for method in methods:
        vals = []
        lo = []
        hi = []
        for term in keep_terms:
            row = next(r for r in subset if r["method"] == method and r["term"] == term)
            vals.append(float(row["estimate"]))
            lo.append(float(row["ci95_low"]))
            hi.append(float(row["ci95_high"]))
        y = base_y + offsets[method]
        ax.errorbar(
            vals,
            y,
            xerr=[np.asarray(vals) - np.asarray(lo), np.asarray(hi) - np.asarray(vals)],
            fmt="o",
            color=colors[method],
            label=method,
            capsize=3,
        )
    ax.set_yticks(base_y)
    ax.set_yticklabels(keep_terms)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_xlabel("Coefficient on standardized all-feature responsiveness distance")
    ax.set_title("OLS, WLS, and Robust Regression Coefficients")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="x", color="#E6E6E6")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig08_regression_coefficients.png")
    plt.close(fig)


def plot_feature_correlation(image_rows: list[dict[str, str]]) -> None:
    features = [f for f in SELECTED_CORR_FEATURES if f in image_rows[0]]
    mat = np.asarray([[float(row[f]) for f in features] for row in image_rows], dtype=np.float64)
    corr = np.corrcoef(mat, rowvar=False)
    fig, ax = plt.subplots(figsize=(8, 7), dpi=180)
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    labels = [FEATURE_LABELS.get(f, f) for f in features]
    ax.set_xticks(np.arange(len(features)))
    ax.set_yticks(np.arange(len(features)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=6)
    ax.set_title("Correlation Matrix of Selected Image Features")
    fig.colorbar(im, ax=ax, shrink=0.75, label="Pearson r")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig09_feature_correlation_heatmap.png")
    plt.close(fig)


def plot_nonparametric(nonparam_rows: list[dict[str, object]], distance_rows: list[dict[str, str]]) -> None:
    metric = "distance_all_selected_features"
    effects = [
        "filmstock_cinestill_minus_portra",
        "lighting_warm_minus_cool",
        "scan_pushed_minus_clean",
    ]
    models = ["chatgpt_image_2", "xai_grok_imagine"]
    fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=180)
    x = np.arange(len(effects))
    width = 0.36
    colors = {"chatgpt_image_2": "#2A6FBB", "xai_grok_imagine": "#C43C39"}
    for j, model in enumerate(models):
        means = []
        errors = []
        for effect in effects:
            vals = [float(row[metric]) for row in distance_rows if row["model_key"] == model and row["effect_type"] == effect]
            mean, se, lo, hi, n = mean_ci(vals)
            means.append(mean)
            errors.append(mean - lo)
        offset = (j - 0.5) * width
        ax.bar(x + offset, means, width=width, color=colors[model], alpha=0.85, label=MODEL_LABELS[model])
        ax.errorbar(x + offset, means, yerr=errors, fmt="none", ecolor="#222222", elinewidth=0.8, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(["Filmstock", "Lighting", "Scan"])
    ax.set_ylabel("Mean standardized distance")
    ax.set_title("Distance Ordering Used in Kruskal-Wallis and Friedman Tests")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#E5E5E5")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig10_nonparametric_effect_ordering.png")
    plt.close(fig)


def contact_sheet() -> None:
    manifest = read_csv(RUN_DIR / "manifest.csv")
    rows = [
        row
        for row in manifest
        if row["scene_key"] == "corner_store"
        and row["scan_key"] == "clean_scan"
        and row["replicate"] == "1"
    ]
    rows.sort(key=lambda row: (row["model_key"], row["film_key"], row["light_key"]))
    by_key = {(row["model_key"], row["film_key"], row["light_key"]): row for row in rows}
    models = ["chatgpt_image_2", "xai_grok_imagine"]
    cols = [
        ("portra400", "cool_ambient"),
        ("portra400", "warm_practical"),
        ("cinestill800t", "cool_ambient"),
        ("cinestill800t", "warm_practical"),
    ]
    thumb = 220
    label_h = 44
    left_w = 150
    canvas = Image.new("RGB", (left_w + len(cols) * thumb, label_h + len(models) * (thumb + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for j, (film, light) in enumerate(cols):
        label = f"{film}\n{light}"
        draw.multiline_text((left_w + j * thumb + 8, 8), label, fill=(20, 20, 20), font=font, spacing=3)
    for i, model in enumerate(models):
        y0 = label_h + i * (thumb + label_h)
        draw.text((8, y0 + thumb // 2 - 8), MODEL_LABELS[model], fill=(20, 20, 20), font=font)
        for j, (film, light) in enumerate(cols):
            row = by_key[(model, film, light)]
            im = Image.open(row["image_path"]).convert("RGB")
            im.thumbnail((thumb, thumb))
            x = left_w + j * thumb
            canvas.paste(im, (x, y0))
    canvas.save(FIG_DIR / "fig11_corner_store_contact_sheet.png")


def write_report(
    paired_rows: list[dict[str, object]],
    targeted_rows: list[dict[str, object]],
    manova_rows: list[dict[str, object]],
    regression_rows: list[dict[str, object]],
    nonparam_rows: list[dict[str, object]],
) -> None:
    all_metric = [row for row in paired_rows if row["distance_metric"] == "distance_all_selected_features"]
    all_metric_lines = []
    for row in all_metric:
        all_metric_lines.append(
            f"- {row['effect_label']}: ChatGPT mean {float(row['mean_chatgpt']):.3f}, "
            f"Grok mean {float(row['mean_xai']):.3f}, paired difference "
            f"{float(row['mean_difference_xai_minus_chatgpt']):.3f}, "
            f"95% CI [{float(row['ci95_low']):.3f}, {float(row['ci95_high']):.3f}], "
            f"p = {float(row['paired_t_p_value']):.3f}."
        )

    strongest_target = sorted(targeted_rows, key=lambda row: abs(float(row["mean_difference_xai_minus_chatgpt"])), reverse=True)[:6]
    target_lines = []
    for row in strongest_target:
        target_lines.append(
            f"- {row['effect_label']} on {row['feature_label']}: model difference "
            f"{float(row['mean_difference_xai_minus_chatgpt']):.4f}, "
            f"p = {float(row['paired_t_p_value']):.3f}, Holm p = {float(row['holm_adjusted_p']):.3f}."
        )

    manova_lines = []
    for row in sorted(manova_rows, key=lambda r: float(r["pillai_trace"]), reverse=True)[:6]:
        manova_lines.append(
            f"- {row['term_label']}: Pillai trace {float(row['pillai_trace']):.3f}, "
            f"permutation p = {float(row['permutation_p_value']):.3f}."
        )

    coef_rows = [
        row for row in regression_rows
        if row["metric"] == "distance_all_selected_features"
        and row["method"] == "OLS HC3"
        and row["term"] in {"xAI model", "xAI by lighting effect", "xAI by scan effect"}
    ]
    coef_lines = []
    for row in coef_rows:
        coef_lines.append(
            f"- {row['term']}: beta {float(row['estimate']):.3f}, "
            f"95% CI [{float(row['ci95_low']):.3f}, {float(row['ci95_high']):.3f}], "
            f"p = {float(row['p_value']):.3f}."
        )

    text = f"""# Statistical Analysis Report

This folder contains the inferential analysis for the filmstock responsiveness benchmark. The analysis uses the 96-image feature table produced from the generated images and emphasizes methods from the advanced statistics syllabus: factorial regression, robust estimation, weighted least squares, PCA, MANOVA-style multivariate testing, repeated-measures paired tests, and nonparametric checks.

## Primary Response Variable

The main response variable is standardized prompt-response distance: the Euclidean distance between paired images when one prompt factor changes and all other design variables are held constant. This converts the image outputs into a clean repeated-measures response suitable for paired tests and regression.

## Model Comparison on Total Responsiveness

{chr(10).join(all_metric_lines)}

Interpretation: in this proof-of-concept sample, total feature-space movement is not significantly different between the two models. The more interesting result is cue-specific and feature-family-specific behavior rather than a single overall winner.

## Targeted Filmstock and Lighting Features

{chr(10).join(target_lines)}

Interpretation: the strongest raw effects occur in the color and lighting variables. Warm practical lighting produces especially large shifts in warmth, Lab yellow-blue, saturation, and red-highlight measures. Filmstock and pushed-scan effects are measurable, but more subtle and more dependent on the feature family.

## PCA and MANOVA-Style Multivariate Testing

{chr(10).join(manova_lines)}

The PCA plots should be used as the visual multivariate summary. The permutation MANOVA table tests whether design factors move the first four PCA dimensions after accounting for scene fixed effects.

## Regression, WLS, and Robust Estimation

{chr(10).join(coef_lines)}

The regression table reports three parallel fits: OLS with HC3 robust standard errors, two-stage WLS using residual variance weights, and Huber IRLS robust regression. Agreement across these fits is a sensitivity check against heteroskedasticity and influential generated-image outliers.

## Nonparametric Checks

`nonparametric_tests.csv` includes Kruskal-Wallis tests across prompt effect types and Friedman repeated-measures tests after aggregating within scene by replicate blocks. These are included as assumption-light checks because generated images can violate normal-error assumptions.

## Recommended Writeup Claim

The defensible claim is not simply that one model is universally more responsive. The stronger statistical claim is that filmstock-style responsiveness is multidimensional: lighting cues dominate total movement, color features show the clearest model-specific behavior, and filmstock/scan cues appear in targeted halation, color, texture, and structure features. This supports the project’s argument that prompt responsiveness should be evaluated as a multivariate artistic-control axis rather than as ordinary prompt adherence.

## Figure List

- `fig01_pca_scores_ellipses.png`: PCA feature space and 95 percent mean ellipses.
- `fig02_pca_loadings.png`: top feature loadings for PC1 and PC2.
- `fig03_responsiveness_distance_bars.png`: mean prompt-response distances with confidence intervals.
- `fig04_model_difference_forest.png`: paired Grok minus ChatGPT model differences.
- `fig05_targeted_feature_effect_heatmap.png`: targeted filmstock, lighting, and scan effects in SD units.
- `fig06_interaction_effect_heatmap.png`: average absolute standardized interaction contrasts.
- `fig07_manova_pillai.png`: permutation MANOVA Pillai trace results.
- `fig08_regression_coefficients.png`: OLS, WLS, and robust regression coefficients.
- `fig09_feature_correlation_heatmap.png`: selected-feature correlation matrix.
- `fig10_nonparametric_effect_ordering.png`: effect ordering used in nonparametric checks.
- `fig11_corner_store_contact_sheet.png`: visual examples for one scene and selected conditions.
"""
    (STATS_DIR / "STATISTICAL_ANALYSIS_REPORT.md").write_text(text)


def main() -> None:
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(216)

    image_rows = read_csv(ANALYSIS_DIR / "image_features.csv")
    effect_rows = read_csv(ANALYSIS_DIR / "effect_pairs_long.csv")
    distance_rows = read_csv(ANALYSIS_DIR / "responsiveness_distances.csv")
    distance_summary = read_csv(ANALYSIS_DIR / "responsiveness_distance_summary.csv")
    pca_rows = read_csv(ANALYSIS_DIR / "pca_image_scores.csv")

    paired_rows = paired_distance_tests(distance_rows, rng)
    targeted_rows = targeted_feature_tests(effect_rows, image_rows, rng)
    regression_rows = regression_tables(distance_rows)
    manova_rows = manova_permutation_tests(pca_rows)
    nonparam_rows = nonparametric_tests(distance_rows)
    interaction_rows = interaction_summary(effect_rows, image_rows)

    write_csv(STATS_DIR / "paired_model_distance_tests.csv", paired_rows)
    write_csv(STATS_DIR / "targeted_feature_tests.csv", targeted_rows)
    write_csv(STATS_DIR / "regression_distance_coefficients.csv", regression_rows)
    write_csv(STATS_DIR / "manova_pillai_permutation.csv", manova_rows)
    write_csv(STATS_DIR / "nonparametric_tests.csv", nonparam_rows)
    write_csv(STATS_DIR / "interaction_contrast_summary.csv", interaction_rows)

    plot_pca_scores()
    plot_pca_loadings()
    plot_distance_bars(distance_summary)
    plot_model_difference_forest(paired_rows)
    plot_targeted_heatmap(targeted_rows)
    plot_interaction_heatmap(interaction_rows)
    plot_manova(manova_rows)
    plot_regression_coefficients(regression_rows)
    plot_feature_correlation(image_rows)
    plot_nonparametric(nonparam_rows, distance_rows)
    contact_sheet()
    write_report(paired_rows, targeted_rows, manova_rows, regression_rows, nonparam_rows)

    summary = {
        "stats_dir": str(STATS_DIR),
        "figures_dir": str(FIG_DIR),
        "n_images": len(image_rows),
        "n_distance_rows": len(distance_rows),
        "n_paired_model_tests": len(paired_rows),
        "n_targeted_feature_tests": len(targeted_rows),
        "n_regression_rows": len(regression_rows),
        "n_manova_terms": len(manova_rows),
        "n_nonparametric_tests": len(nonparam_rows),
        "n_figures": len(list(FIG_DIR.glob("*.png"))),
    }
    (STATS_DIR / "analysis_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
