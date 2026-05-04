#!/usr/bin/env python3
"""Run inferential statistics for the pooled three-run filmstock benchmark."""

from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.linalg import pinv
from scipy.stats import friedmanchisquare, kruskal, t as t_dist

import run_statistical_analysis as base


REPO_ROOT = Path(__file__).resolve().parents[1]
POOLED_DIR = Path(os.environ.get("FILMSTOCK_POOLED_DIR", REPO_ROOT / "data" / "pooled_three_run_analysis"))
ANALYSIS_DIR = Path(os.environ.get("FILMSTOCK_ANALYSIS_DIR", REPO_ROOT / "data" / "analysis"))
STATS_DIR = ANALYSIS_DIR / "stats"
FIG_DIR = STATS_DIR / "figures"

MODEL_LABELS = base.MODEL_LABELS
EFFECT_LABELS = base.EFFECT_LABELS
METRIC_LABELS = base.METRIC_LABELS
FEATURE_LABELS = base.FEATURE_LABELS
TARGETED_TESTS = base.TARGETED_TESTS
SELECTED_CORR_FEATURES = base.SELECTED_CORR_FEATURES


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


def safe_float(value: float) -> float:
    return base.safe_float(value)


def holm_adjust(p_values: list[float]) -> list[float]:
    return base.holm_adjust(p_values)


def bh_adjust(p_values: list[float]) -> list[float]:
    return base.bh_adjust(p_values)


def mean_ci(values, alpha: float = 0.05):
    return base.mean_ci(values, alpha=alpha)


def paired_test(gpt_values: list[float], xai_values: list[float], rng: np.random.Generator) -> dict[str, object]:
    return base.paired_test(gpt_values, xai_values, rng)


def global_feature_sds(image_rows: list[dict[str, str]]) -> dict[str, float]:
    return base.global_feature_sds(image_rows)


def matched_key_for_effect(row: dict[str, str]) -> tuple[str, ...]:
    effect = row["effect_type"]
    if effect == "filmstock_cinestill_minus_portra":
        return (row["run_id"], row["scene_key"], row["light_key"], row["scan_key"], row["replicate"])
    if effect == "lighting_warm_minus_cool":
        return (row["run_id"], row["scene_key"], row["film_key"], row["scan_key"], row["replicate"])
    if effect == "scan_pushed_minus_clean":
        return (row["run_id"], row["scene_key"], row["film_key"], row["light_key"], row["replicate"])
    return (row.get("run_id", ""), row.get("scene_key", ""), row.get("replicate", ""))


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
            matched = sorted({
                matched_key_for_effect(row)
                for row in main_rows
                if row["effect_type"] == effect_type and row["feature"] == feature
            })
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
            out.append(
                {
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
            )

    pvals = [float(row["paired_t_p_value"]) for row in out]
    for row, holm_p, bh_p in zip(out, holm_adjust(pvals), bh_adjust(pvals)):
        row["holm_adjusted_p"] = safe_float(holm_p)
        row["bh_fdr_p"] = safe_float(bh_p)
    return out


def feature_family(feature: str) -> str:
    if feature.startswith(("rgb_", "lab_", "hue_", "palette_", "shadow_", "midtone_", "highlight_", "split_tone", "local_lab")):
        return "color"
    if feature.startswith(("luma_", "rms_", "dynamic_", "michelson_", "fft_", "texture_", "laplacian_", "local_contrast")):
        return "tone_texture"
    if feature.startswith(("edge_", "gradient_", "orientation_", "image_entropy")):
        return "structure"
    if "halation" in feature or "red_highlight" in feature or "warm_" in feature:
        return "film_response"
    if "spatial" in feature or "cov" in feature or "corr" in feature:
        return "spatial_covariance"
    return "other"


def all_feature_screening_tests(
    effect_rows: list[dict[str, str]],
    image_rows: list[dict[str, str]],
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    main_rows = [row for row in effect_rows if row.get("effect_family") == "main_effect"]
    features = sorted({row["feature"] for row in main_rows})
    feature_sds: dict[str, float] = {}
    for feature in features:
        vals = np.asarray([float(row[feature]) for row in image_rows if feature in row], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size > 1:
            sd = float(vals.std(ddof=1))
            feature_sds[feature] = sd if sd > 0 else 1.0
    indexed: dict[tuple[str, str, str, tuple[str, ...]], float] = {}
    for row in main_rows:
        indexed[(row["effect_type"], row["feature"], row["model_key"], matched_key_for_effect(row))] = float(row["effect_value"])

    out: list[dict[str, object]] = []
    for effect_type in sorted({row["effect_type"] for row in main_rows}):
        for feature in features:
            matched = sorted({
                matched_key_for_effect(row)
                for row in main_rows
                if row["effect_type"] == effect_type and row["feature"] == feature
            })
            gpt_vals: list[float] = []
            xai_vals: list[float] = []
            for key in matched:
                gpt = indexed.get((effect_type, feature, "chatgpt_image_2", key))
                xai = indexed.get((effect_type, feature, "xai_grok_imagine", key))
                if gpt is not None and xai is not None:
                    gpt_vals.append(gpt)
                    xai_vals.append(xai)
            if len(gpt_vals) < 2:
                continue
            test = paired_test(gpt_vals, xai_vals, rng)
            sd = feature_sds.get(feature, 1.0)
            out.append(
                {
                    "effect_type": effect_type,
                    "effect_label": EFFECT_LABELS.get(effect_type, effect_type),
                    "feature": feature,
                    "feature_label": FEATURE_LABELS.get(feature, feature),
                    "feature_family": feature_family(feature),
                    "mean_chatgpt_raw": safe_float(np.mean(gpt_vals)),
                    "mean_xai_raw": safe_float(np.mean(xai_vals)),
                    "mean_chatgpt_standardized": safe_float(np.mean(np.asarray(gpt_vals) / sd)),
                    "mean_xai_standardized": safe_float(np.mean(np.asarray(xai_vals) / sd)),
                    "abs_standardized_model_gap": safe_float(abs((np.mean(xai_vals) - np.mean(gpt_vals)) / sd)),
                    **test,
                }
            )

    pvals = [float(row["paired_t_p_value"]) if np.isfinite(float(row["paired_t_p_value"])) else 1.0 for row in out]
    for row, holm_p, bh_p in zip(out, holm_adjust(pvals), bh_adjust(pvals)):
        row["holm_adjusted_p_all_features"] = safe_float(holm_p)
        row["bh_fdr_p_all_features"] = safe_float(bh_p)
    return sorted(out, key=lambda row: float(row["abs_standardized_model_gap"]), reverse=True)


@dataclass
class RegressionResult:
    method: str
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
        "Run block 2",
        "Run block 3",
        "Within-run replicate 2",
    ]
    x = []
    for row in rows:
        model = 1.0 if row["model_key"] == "xai_grok_imagine" else 0.0
        lighting = 1.0 if row["effect_type"] == "lighting_warm_minus_cool" else 0.0
        scan = 1.0 if row["effect_type"] == "scan_pushed_minus_clean" else 0.0
        backstage = 1.0 if row["scene_key"] == "backstage" else 0.0
        corner = 1.0 if row["scene_key"] == "corner_store" else 0.0
        run2 = 1.0 if str(row["run_block"]) == "2" else 0.0
        run3 = 1.0 if str(row["run_block"]) == "3" else 0.0
        rep2 = 1.0 if str(row["replicate"]) == "2" else 0.0
        x.append([1.0, model, lighting, scan, model * lighting, model * scan, backstage, corner, run2, run3, rep2])
    return y, np.asarray(x, dtype=np.float64), terms


def fit_linear_model(
    y: np.ndarray,
    x: np.ndarray,
    method: str,
    weights: np.ndarray | None = None,
) -> RegressionResult:
    n, _ = x.shape
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
    metrics = list(METRIC_LABELS.keys())
    out: list[dict[str, object]] = []
    for metric in metrics:
        y, x, terms = design_distance_model(distance_rows, metric)
        ols = fit_linear_model(y, x, "OLS HC3")

        group_vars: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        for row, resid in zip(distance_rows, ols.residuals):
            group_vars[(row["effect_type"], row["model_key"], row["run_id"])].append(float(resid))
        variances = {}
        for key, vals in group_vars.items():
            arr = np.asarray(vals, dtype=np.float64)
            variances[key] = float(arr.var(ddof=1)) if arr.size > 1 else 1.0
        weights = np.asarray([
            1.0 / max(variances[(row["effect_type"], row["model_key"], row["run_id"])], 1e-8)
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
        elif term == "run_block":
            columns.append(np.asarray([1.0 if row["run_block"] == "2" else 0.0 for row in rows]))
            names.append("run_block_2")
            columns.append(np.asarray([1.0 if row["run_block"] == "3" else 0.0 for row in rows]))
            names.append("run_block_3")
        elif term == "replicate":
            columns.append(np.asarray([1.0 if row["replicate"] == "2" else 0.0 for row in rows]))
            names.append("within_run_replicate_2")
        elif ":" in term:
            col = np.ones(len(rows))
            for part in term.split(":"):
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
    full_terms = ["scene", "run_block", "replicate", *tested_terms]
    x_full, _ = design_matrix_for_manova(pca_rows, full_terms)
    block_indices: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(pca_rows):
        block_indices[row["run_id"]].append(i)
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
            y_perm = fitted_red.copy()
            for idxs in block_indices.values():
                shuffled = rng.permutation(idxs)
                y_perm[idxs, :] += resid_red[shuffled, :]
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
                "covariates": "scene fixed effects, run-block fixed effects, within-run replicate fixed effect",
                "permutation_scheme": "residual permutations constrained within run_id blocks",
            }
        )
    pvals = [float(row["permutation_p_value"]) for row in results]
    for row, holm_p, bh_p in zip(results, holm_adjust(pvals), bh_adjust(pvals)):
        row["holm_adjusted_p"] = safe_float(holm_p)
        row["bh_fdr_p"] = safe_float(bh_p)
    return results


def paired_distance_tests(
    distance_rows: list[dict[str, str]],
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for effect_type in sorted({row["effect_type"] for row in distance_rows}):
        rows = [row for row in distance_rows if row["effect_type"] == effect_type]
        match_keys = sorted({
            (
                row["run_id"],
                row["scene_key"],
                row.get("film_key", ""),
                row.get("light_key", ""),
                row.get("scan_key", ""),
                row["replicate"],
            )
            for row in rows
        })
        for metric in METRIC_LABELS:
            gpt_vals: list[float] = []
            xai_vals: list[float] = []
            index = {
                (
                    row["model_key"],
                    row["run_id"],
                    row["scene_key"],
                    row.get("film_key", ""),
                    row.get("light_key", ""),
                    row.get("scan_key", ""),
                    row["replicate"],
                ): float(row[metric])
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
    for row, holm_p, bh_p in zip(out, holm_adjust(pvals), bh_adjust(pvals)):
        row["holm_adjusted_p"] = safe_float(holm_p)
        row["bh_fdr_p"] = safe_float(bh_p)
    return out


def nonparametric_tests(distance_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for model in sorted({row["model_key"] for row in distance_rows}):
        model_rows = [row for row in distance_rows if row["model_key"] == model]
        for metric in METRIC_LABELS:
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

            blocks = sorted({(row["run_id"], row["scene_key"], row["replicate"]) for row in model_rows})
            effect_values: dict[str, list[float]] = {effect: [] for effect in effects}
            for block in blocks:
                for effect in effects:
                    vals = [
                        float(row[metric])
                        for row in model_rows
                        if row["effect_type"] == effect and (row["run_id"], row["scene_key"], row["replicate"]) == block
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
                        "grouping": "effect_type within run by scene by replicate blocks",
                        "statistic": safe_float(fr.statistic),
                        "p_value": safe_float(fr.pvalue),
                        "n_groups": len(effects),
                        "n_blocks": len(blocks),
                    }
                )
    pvals = [float(row["p_value"]) for row in out]
    for row, holm_p, bh_p in zip(out, holm_adjust(pvals), bh_adjust(pvals)):
        row["holm_adjusted_p"] = safe_float(holm_p)
        row["bh_fdr_p"] = safe_float(bh_p)
    return out


def interaction_summary(effect_rows: list[dict[str, str]], image_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    return base.interaction_summary(effect_rows, image_rows)


def run_block_variation_summary(distance_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for model in sorted({row["model_key"] for row in distance_rows}):
        for effect_type in sorted({row["effect_type"] for row in distance_rows}):
            subset = [row for row in distance_rows if row["model_key"] == model and row["effect_type"] == effect_type]
            for metric in METRIC_LABELS:
                by_run: dict[str, list[float]] = defaultdict(list)
                for row in subset:
                    by_run[row["run_id"]].append(float(row[metric]))
                all_values = np.asarray([v for vals in by_run.values() for v in vals], dtype=np.float64)
                run_means = np.asarray([np.mean(vals) for vals in by_run.values()], dtype=np.float64)
                within_num = 0.0
                within_den = 0
                for vals in by_run.values():
                    arr = np.asarray(vals, dtype=np.float64)
                    if arr.size > 1:
                        within_num += float((arr.size - 1) * arr.var(ddof=1))
                        within_den += arr.size - 1
                within_var = within_num / within_den if within_den else float("nan")
                between_var = float(run_means.var(ddof=1)) if run_means.size > 1 else float("nan")
                denom = between_var + within_var if np.isfinite(between_var) and np.isfinite(within_var) else float("nan")
                out.append(
                    {
                        "model_key": model,
                        "effect_type": effect_type,
                        "effect_label": EFFECT_LABELS.get(effect_type, effect_type),
                        "distance_metric": metric,
                        "metric_label": METRIC_LABELS.get(metric, metric),
                        "n_runs": len(by_run),
                        "n_distances": int(all_values.size),
                        "grand_mean": safe_float(all_values.mean()),
                        "total_sd": safe_float(all_values.std(ddof=1)) if all_values.size > 1 else float("nan"),
                        "between_run_sd_of_means": safe_float(np.sqrt(between_var)) if np.isfinite(between_var) else float("nan"),
                        "pooled_within_run_sd": safe_float(np.sqrt(within_var)) if np.isfinite(within_var) else float("nan"),
                        "run_variance_share_descriptive": safe_float(between_var / denom) if denom and np.isfinite(denom) and denom > 0 else float("nan"),
                    }
                )
    return out


def condition_seed_variation(image_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    features = [
        f for f in [
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
            "hue_teal_cyan_share",
            "highlight_minus_shadow_warmth",
            "highlight_minus_shadow_lab_b",
            "split_tone_distance_lab_ab",
            "luma_kurtosis_excess",
            "highlight_rolloff_ratio",
            "fft_high_freq_power_share",
            "fft_high_mid_power_ratio",
            "palette_entropy_6",
            "palette_mean_pairwise_lab_distance",
        ]
        if f in image_rows[0]
    ]
    mat = np.asarray([[float(row[f]) for f in features] for row in image_rows], dtype=np.float64)
    means = np.nanmean(mat, axis=0)
    sds = np.nanstd(mat, axis=0, ddof=1)
    sds[sds == 0] = 1.0
    z_by_variant = {
        row["variant_id"]: (mat[i] - means) / sds
        for i, row in enumerate(image_rows)
    }
    groups: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in image_rows:
        groups[(row["model_key"], row["scene_key"], row["film_key"], row["light_key"], row["scan_key"])].append(row)

    out: list[dict[str, object]] = []
    for key, rows in sorted(groups.items()):
        points = np.asarray([z_by_variant[row["variant_id"]] for row in rows], dtype=np.float64)
        centroid = points.mean(axis=0)
        centroid_dists = np.sqrt(np.sum((points - centroid) ** 2, axis=1))
        pairwise = []
        for i in range(points.shape[0]):
            for j in range(i + 1, points.shape[0]):
                pairwise.append(float(np.sqrt(np.sum((points[i] - points[j]) ** 2))))
        run_centroids = []
        for run_id in sorted({row["run_id"] for row in rows}):
            run_points = np.asarray([z_by_variant[row["variant_id"]] for row in rows if row["run_id"] == run_id], dtype=np.float64)
            run_centroids.append(run_points.mean(axis=0))
        run_centroids_arr = np.asarray(run_centroids, dtype=np.float64)
        run_centroid_dists = np.sqrt(np.sum((run_centroids_arr - centroid) ** 2, axis=1))
        model, scene, film, light, scan = key
        out.append(
            {
                "model_key": model,
                "scene_key": scene,
                "film_key": film,
                "light_key": light,
                "scan_key": scan,
                "condition_id": f"{scene}_{film}_{light}_{scan}",
                "n_images": len(rows),
                "n_runs": len(run_centroids),
                "n_features": len(features),
                "mean_distance_to_condition_centroid": safe_float(centroid_dists.mean()),
                "sd_distance_to_condition_centroid": safe_float(centroid_dists.std(ddof=1)) if centroid_dists.size > 1 else float("nan"),
                "mean_pairwise_seed_distance": safe_float(np.mean(pairwise)) if pairwise else float("nan"),
                "sd_pairwise_seed_distance": safe_float(np.std(pairwise, ddof=1)) if len(pairwise) > 1 else float("nan"),
                "mean_run_centroid_distance": safe_float(run_centroid_dists.mean()),
                "sd_run_centroid_distance": safe_float(run_centroid_dists.std(ddof=1)) if run_centroid_dists.size > 1 else float("nan"),
            }
        )
    return out


def seed_variation_model_summary(condition_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for model in sorted({row["model_key"] for row in condition_rows}):
        rows = [row for row in condition_rows if row["model_key"] == model]
        for metric in [
            "mean_distance_to_condition_centroid",
            "mean_pairwise_seed_distance",
            "mean_run_centroid_distance",
        ]:
            vals = [float(row[metric]) for row in rows]
            mean, se, lo, hi, n = mean_ci(vals)
            out.append(
                {
                    "model_key": model,
                    "variation_metric": metric,
                    "n_conditions": n,
                    "mean": safe_float(mean),
                    "se": safe_float(se),
                    "ci95_low": safe_float(lo),
                    "ci95_high": safe_float(hi),
                }
            )
    return out


def plot_seed_and_run_variation(
    run_rows: list[dict[str, object]],
    seed_summary: list[dict[str, object]],
) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    colors = {"chatgpt_image_2": "#2A6FBB", "xai_grok_imagine": "#C43C39"}
    effects = [
        "filmstock_cinestill_minus_portra",
        "lighting_warm_minus_cool",
        "scan_pushed_minus_clean",
    ]
    models = ["chatgpt_image_2", "xai_grok_imagine"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=180)
    metric = "distance_all_selected_features"
    x = np.arange(len(effects))
    width = 0.36
    for j, model in enumerate(models):
        vals = []
        for effect in effects:
            row = next(
                r for r in run_rows
                if r["model_key"] == model and r["effect_type"] == effect and r["distance_metric"] == metric
            )
            vals.append(float(row["between_run_sd_of_means"]))
        axes[0].bar(x + (j - 0.5) * width, vals, width=width, color=colors[model], alpha=0.85, label=MODEL_LABELS[model])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["Filmstock", "Lighting", "Scan"])
    axes[0].set_ylabel("SD of run means")
    axes[0].set_title("Run-Block Variation in Responsiveness")
    axes[0].grid(axis="y", color="#E5E5E5")

    seed_metric = "mean_pairwise_seed_distance"
    xs = np.arange(len(models))
    means = []
    errors = []
    for model in models:
        row = next(r for r in seed_summary if r["model_key"] == model and r["variation_metric"] == seed_metric)
        means.append(float(row["mean"]))
        errors.append(float(row["mean"]) - float(row["ci95_low"]))
    axes[1].bar(xs, means, color=[colors[m] for m in models], alpha=0.85)
    axes[1].errorbar(xs, means, yerr=errors, fmt="none", ecolor="#222222", capsize=3, linewidth=0.9)
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels([MODEL_LABELS[m] for m in models])
    axes[1].set_ylabel("Mean pairwise distance")
    axes[1].set_title("Seed-to-Seed Variation Within Exact Prompts")
    axes[1].grid(axis="y", color="#E5E5E5")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig12_seed_and_run_variation.png")
    plt.close(fig)


def contact_sheet(image_rows: list[dict[str, str]]) -> None:
    rows = [
        row for row in image_rows
        if row["run_block"] == "1"
        and row["scene_key"] == "corner_store"
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
        draw.multiline_text((left_w + j * thumb + 8, 8), f"{film}\n{light}", fill=(20, 20, 20), font=font, spacing=3)
    for i, model in enumerate(models):
        y0 = label_h + i * (thumb + label_h)
        draw.text((8, y0 + thumb // 2 - 8), MODEL_LABELS[model], fill=(20, 20, 20), font=font)
        for j, (film, light) in enumerate(cols):
            row = by_key[(model, film, light)]
            im = Image.open(row["image_path"]).convert("RGB")
            im.thumbnail((thumb, thumb))
            canvas.paste(im, (left_w + j * thumb, y0))
    canvas.save(FIG_DIR / "fig11_corner_store_contact_sheet.png")


def write_report(
    paired_rows: list[dict[str, object]],
    targeted_rows: list[dict[str, object]],
    manova_rows: list[dict[str, object]],
    regression_rows: list[dict[str, object]],
    nonparam_rows: list[dict[str, object]],
    run_rows: list[dict[str, object]],
    seed_summary: list[dict[str, object]],
) -> None:
    all_metric = [row for row in paired_rows if row["distance_metric"] == "distance_all_selected_features"]
    all_metric_lines = [
        f"- {row['effect_label']}: ChatGPT mean {float(row['mean_chatgpt']):.3f}, "
        f"Grok mean {float(row['mean_xai']):.3f}, paired difference "
        f"{float(row['mean_difference_xai_minus_chatgpt']):.3f}, "
        f"95% CI [{float(row['ci95_low']):.3f}, {float(row['ci95_high']):.3f}], "
        f"p = {float(row['paired_t_p_value']):.3f}."
        for row in all_metric
    ]

    strongest_target = sorted(targeted_rows, key=lambda row: abs(float(row["mean_difference_xai_minus_chatgpt"])), reverse=True)[:6]
    target_lines = [
        f"- {row['effect_label']} on {row['feature_label']}: model difference "
        f"{float(row['mean_difference_xai_minus_chatgpt']):.4f}, "
        f"p = {float(row['paired_t_p_value']):.3f}, Holm p = {float(row['holm_adjusted_p']):.3f}."
        for row in strongest_target
    ]

    manova_lines = [
        f"- {row['term_label']}: Pillai trace {float(row['pillai_trace']):.3f}, "
        f"blocked permutation p = {float(row['permutation_p_value']):.3f}."
        for row in sorted(manova_rows, key=lambda r: float(r["pillai_trace"]), reverse=True)[:6]
    ]

    coef_rows = [
        row for row in regression_rows
        if row["metric"] == "distance_all_selected_features"
        and row["method"] == "OLS HC3"
        and row["term"] in {"xAI model", "xAI by lighting effect", "xAI by scan effect", "Run block 2", "Run block 3"}
    ]
    coef_lines = [
        f"- {row['term']}: beta {float(row['estimate']):.3f}, "
        f"95% CI [{float(row['ci95_low']):.3f}, {float(row['ci95_high']):.3f}], "
        f"p = {float(row['p_value']):.3f}."
        for row in coef_rows
    ]

    seed_lines = []
    for row in seed_summary:
        if row["variation_metric"] == "mean_pairwise_seed_distance":
            seed_lines.append(
                f"- {MODEL_LABELS[row['model_key']]}: mean pairwise exact-prompt seed distance "
                f"{float(row['mean']):.3f}, 95% CI [{float(row['ci95_low']):.3f}, {float(row['ci95_high']):.3f}]."
            )
    run_metric_lines = []
    for row in run_rows:
        if row["distance_metric"] == "distance_all_selected_features":
            run_metric_lines.append(
                f"- {MODEL_LABELS[row['model_key']]}, {row['effect_label']}: between-run SD of means "
                f"{float(row['between_run_sd_of_means']):.3f}; pooled within-run SD "
                f"{float(row['pooled_within_run_sd']):.3f}."
            )

    text = f"""# Pooled Statistical Analysis Report

This folder contains the pooled inferential analysis for the three-run filmstock responsiveness benchmark. The analysis uses 288 generated images: three independent generation runs, two models, three scenes, two film stocks, two lighting conditions, two scan conditions, and two within-run replicates.

The key design change from the first analysis is that `run_id` is now treated as a blocking factor. Main-effect contrasts are paired within run and within replicate, regression models include run-block fixed effects, nonparametric repeated-measures tests block by run-scene-replicate, and the permutation MANOVA permutes residuals within run blocks.

## Model Comparison on Total Responsiveness

{chr(10).join(all_metric_lines)}

## Targeted Filmstock and Lighting Features

{chr(10).join(target_lines)}

`all_feature_model_screening.csv` screens every extracted feature, not just the pre-registered targeted set. Use that table as the exploratory layer for deciding which richer image descriptors tell the clearest story in the final writeup.

## PCA and Blocked MANOVA-Style Testing

{chr(10).join(manova_lines)}

## Regression, WLS, and Robust Estimation

{chr(10).join(coef_lines)}

The regression table reports OLS with HC3 robust standard errors, two-stage WLS, and Huber IRLS robust regression. Run blocks are included directly as fixed effects, so the model comparison is not simply pooling all generations as if they came from one batch.

## Seed-to-Seed and Run-to-Run Variation

{chr(10).join(seed_lines)}

For prompt-response distances, the run-level descriptive variance checks are:

{chr(10).join(run_metric_lines)}

## Nonparametric Checks

`nonparametric_tests.csv` includes Kruskal-Wallis tests across prompt effect types and Friedman repeated-measures tests blocked by run, scene, and within-run replicate.

## Figure List

- `fig01_pca_scores_ellipses.png`: PCA feature space and 95 percent mean ellipses.
- `fig02_pca_loadings.png`: top feature loadings for PC1 and PC2.
- `fig03_responsiveness_distance_bars.png`: blocked mean prompt-response distances.
- `fig04_model_difference_forest.png`: paired Grok minus ChatGPT model differences.
- `fig05_targeted_feature_effect_heatmap.png`: targeted prompt effects in SD units.
- `fig06_interaction_effect_heatmap.png`: average absolute standardized interaction contrasts.
- `fig07_manova_pillai.png`: blocked permutation MANOVA Pillai trace results.
- `fig08_regression_coefficients.png`: OLS, WLS, and robust regression coefficients.
- `fig09_feature_correlation_heatmap.png`: selected-feature correlation matrix.
- `fig10_nonparametric_effect_ordering.png`: effect ordering used in nonparametric checks.
- `fig11_corner_store_contact_sheet.png`: visual examples from the first complete run.
- `fig12_seed_and_run_variation.png`: seed-to-seed and run-block variation summary.
"""
    (STATS_DIR / "STATISTICAL_ANALYSIS_REPORT.md").write_text(text)


def configure_base_plot_paths() -> None:
    base.ANALYSIS_DIR = ANALYSIS_DIR
    base.STATS_DIR = STATS_DIR
    base.FIG_DIR = FIG_DIR


def main() -> None:
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    configure_base_plot_paths()
    rng = np.random.default_rng(216)

    image_rows = read_csv(ANALYSIS_DIR / "image_features.csv")
    effect_rows = read_csv(ANALYSIS_DIR / "effect_pairs_long.csv")
    distance_rows = read_csv(ANALYSIS_DIR / "responsiveness_distances.csv")
    distance_summary = read_csv(ANALYSIS_DIR / "responsiveness_distance_summary.csv")
    pca_rows = read_csv(ANALYSIS_DIR / "pca_image_scores.csv")

    paired_rows = paired_distance_tests(distance_rows, rng)
    targeted_rows = targeted_feature_tests(effect_rows, image_rows, rng)
    feature_screening_rows = all_feature_screening_tests(effect_rows, image_rows, rng)
    regression_rows = regression_tables(distance_rows)
    manova_rows = manova_permutation_tests(pca_rows)
    nonparam_rows = nonparametric_tests(distance_rows)
    interaction_rows = interaction_summary(effect_rows, image_rows)
    run_variation_rows = run_block_variation_summary(distance_rows)
    condition_variation_rows = condition_seed_variation(image_rows)
    seed_summary_rows = seed_variation_model_summary(condition_variation_rows)

    write_csv(STATS_DIR / "paired_model_distance_tests.csv", paired_rows)
    write_csv(STATS_DIR / "targeted_feature_tests.csv", targeted_rows)
    write_csv(STATS_DIR / "all_feature_model_screening.csv", feature_screening_rows)
    write_csv(STATS_DIR / "regression_distance_coefficients.csv", regression_rows)
    write_csv(STATS_DIR / "manova_pillai_permutation.csv", manova_rows)
    write_csv(STATS_DIR / "nonparametric_tests.csv", nonparam_rows)
    write_csv(STATS_DIR / "interaction_contrast_summary.csv", interaction_rows)
    write_csv(STATS_DIR / "run_block_variation_summary.csv", run_variation_rows)
    write_csv(STATS_DIR / "condition_seed_variation.csv", condition_variation_rows)
    write_csv(STATS_DIR / "seed_variation_model_summary.csv", seed_summary_rows)

    base.plot_pca_scores()
    base.plot_pca_loadings()
    base.plot_distance_bars(distance_summary)
    base.plot_model_difference_forest(paired_rows)
    base.plot_targeted_heatmap(targeted_rows)
    base.plot_interaction_heatmap(interaction_rows)
    base.plot_manova(manova_rows)
    base.plot_regression_coefficients(regression_rows)
    base.plot_feature_correlation(image_rows)
    base.plot_nonparametric(nonparam_rows, distance_rows)
    contact_sheet(image_rows)
    plot_seed_and_run_variation(run_variation_rows, seed_summary_rows)
    write_report(paired_rows, targeted_rows, manova_rows, regression_rows, nonparam_rows, run_variation_rows, seed_summary_rows)

    summary = {
        "stats_dir": str(STATS_DIR),
        "figures_dir": str(FIG_DIR),
        "n_images": len(image_rows),
        "n_distance_rows": len(distance_rows),
        "n_paired_model_tests": len(paired_rows),
        "n_targeted_feature_tests": len(targeted_rows),
        "n_all_feature_screening_tests": len(feature_screening_rows),
        "n_regression_rows": len(regression_rows),
        "n_manova_terms": len(manova_rows),
        "n_nonparametric_tests": len(nonparam_rows),
        "n_run_variation_rows": len(run_variation_rows),
        "n_condition_seed_variation_rows": len(condition_variation_rows),
        "n_figures": len(list(FIG_DIR.glob("*.png"))),
    }
    (STATS_DIR / "analysis_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
