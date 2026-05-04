#!/usr/bin/env python3
"""Advanced statistical suite for the pooled filmstock benchmark.

The core pooled analysis is deliberately conservative. This script adds the
"above and beyond" layer: blocked permutation tests, hierarchical bootstrap,
variance decomposition, PLS, shrinkage classifiers, feature reliability, and
robust outlier diagnostics. It avoids pandas/statsmodels so it can run on the
current machine without installing anything.
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import pinv
from scipy.stats import chi2, f as f_dist, t as t_dist
from sklearn.covariance import MinCovDet
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = Path(os.environ.get("FILMSTOCK_ANALYSIS_DIR", REPO_ROOT / "data" / "analysis"))
CORE_STATS_DIR = ANALYSIS_DIR / "stats"
ADV_DIR = CORE_STATS_DIR / "advanced_suite"
FIG_DIR = ADV_DIR / "figures"

RNG_SEED = 216
N_PERM = 999
N_BOOT = 5000

MODEL_LABELS = {
    "chatgpt_image_2": "ChatGPT Image 2",
    "xai_grok_imagine": "Grok Imagine",
}
TARGET_LABELS = {
    "model_key": "Model",
    "film_key": "Filmstock",
    "light_key": "Lighting",
    "scan_key": "Scan",
    "scene_key": "Scene",
}
EFFECT_LABELS = {
    "filmstock_cinestill_minus_portra": "CineStill minus Portra",
    "lighting_warm_minus_cool": "Warm practical minus cool ambient",
    "scan_pushed_minus_clean": "Pushed scan minus clean scan",
}
METRIC_LABELS = {
    "distance_all_selected_features": "All selected features",
    "distance_color_features": "Color features",
    "distance_contrast_texture_features": "Contrast and texture",
    "distance_structure_features": "Structure features",
}


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
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return float("nan")
    return float(value)


def mean_ci(values: Iterable[float]) -> tuple[float, float, float, float, int]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), 0
    if arr.size == 1:
        val = float(arr[0])
        return val, float("nan"), float("nan"), float("nan"), 1
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1))
    se = sd / math.sqrt(arr.size)
    crit = t_dist.ppf(0.975, df=arr.size - 1)
    return mean, se, mean - crit * se, mean + crit * se, int(arr.size)


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


def design_columns(rows: list[dict[str, str]], terms: list[str]) -> tuple[np.ndarray, list[str]]:
    columns = [np.ones(len(rows))]
    names = ["Intercept"]

    def binary(field: str, high: str) -> np.ndarray:
        return np.asarray([1.0 if row[field] == high else 0.0 for row in rows], dtype=np.float64)

    binaries = {
        "model": binary("model_key", "xai_grok_imagine"),
        "film": binary("film_key", "cinestill800t"),
        "light": binary("light_key", "warm_practical"),
        "scan": binary("scan_key", "pushed_scan"),
        "replicate": binary("replicate", "2"),
        "effect_lighting": np.asarray([1.0 if row["effect_type"] == "lighting_warm_minus_cool" else 0.0 for row in rows], dtype=np.float64) if rows and "effect_type" in rows[0] else None,
        "effect_scan": np.asarray([1.0 if row["effect_type"] == "scan_pushed_minus_clean" else 0.0 for row in rows], dtype=np.float64) if rows and "effect_type" in rows[0] else None,
    }

    for term in terms:
        if term == "scene":
            columns.append(binary("scene_key", "backstage"))
            names.append("scene_backstage")
            columns.append(binary("scene_key", "corner_store"))
            names.append("scene_corner_store")
        elif term == "run_block":
            columns.append(np.asarray([1.0 if row["run_block"] == "2" else 0.0 for row in rows], dtype=np.float64))
            names.append("run_block_2")
            columns.append(np.asarray([1.0 if row["run_block"] == "3" else 0.0 for row in rows], dtype=np.float64))
            names.append("run_block_3")
        elif term == "effect_type":
            columns.append(binaries["effect_lighting"])
            names.append("effect_lighting")
            columns.append(binaries["effect_scan"])
            names.append("effect_scan")
        elif term == "model:effect_type":
            columns.append(binaries["model"] * binaries["effect_lighting"])
            names.append("model_by_lighting_effect")
            columns.append(binaries["model"] * binaries["effect_scan"])
            names.append("model_by_scan_effect")
        elif ":" in term:
            col = np.ones(len(rows), dtype=np.float64)
            for part in term.split(":"):
                col = col * binaries[part]
            columns.append(col)
            names.append(term)
        else:
            columns.append(binaries[term])
            names.append(term)
    return np.column_stack(columns), names


def rss_univariate(y: np.ndarray, x: np.ndarray) -> float:
    beta = pinv(x) @ y
    resid = y - x @ beta
    return float(resid.T @ resid)


def rss_multivariate(y: np.ndarray, x: np.ndarray) -> float:
    beta = pinv(x) @ y
    resid = y - x @ beta
    return float(np.sum(resid * resid))


def pca_matrix(image_rows: list[dict[str, str]], feature_cols: list[str], n_components: int = 20) -> tuple[np.ndarray, PCA, StandardScaler]:
    mat = np.asarray([[float(row[col]) for col in feature_cols] for row in image_rows], dtype=np.float64)
    scaler = StandardScaler()
    z = scaler.fit_transform(mat)
    pca = PCA(n_components=min(n_components, z.shape[0], z.shape[1]), random_state=RNG_SEED)
    scores = pca.fit_transform(z)
    return scores, pca, scaler


def blocked_permutation_permanova(
    image_rows: list[dict[str, str]],
    y: np.ndarray,
    n_perm: int = N_PERM,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(RNG_SEED)
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
        "film:light:scan",
    ]
    full_terms = ["scene", "run_block", "replicate", *tested_terms]
    x_full, _ = design_columns(image_rows, full_terms)
    rss_full = rss_multivariate(y, x_full)
    rank_full = np.linalg.matrix_rank(x_full)
    df_full_resid = max(y.shape[0] - rank_full, 1)

    block_indices: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(image_rows):
        block_indices[row["run_id"]].append(i)

    out: list[dict[str, object]] = []
    for term in tested_terms:
        reduced_terms = [t for t in full_terms if t != term]
        x_reduced, _ = design_columns(image_rows, reduced_terms)
        rss_reduced = rss_multivariate(y, x_reduced)
        rank_reduced = np.linalg.matrix_rank(x_reduced)
        df_term = max(rank_full - rank_reduced, 1)
        ss_term = max(rss_reduced - rss_full, 0.0)
        pseudo_f = (ss_term / df_term) / max(rss_full / df_full_resid, 1e-12)
        partial_r2 = ss_term / max(rss_reduced, 1e-12)

        beta_red = pinv(x_reduced) @ y
        fitted_red = x_reduced @ beta_red
        resid_red = y - fitted_red
        perm_stats = []
        for _ in range(n_perm):
            y_perm = fitted_red.copy()
            for idxs in block_indices.values():
                shuffled = rng.permutation(idxs)
                y_perm[idxs, :] += resid_red[shuffled, :]
            rss_perm_full = rss_multivariate(y_perm, x_full)
            rss_perm_reduced = rss_multivariate(y_perm, x_reduced)
            ss_perm = max(rss_perm_reduced - rss_perm_full, 0.0)
            f_perm = (ss_perm / df_term) / max(rss_perm_full / df_full_resid, 1e-12)
            perm_stats.append(f_perm)
        perm_stats_arr = np.asarray(perm_stats)
        p_perm = (1 + np.sum(perm_stats_arr >= pseudo_f)) / (n_perm + 1)
        out.append(
            {
                "term": term,
                "term_label": term.replace(":", " by "),
                "pseudo_f": safe_float(pseudo_f),
                "partial_r2": safe_float(partial_r2),
                "df_term": int(df_term),
                "df_residual": int(df_full_resid),
                "permutation_p_value": safe_float(p_perm),
                "n_permutations": n_perm,
                "response_space": f"PC1-PC{y.shape[1]} from all image features",
                "permutation_scheme": "Freedman-Lane residual permutation within run_id blocks",
            }
        )
    pvals = [float(row["permutation_p_value"]) for row in out]
    for row, fdr in zip(out, bh_adjust(pvals)):
        row["bh_fdr_p"] = safe_float(fdr)
    return out


def hierarchical_bootstrap_distance_tests(distance_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    rng = np.random.default_rng(RNG_SEED)
    out: list[dict[str, object]] = []
    metrics = list(METRIC_LABELS)
    run_ids = sorted({row["run_id"] for row in distance_rows})
    for effect_type in sorted({row["effect_type"] for row in distance_rows}):
        rows = [row for row in distance_rows if row["effect_type"] == effect_type]
        keys = sorted({
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
        index = {
            (
                row["model_key"],
                row["run_id"],
                row["scene_key"],
                row.get("film_key", ""),
                row.get("light_key", ""),
                row.get("scan_key", ""),
                row["replicate"],
            ): row
            for row in rows
        }
        keys_by_run: dict[str, list[tuple[str, str, str, str, str, str]]] = defaultdict(list)
        for key in keys:
            keys_by_run[key[0]].append(key)
        for metric in metrics:
            diffs_by_run: dict[str, list[float]] = defaultdict(list)
            diffs = []
            for key in keys:
                gpt = index.get(("chatgpt_image_2", *key))
                xai = index.get(("xai_grok_imagine", *key))
                if gpt and xai:
                    diff = float(xai[metric]) - float(gpt[metric])
                    diffs.append(diff)
                    diffs_by_run[key[0]].append(diff)
            arr = np.asarray(diffs, dtype=np.float64)
            boot = []
            for _ in range(N_BOOT):
                sample = []
                sampled_runs = rng.choice(run_ids, size=len(run_ids), replace=True)
                for run_id in sampled_runs:
                    vals = diffs_by_run[run_id]
                    sample.extend(rng.choice(vals, size=len(vals), replace=True))
                boot.append(float(np.mean(sample)))
            boot_arr = np.asarray(boot, dtype=np.float64)
            mean, se, lo, hi, n = mean_ci(arr)
            p_boot = 2 * min(np.mean(boot_arr >= 0), np.mean(boot_arr <= 0))
            out.append(
                {
                    "effect_type": effect_type,
                    "effect_label": EFFECT_LABELS.get(effect_type, effect_type),
                    "distance_metric": metric,
                    "metric_label": METRIC_LABELS.get(metric, metric),
                    "n_paired_conditions": n,
                    "mean_difference_xai_minus_chatgpt": safe_float(mean),
                    "se_classic": safe_float(se),
                    "classic_ci95_low": safe_float(lo),
                    "classic_ci95_high": safe_float(hi),
                    "hierarchical_boot_ci95_low": safe_float(np.percentile(boot_arr, 2.5)),
                    "hierarchical_boot_ci95_high": safe_float(np.percentile(boot_arr, 97.5)),
                    "hierarchical_boot_p_two_sided": safe_float(min(1.0, p_boot)),
                    "n_bootstrap": N_BOOT,
                }
            )
    return out


def fixed_effect_variance_decomposition(distance_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    rows = distance_rows
    terms = ["model", "effect_type", "model:effect_type", "scene", "run_block", "replicate"]
    full_x, _ = design_columns(rows, terms)
    out: list[dict[str, object]] = []
    for metric in METRIC_LABELS:
        y = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        rss_full = rss_univariate(y, full_x)
        rank_full = np.linalg.matrix_rank(full_x)
        df_resid = max(len(y) - rank_full, 1)
        for term in terms:
            red_terms = [t for t in terms if t != term]
            x_red, _ = design_columns(rows, red_terms)
            rss_red = rss_univariate(y, x_red)
            df_term = max(rank_full - np.linalg.matrix_rank(x_red), 1)
            ss_term = max(rss_red - rss_full, 0.0)
            f_stat = (ss_term / df_term) / max(rss_full / df_resid, 1e-12)
            p_val = 1 - f_dist.cdf(f_stat, df_term, df_resid)
            out.append(
                {
                    "distance_metric": metric,
                    "metric_label": METRIC_LABELS.get(metric, metric),
                    "term": term,
                    "term_label": term.replace(":", " by "),
                    "partial_r2": safe_float(ss_term / max(rss_red, 1e-12)),
                    "partial_eta2": safe_float(ss_term / max(ss_term + rss_full, 1e-12)),
                    "f_statistic": safe_float(f_stat),
                    "df_term": int(df_term),
                    "df_residual": int(df_resid),
                    "p_value": safe_float(p_val),
                }
            )
    pvals = [float(row["p_value"]) for row in out]
    for row, fdr in zip(out, bh_adjust(pvals)):
        row["bh_fdr_p"] = safe_float(fdr)
    return out


def image_feature_variance_components(
    image_rows: list[dict[str, str]],
    feature_cols: list[str],
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for model in sorted({row["model_key"] for row in image_rows}):
        model_rows = [row for row in image_rows if row["model_key"] == model]
        conditions = sorted({(row["scene_key"], row["film_key"], row["light_key"], row["scan_key"]) for row in model_rows})
        runs = sorted({row["run_id"] for row in model_rows})
        reps = sorted({row["replicate"] for row in model_rows})
        c_count, r_count, k_count = len(conditions), len(runs), len(reps)
        index = {
            (row["scene_key"], row["film_key"], row["light_key"], row["scan_key"], row["run_id"], row["replicate"]): row
            for row in model_rows
        }
        for feature in feature_cols:
            y = np.zeros((c_count, r_count, k_count), dtype=np.float64)
            complete = True
            for ci, cond in enumerate(conditions):
                for ri, run_id in enumerate(runs):
                    for ki, rep in enumerate(reps):
                        row = index.get((*cond, run_id, rep))
                        if row is None:
                            complete = False
                            break
                        y[ci, ri, ki] = float(row[feature])
            if not complete:
                continue
            grand = y.mean()
            cond_means = y.mean(axis=(1, 2))
            run_means = y.mean(axis=(0, 2))
            cell_means = y.mean(axis=2)
            ss_total = float(np.sum((y - grand) ** 2))
            if ss_total <= 1e-12:
                continue
            ss_condition = float(r_count * k_count * np.sum((cond_means - grand) ** 2))
            ss_run = float(c_count * k_count * np.sum((run_means - grand) ** 2))
            ss_condition_run = float(k_count * np.sum((cell_means - cond_means[:, None] - run_means[None, :] + grand) ** 2))
            ss_residual = float(np.sum((y - cell_means[:, :, None]) ** 2))
            out.append(
                {
                    "model_key": model,
                    "feature": feature,
                    "ss_total": safe_float(ss_total),
                    "condition_share": safe_float(ss_condition / ss_total),
                    "run_block_share": safe_float(ss_run / ss_total),
                    "condition_by_run_share": safe_float(ss_condition_run / ss_total),
                    "within_run_replicate_share": safe_float(ss_residual / ss_total),
                    "condition_to_seed_variance_ratio": safe_float(ss_condition / max(ss_run + ss_condition_run + ss_residual, 1e-12)),
                    "n_conditions": c_count,
                    "n_runs": r_count,
                    "n_replicates_per_run": k_count,
                }
            )
    return out


def label_for_target(row: dict[str, str], target: str) -> int:
    if target == "model_key":
        return 1 if row[target] == "xai_grok_imagine" else 0
    if target == "film_key":
        return 1 if row[target] == "cinestill800t" else 0
    if target == "light_key":
        return 1 if row[target] == "warm_practical" else 0
    if target == "scan_key":
        return 1 if row[target] == "pushed_scan" else 0
    if target == "scene_key":
        return 1 if row[target] == "corner_store" else 0
    raise ValueError(target)


def pls_vip(pls: PLSRegression, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    t_scores = pls.x_scores_
    w = pls.x_weights_
    q = pls.y_loadings_.reshape(-1)
    p = w.shape[0]
    ss = np.sum(t_scores ** 2, axis=0) * (q[: t_scores.shape[1]] ** 2)
    total_ss = np.sum(ss)
    if total_ss <= 1e-12:
        return np.zeros(p)
    vip = np.sqrt(p * ((w ** 2) @ ss) / total_ss)
    return vip


def pls_da_cv(
    image_rows: list[dict[str, str]],
    feature_cols: list[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    x_all = np.asarray([[float(row[col]) for col in feature_cols] for row in image_rows], dtype=np.float64)
    groups = np.asarray([row["run_id"] for row in image_rows])
    run_ids = sorted(set(groups))
    targets = ["model_key", "film_key", "light_key", "scan_key"]
    cv_rows: list[dict[str, object]] = []
    vip_rows: list[dict[str, object]] = []

    for target in targets:
        y_all = np.asarray([label_for_target(row, target) for row in image_rows], dtype=np.float64)
        best_auc = -np.inf
        best_components = 1
        for n_comp in range(1, 11):
            pred = np.zeros_like(y_all, dtype=np.float64)
            for run_id in run_ids:
                test = groups == run_id
                train = ~test
                scaler = StandardScaler()
                x_train = scaler.fit_transform(x_all[train])
                x_test = scaler.transform(x_all[test])
                pls = PLSRegression(n_components=min(n_comp, x_train.shape[1], x_train.shape[0] - 1))
                pls.fit(x_train, y_all[train])
                pred[test] = pls.predict(x_test).reshape(-1)
            pred_clip = np.clip(pred, 0, 1)
            acc = accuracy_score(y_all, pred_clip >= 0.5)
            auc = roc_auc_score(y_all, pred_clip) if len(set(y_all)) == 2 else float("nan")
            rmse = math.sqrt(float(np.mean((pred - y_all) ** 2)))
            cv_rows.append(
                {
                    "target": target,
                    "target_label": TARGET_LABELS[target],
                    "method": "PLS-DA",
                    "n_components": n_comp,
                    "leave_run_out_accuracy": safe_float(acc),
                    "leave_run_out_auc": safe_float(auc),
                    "leave_run_out_rmse": safe_float(rmse),
                }
            )
            if auc > best_auc:
                best_auc = auc
                best_components = n_comp

        scaler = StandardScaler()
        x_z = scaler.fit_transform(x_all)
        pls = PLSRegression(n_components=min(best_components, x_z.shape[1], x_z.shape[0] - 1))
        pls.fit(x_z, y_all)
        vip = pls_vip(pls, x_z, y_all)
        order = np.argsort(vip)[::-1][:40]
        for rank, idx in enumerate(order, start=1):
            vip_rows.append(
                {
                    "target": target,
                    "target_label": TARGET_LABELS[target],
                    "best_n_components_by_auc": best_components,
                    "rank": rank,
                    "feature": feature_cols[idx],
                    "vip": safe_float(vip[idx]),
                    "x_weight_component_1": safe_float(pls.x_weights_[idx, 0]),
                }
            )
    return cv_rows, vip_rows


def regularized_classification_cv(
    image_rows: list[dict[str, str]],
    feature_cols: list[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    x_all = np.asarray([[float(row[col]) for col in feature_cols] for row in image_rows], dtype=np.float64)
    groups = np.asarray([row["run_id"] for row in image_rows])
    run_ids = sorted(set(groups))
    targets = ["model_key", "film_key", "light_key", "scan_key"]
    penalties = [("LASSO logistic", 1.0), ("Ridge logistic", 0.0)]
    c_grid = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    result_rows: list[dict[str, object]] = []
    coef_rows: list[dict[str, object]] = []

    for target in targets:
        y_all = np.asarray([label_for_target(row, target) for row in image_rows], dtype=int)
        for method_label, l1_ratio in penalties:
            pred = np.zeros_like(y_all, dtype=np.float64)
            selected_cs = []
            for test_run in run_ids:
                test = groups == test_run
                train = ~test
                train_runs = [run for run in run_ids if run != test_run]
                c_scores = []
                for c in c_grid:
                    val_preds = []
                    val_true = []
                    for val_run in train_runs:
                        val = groups == val_run
                        inner_train = train & ~val
                        scaler = StandardScaler()
                        x_inner = scaler.fit_transform(x_all[inner_train])
                        x_val = scaler.transform(x_all[val])
                        clf = LogisticRegression(
                            solver="liblinear",
                            l1_ratio=l1_ratio,
                            C=c,
                            max_iter=2000,
                            class_weight="balanced",
                            random_state=RNG_SEED,
                        )
                        clf.fit(x_inner, y_all[inner_train])
                        val_preds.extend(clf.predict_proba(x_val)[:, 1])
                        val_true.extend(y_all[val])
                    c_scores.append(roc_auc_score(val_true, val_preds))
                best_c = c_grid[int(np.argmax(c_scores))]
                selected_cs.append(best_c)
                scaler = StandardScaler()
                x_train = scaler.fit_transform(x_all[train])
                x_test = scaler.transform(x_all[test])
                clf = LogisticRegression(
                    solver="liblinear",
                    l1_ratio=l1_ratio,
                    C=best_c,
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RNG_SEED,
                )
                clf.fit(x_train, y_all[train])
                pred[test] = clf.predict_proba(x_test)[:, 1]

            acc = accuracy_score(y_all, pred >= 0.5)
            auc = roc_auc_score(y_all, pred)
            result_rows.append(
                {
                    "target": target,
                    "target_label": TARGET_LABELS[target],
                    "method": method_label,
                    "leave_run_out_accuracy": safe_float(acc),
                    "leave_run_out_auc": safe_float(auc),
                    "selected_c_median": safe_float(np.median(selected_cs)),
                    "selected_c_values": ";".join(str(c) for c in selected_cs),
                }
            )

            best_c_all = float(np.median(selected_cs))
            scaler = StandardScaler()
            x_z = scaler.fit_transform(x_all)
            clf = LogisticRegression(
                solver="liblinear",
                l1_ratio=l1_ratio,
                C=best_c_all,
                max_iter=2000,
                class_weight="balanced",
                random_state=RNG_SEED,
            )
            clf.fit(x_z, y_all)
            coef = clf.coef_.reshape(-1)
            order = np.argsort(np.abs(coef))[::-1][:40]
            for rank, idx in enumerate(order, start=1):
                coef_rows.append(
                    {
                        "target": target,
                        "target_label": TARGET_LABELS[target],
                        "method": method_label,
                        "rank": rank,
                        "feature": feature_cols[idx],
                        "coefficient": safe_float(coef[idx]),
                        "abs_coefficient": safe_float(abs(coef[idx])),
                    }
                )
    return result_rows, coef_rows


def cross_run_feature_reliability(
    effect_rows: list[dict[str, str]],
    image_rows: list[dict[str, str]],
    feature_cols: list[str],
) -> list[dict[str, object]]:
    feature_sds = {}
    for feature in feature_cols:
        vals = np.asarray([float(row[feature]) for row in image_rows], dtype=np.float64)
        sd = float(vals.std(ddof=1))
        feature_sds[feature] = sd if sd > 0 else 1.0

    main_rows = [row for row in effect_rows if row["effect_family"] == "main_effect"]
    grouped: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in main_rows:
        grouped[(row["model_key"], row["effect_type"], row["feature"], row["run_id"])].append(float(row["effect_value"]))

    out: list[dict[str, object]] = []
    keys = sorted({(m, e, f) for (m, e, f, _) in grouped})
    for model, effect_type, feature in keys:
        run_means = []
        for run_id in sorted({row["run_id"] for row in main_rows}):
            vals = grouped.get((model, effect_type, feature, run_id), [])
            if vals:
                run_means.append(float(np.mean(vals)))
        if len(run_means) < 2:
            continue
        arr = np.asarray(run_means, dtype=np.float64)
        mean = float(arr.mean())
        sd = float(arr.std(ddof=1))
        signs = np.sign(arr)
        overall_sign = np.sign(mean)
        same_sign = int(np.sum(signs == overall_sign)) if overall_sign != 0 else 0
        _, _, lo, hi, n = mean_ci(arr)
        out.append(
            {
                "model_key": model,
                "effect_type": effect_type,
                "effect_label": EFFECT_LABELS.get(effect_type, effect_type),
                "feature": feature,
                "n_runs": n,
                "mean_run_effect": safe_float(mean),
                "sd_run_effect": safe_float(sd),
                "ci95_low_across_runs": safe_float(lo),
                "ci95_high_across_runs": safe_float(hi),
                "standardized_mean_run_effect": safe_float(mean / feature_sds[feature]),
                "abs_standardized_mean_run_effect": safe_float(abs(mean / feature_sds[feature])),
                "same_sign_run_count": same_sign,
                "all_runs_same_sign": int(same_sign == n),
                "coefficient_of_variation_abs": safe_float(sd / max(abs(mean), 1e-12)),
                "run_effects": ";".join(f"{x:.8g}" for x in arr),
            }
        )
    out.sort(key=lambda row: (int(row["all_runs_same_sign"]), float(row["abs_standardized_mean_run_effect"])), reverse=True)
    return out


def robust_outlier_diagnostics(
    image_rows: list[dict[str, str]],
    pc_scores: np.ndarray,
) -> list[dict[str, object]]:
    y = pc_scores[:, :10]
    mcd = MinCovDet(random_state=RNG_SEED, support_fraction=0.75).fit(y)
    robust_dist = mcd.mahalanobis(y)
    emp_center = y.mean(axis=0)
    emp_cov_inv = pinv(np.cov(y, rowvar=False, ddof=1))
    emp_diff = y - emp_center
    emp_dist = np.sum((emp_diff @ emp_cov_inv) * emp_diff, axis=1)
    cutoff = chi2.ppf(0.975, df=y.shape[1])
    rows: list[dict[str, object]] = []
    for i, row in enumerate(image_rows):
        rows.append(
            {
                "rank_robust_distance": 0,
                "variant_id": row["variant_id"],
                "run_id": row["run_id"],
                "model_key": row["model_key"],
                "scene_key": row["scene_key"],
                "film_key": row["film_key"],
                "light_key": row["light_key"],
                "scan_key": row["scan_key"],
                "replicate": row["replicate"],
                "image_path": row["image_path"],
                "robust_mahalanobis_sq_pc10": safe_float(robust_dist[i]),
                "empirical_mahalanobis_sq_pc10": safe_float(emp_dist[i]),
                "chi_square_df": y.shape[1],
                "chi_square_975_cutoff": safe_float(cutoff),
                "flag_robust_outlier_975": int(robust_dist[i] > cutoff),
                "robust_tail_p_value": safe_float(1 - chi2.cdf(robust_dist[i], df=y.shape[1])),
            }
        )
    rows.sort(key=lambda row: float(row["robust_mahalanobis_sq_pc10"]), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank_robust_distance"] = rank
    return rows


def plot_permanova(rows: list[dict[str, object]]) -> None:
    rows = sorted(rows, key=lambda row: float(row["partial_r2"]), reverse=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=180)
    y = np.arange(len(rows))
    vals = [float(row["partial_r2"]) for row in rows]
    ax.barh(y, vals, color="#4E79A7")
    ax.set_yticks(y)
    ax.set_yticklabels([row["term_label"] for row in rows], fontsize=8)
    ax.invert_yaxis()
    for i, row in enumerate(rows):
        ax.text(vals[i] + max(vals) * 0.02, i, f"p={float(row['permutation_p_value']):.3f}", va="center", fontsize=8)
    ax.set_xlabel("Partial R2 in PC1-PC10 feature space")
    ax.set_title("Blocked Permutation PERMANOVA")
    ax.grid(axis="x", color="#E6E6E6")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig13_blocked_permanova_pc10.png")
    plt.close(fig)


def plot_cv_results(pls_rows: list[dict[str, object]], reg_rows: list[dict[str, object]]) -> None:
    best_pls = {}
    for row in pls_rows:
        target = row["target"]
        if target not in best_pls or float(row["leave_run_out_auc"]) > float(best_pls[target]["leave_run_out_auc"]):
            best_pls[target] = row
    rows = [*best_pls.values(), *reg_rows]
    targets = ["model_key", "film_key", "light_key", "scan_key"]
    methods = ["PLS-DA", "LASSO logistic", "Ridge logistic"]
    fig, ax = plt.subplots(figsize=(9.5, 5), dpi=180)
    x = np.arange(len(targets))
    width = 0.24
    colors = {"PLS-DA": "#4E79A7", "LASSO logistic": "#C43C39", "Ridge logistic": "#59A14F"}
    for j, method in enumerate(methods):
        vals = []
        for target in targets:
            row = next(r for r in rows if r["target"] == target and r["method"] == method)
            vals.append(float(row["leave_run_out_auc"]))
        ax.bar(x + (j - 1) * width, vals, width=width, label=method, color=colors[method], alpha=0.88)
    ax.axhline(0.5, color="#333333", linewidth=0.9, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([TARGET_LABELS[t] for t in targets])
    ax.set_ylim(0.45, 1.02)
    ax.set_ylabel("Leave-run-out AUC")
    ax.set_title("Cross-Validated Predictability from Image Features")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", color="#E6E6E6")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig14_pls_lasso_ridge_cv_auc.png")
    plt.close(fig)


def plot_distance_variance_decomposition(rows: list[dict[str, object]]) -> None:
    metric = "distance_all_selected_features"
    subset = [row for row in rows if row["distance_metric"] == metric]
    subset.sort(key=lambda row: float(row["partial_eta2"]), reverse=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=180)
    y = np.arange(len(subset))
    vals = [float(row["partial_eta2"]) for row in subset]
    ax.barh(y, vals, color="#756BB1")
    ax.set_yticks(y)
    ax.set_yticklabels([row["term_label"] for row in subset])
    ax.invert_yaxis()
    ax.set_xlabel("Partial eta squared")
    ax.set_title("Variance Decomposition for All-Feature Responsiveness Distance")
    ax.grid(axis="x", color="#E6E6E6")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig15_distance_variance_decomposition.png")
    plt.close(fig)


def plot_feature_variance_components(rows: list[dict[str, object]]) -> None:
    models = ["chatgpt_image_2", "xai_grok_imagine"]
    components = ["condition_share", "run_block_share", "condition_by_run_share", "within_run_replicate_share"]
    labels = ["Prompt condition", "Run block", "Condition by run", "Within-run replicate"]
    colors = ["#4E79A7", "#F28E2B", "#59A14F", "#C43C39"]
    fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=180)
    bottom = np.zeros(len(models))
    for comp, label, color in zip(components, labels, colors):
        vals = []
        for model in models:
            vals.append(float(np.mean([float(row[comp]) for row in rows if row["model_key"] == model])))
        ax.bar(np.arange(len(models)), vals, bottom=bottom, color=color, label=label, alpha=0.9)
        bottom += np.asarray(vals)
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels([MODEL_LABELS[m] for m in models])
    ax.set_ylabel("Mean variance share across features")
    ax.set_title("Feature-Level Variance Components Across 288 Images")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig16_image_feature_variance_components.png")
    plt.close(fig)


def plot_feature_screening_volcano(rows: list[dict[str, str]]) -> None:
    colors = {
        "filmstock_cinestill_minus_portra": "#4E79A7",
        "lighting_warm_minus_cool": "#C43C39",
        "scan_pushed_minus_clean": "#59A14F",
    }
    fig, ax = plt.subplots(figsize=(8.8, 5.6), dpi=180)
    for effect, color in colors.items():
        subset = [row for row in rows if row["effect_type"] == effect]
        x = [float(row["abs_standardized_model_gap"]) for row in subset]
        y = [-math.log10(max(float(row["bh_fdr_p_all_features"]), 1e-12)) for row in subset]
        ax.scatter(x, y, s=18, alpha=0.65, color=color, label=EFFECT_LABELS[effect])
    ax.axhline(-math.log10(0.05), color="#333333", linestyle="--", linewidth=0.9)
    ax.set_xlabel("Absolute standardized model gap")
    ax.set_ylabel("-log10(FDR-adjusted p)")
    ax.set_title("Exploratory All-Feature Screening")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(color="#E6E6E6")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig17_all_feature_screening_volcano.png")
    plt.close(fig)


def plot_reliable_features(rows: list[dict[str, object]]) -> None:
    subset = [row for row in rows if int(row["all_runs_same_sign"]) == 1][:20]
    subset = subset[::-1]
    fig, ax = plt.subplots(figsize=(9, 7), dpi=180)
    vals = [float(row["standardized_mean_run_effect"]) for row in subset]
    labels = [f"{row['effect_label'].split(' minus ')[0]} | {row['feature'][:34]}" for row in subset]
    colors = ["#C43C39" if v > 0 else "#4E79A7" for v in vals]
    ax.barh(np.arange(len(subset)), vals, color=colors, alpha=0.86)
    ax.axvline(0, color="#333333", linewidth=0.9)
    ax.set_yticks(np.arange(len(subset)))
    ax.set_yticklabels(labels, fontsize=7.2)
    ax.set_xlabel("Mean effect across runs, standardized by feature SD")
    ax.set_title("Largest Effects with Same Sign in All Three Runs")
    ax.grid(axis="x", color="#E6E6E6")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig18_cross_run_reliable_features.png")
    plt.close(fig)


def plot_outliers(rows: list[dict[str, object]]) -> None:
    top = rows[:20][::-1]
    fig, ax = plt.subplots(figsize=(9, 6.5), dpi=180)
    vals = [float(row["robust_mahalanobis_sq_pc10"]) for row in top]
    labels = [f"{row['model_key'].replace('_', ' ')} | {row['scene_key']} | {row['film_key']}" for row in top]
    ax.barh(np.arange(len(top)), vals, color="#B07AA1", alpha=0.86)
    ax.axvline(float(top[0]["chi_square_975_cutoff"]), color="#333333", linestyle="--", linewidth=0.9, label="chi-square .975 cutoff")
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Robust Mahalanobis distance squared, PC1-PC10")
    ax.set_title("Robust Multivariate Outlier Diagnostics")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig19_robust_outlier_diagnostics.png")
    plt.close(fig)


def write_report(summary: dict[str, object], key_rows: dict[str, list[dict[str, object]]]) -> None:
    permanova = sorted(key_rows["permanova"], key=lambda row: float(row["partial_r2"]), reverse=True)[:6]
    permanova_lines = [
        f"- {row['term_label']}: partial R2 {float(row['partial_r2']):.3f}, blocked permutation p = {float(row['permutation_p_value']):.3f}."
        for row in permanova
    ]
    boot = [row for row in key_rows["bootstrap"] if row["distance_metric"] == "distance_all_selected_features"]
    boot_lines = [
        f"- {row['effect_label']}: mean Grok minus ChatGPT {float(row['mean_difference_xai_minus_chatgpt']):.3f}, "
        f"hierarchical bootstrap 95% CI [{float(row['hierarchical_boot_ci95_low']):.3f}, {float(row['hierarchical_boot_ci95_high']):.3f}], "
        f"p = {float(row['hierarchical_boot_p_two_sided']):.3f}."
        for row in boot
    ]
    pls_best = []
    for target in ["model_key", "film_key", "light_key", "scan_key"]:
        rows = [row for row in key_rows["pls_cv"] if row["target"] == target]
        best = max(rows, key=lambda row: float(row["leave_run_out_auc"]))
        pls_best.append(
            f"- {best['target_label']}: best PLS-DA AUC {float(best['leave_run_out_auc']):.3f} with {best['n_components']} components."
        )
    reg_best = []
    for row in sorted(key_rows["regularized_cv"], key=lambda r: float(r["leave_run_out_auc"]), reverse=True)[:6]:
        reg_best.append(
            f"- {row['target_label']} using {row['method']}: AUC {float(row['leave_run_out_auc']):.3f}, accuracy {float(row['leave_run_out_accuracy']):.3f}."
        )
    reliable = key_rows["reliability"][:8]
    reliable_lines = [
        f"- {row['model_key']}, {row['effect_label']} on {row['feature']}: standardized effect {float(row['standardized_mean_run_effect']):.3f}, same sign in {row['same_sign_run_count']}/3 runs."
        for row in reliable
    ]

    text = f"""# Advanced Statistical Suite

This suite extends the pooled 288-image analysis beyond the original project pass. It adds blocked permutation PERMANOVA, hierarchical bootstrap inference, fixed-effect variance decomposition, image-feature variance components, PLS-DA, LASSO/ridge logistic classification, cross-run feature reliability, and robust multivariate outlier diagnostics.

## Data Used

- Images: `{summary['n_images']}`
- Numeric image features: `{summary['n_features']}`
- Prompt-response distance rows: `{summary['n_distance_rows']}`
- Advanced output folder: `{ADV_DIR}`

## Blocked PERMANOVA

The PERMANOVA uses PC1-PC10 from all image features and permutes residuals within `run_id` blocks.

{chr(10).join(permanova_lines)}

## Hierarchical Bootstrap Model Comparisons

The bootstrap resamples run blocks, then matched prompt contrasts within sampled runs.

{chr(10).join(boot_lines)}

## PLS and Shrinkage Predictive Checks

{chr(10).join(pls_best)}

Top regularized classification results:

{chr(10).join(reg_best)}

## Cross-Run Feature Reliability

These are the strongest feature effects that keep the same direction across all three runs.

{chr(10).join(reliable_lines)}

## Output Tables

- `blocked_permanova_pc10.csv`
- `hierarchical_bootstrap_distance_tests.csv`
- `fixed_effect_distance_variance_decomposition.csv`
- `image_feature_variance_components.csv`
- `pls_da_leave_run_out_cv.csv`
- `pls_vip_top_features.csv`
- `regularized_leave_run_out_cv.csv`
- `regularized_top_coefficients.csv`
- `cross_run_feature_reliability.csv`
- `robust_mahalanobis_outliers.csv`

## Figures

- `fig13_blocked_permanova_pc10.png`
- `fig14_pls_lasso_ridge_cv_auc.png`
- `fig15_distance_variance_decomposition.png`
- `fig16_image_feature_variance_components.png`
- `fig17_all_feature_screening_volcano.png`
- `fig18_cross_run_reliable_features.png`
- `fig19_robust_outlier_diagnostics.png`
"""
    (ADV_DIR / "ADVANCED_STATISTICAL_SUITE_REPORT.md").write_text(text)


def main() -> None:
    ADV_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    image_rows = read_csv(ANALYSIS_DIR / "image_features.csv")
    feature_dict = read_csv(ANALYSIS_DIR / "feature_dictionary.csv")
    distance_rows = read_csv(ANALYSIS_DIR / "responsiveness_distances.csv")
    effect_rows = read_csv(ANALYSIS_DIR / "effect_pairs_long.csv")
    all_feature_screening = read_csv(CORE_STATS_DIR / "all_feature_model_screening.csv")

    feature_cols = [row["feature"] for row in feature_dict if row["feature"] in image_rows[0]]
    # Drop constant columns for predictive and multivariate methods.
    mat = np.asarray([[float(row[col]) for col in feature_cols] for row in image_rows], dtype=np.float64)
    keep = np.std(mat, axis=0, ddof=1) > 1e-12
    feature_cols = [col for col, ok in zip(feature_cols, keep) if ok]

    pc_scores, pca, _ = pca_matrix(image_rows, feature_cols, n_components=20)
    permanova_rows = blocked_permutation_permanova(image_rows, pc_scores[:, :10])
    bootstrap_rows = hierarchical_bootstrap_distance_tests(distance_rows)
    distance_variance_rows = fixed_effect_variance_decomposition(distance_rows)
    feature_variance_rows = image_feature_variance_components(image_rows, feature_cols)
    pls_cv_rows, pls_vip_rows = pls_da_cv(image_rows, feature_cols)
    regularized_cv_rows, regularized_coef_rows = regularized_classification_cv(image_rows, feature_cols)
    reliability_rows = cross_run_feature_reliability(effect_rows, image_rows, feature_cols)
    outlier_rows = robust_outlier_diagnostics(image_rows, pc_scores)

    write_csv(ADV_DIR / "blocked_permanova_pc10.csv", permanova_rows)
    write_csv(ADV_DIR / "hierarchical_bootstrap_distance_tests.csv", bootstrap_rows)
    write_csv(ADV_DIR / "fixed_effect_distance_variance_decomposition.csv", distance_variance_rows)
    write_csv(ADV_DIR / "image_feature_variance_components.csv", feature_variance_rows)
    write_csv(ADV_DIR / "pls_da_leave_run_out_cv.csv", pls_cv_rows)
    write_csv(ADV_DIR / "pls_vip_top_features.csv", pls_vip_rows)
    write_csv(ADV_DIR / "regularized_leave_run_out_cv.csv", regularized_cv_rows)
    write_csv(ADV_DIR / "regularized_top_coefficients.csv", regularized_coef_rows)
    write_csv(ADV_DIR / "cross_run_feature_reliability.csv", reliability_rows)
    write_csv(ADV_DIR / "robust_mahalanobis_outliers.csv", outlier_rows)

    plot_permanova(permanova_rows)
    plot_cv_results(pls_cv_rows, regularized_cv_rows)
    plot_distance_variance_decomposition(distance_variance_rows)
    plot_feature_variance_components(feature_variance_rows)
    plot_feature_screening_volcano(all_feature_screening)
    plot_reliable_features(reliability_rows)
    plot_outliers(outlier_rows)

    summary = {
        "advanced_dir": str(ADV_DIR),
        "figures_dir": str(FIG_DIR),
        "n_images": len(image_rows),
        "n_features": len(feature_cols),
        "n_distance_rows": len(distance_rows),
        "n_permanova_terms": len(permanova_rows),
        "n_bootstrap_tests": len(bootstrap_rows),
        "n_distance_variance_rows": len(distance_variance_rows),
        "n_feature_variance_rows": len(feature_variance_rows),
        "n_pls_cv_rows": len(pls_cv_rows),
        "n_pls_vip_rows": len(pls_vip_rows),
        "n_regularized_cv_rows": len(regularized_cv_rows),
        "n_regularized_coef_rows": len(regularized_coef_rows),
        "n_reliability_rows": len(reliability_rows),
        "n_outlier_rows": len(outlier_rows),
        "pca_pc10_cumulative_variance": safe_float(np.sum(pca.explained_variance_ratio_[:10])),
        "n_figures": len(list(FIG_DIR.glob("*.png"))),
    }
    (ADV_DIR / "advanced_suite_summary.json").write_text(json.dumps(summary, indent=2))
    write_report(
        summary,
        {
            "permanova": permanova_rows,
            "bootstrap": bootstrap_rows,
            "pls_cv": pls_cv_rows,
            "regularized_cv": regularized_cv_rows,
            "reliability": reliability_rows,
        },
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
