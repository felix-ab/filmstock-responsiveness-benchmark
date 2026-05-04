#!/usr/bin/env python3
"""Generate V2 candidate figures and figure-selection metadata."""

from __future__ import annotations

import csv
import math
import os
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import chi2, t as t_dist


REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = Path(os.environ.get("FILMSTOCK_ANALYSIS_DIR", REPO_ROOT / "data" / "analysis"))
STATS_DIR = ANALYSIS_DIR / "stats"
ADV_DIR = STATS_DIR / "advanced_suite"
ASSET_DIR = Path(os.environ.get("FILMSTOCK_ASSET_DIR", REPO_ROOT / "figures"))
FIG_DIR = Path(os.environ.get("FILMSTOCK_FIGURE_DIR", REPO_ROOT / "figures" / "paper_figures"))
SELECTED_DIR = ASSET_DIR / "selected_figures"

MODEL_LABELS = {
    "chatgpt_image_2": "ChatGPT Image 2",
    "xai_grok_imagine": "Grok Imagine",
}
MODEL_COLORS = {"chatgpt_image_2": "#2A6FBB", "xai_grok_imagine": "#C43C39"}
MUTED_BLUE_CMAP = LinearSegmentedColormap.from_list(
    "paper_muted_blue",
    ["#C9D2DA", "#B5C2CE", "#96AABC", "#728CA5", "#4B6C8A", "#254A68"],
)
FIG_TITLE_SIZE = 13.0
FIG_LABEL_SIZE = 10.5
FIG_TICK_SIZE = 9.0
FIG_LEGEND_SIZE = 9.0
GRID_COLOR = (0, 0, 0, 0.09)
REFERENCE_LINE_COLOR = (0, 0, 0, 0.38)
GUIDE_LINE_COLOR = (0, 0, 0, 0.22)
OLD_GRID_COLORS = {"#E6E6E6", "#EFEFEF", "#D9D9D9", "#BDBDBD"}
EFFECT_LABELS = {
    "filmstock_cinestill_minus_portra": "CineStill - Portra",
    "lighting_warm_minus_cool": "Warm practical - cool ambient",
    "scan_pushed_minus_clean": "Pushed scan - clean scan",
}
EFFECT_SHORT = {
    "filmstock_cinestill_minus_portra": "Filmstock",
    "lighting_warm_minus_cool": "Lighting",
    "scan_pushed_minus_clean": "Scan",
}
METRIC_LABELS = {
    "distance_all_selected_features": "All features",
    "distance_color_features": "Color",
    "distance_contrast_texture_features": "Contrast and texture",
    "distance_structure_features": "Structure",
}
FEATURE_LABELS = {
    "local_lab_cov_l_b_abs_mean": "Local L-b covariance",
    "local_lab_cov_l_b_p90_abs": "Local L-b covariance p90",
    "split_tone_distance_lab_ab": "Split-tone Lab distance",
    "highlight_minus_shadow_lab_b": "Highlight-shadow Lab b",
    "highlight_minus_shadow_warmth": "Highlight-shadow warmth",
    "lab_mean_b": "Lab yellow-blue",
    "halation_index_ring_minus_background": "Halation index",
    "halation_ring_red_excess_mean": "Red ring excess",
    "warmth_rgb_mean_r_minus_b": "Warmth R-B",
    "saturation_mean": "Saturation",
    "fft_high_freq_log_power_mean": "High-frequency log power",
    "fft_mid_freq_log_power_mean": "Mid-frequency log power",
    "edge_density_canny": "Edge density",
    "gradient_mag_mean": "Gradient magnitude",
    "luma_p05": "Luma p05",
    "palette_top2_lab_distance": "Top-two palette distance",
    "palette_entropy_6": "Palette entropy",
    "texture_highpass_mad": "High-pass texture MAD",
    "texture_highpass_std": "High-pass texture SD",
    "laplacian_variance": "Laplacian variance",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def f(value: str | float) -> float:
    return float(value)


def mean_ci(vals: list[float]) -> tuple[float, float]:
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size < 2:
        return float(arr.mean()), 0.0
    mean = float(arr.mean())
    se = float(arr.std(ddof=1) / math.sqrt(arr.size))
    return mean, float(t_dist.ppf(0.975, arr.size - 1) * se)


def clean_label(name: str, max_len: int = 34) -> str:
    label = FEATURE_LABELS.get(name, name.replace("_", " "))
    if len(label) > max_len:
        return label[: max_len - 1] + "..."
    return label


def save(fig, filename: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / filename
    style_figure(fig)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def color_to_hex(color) -> str | None:
    if isinstance(color, str) and color.startswith("#"):
        return color.upper()
    return None


def style_figure(fig) -> None:
    for ax in fig.axes:
        ax.set_axisbelow(True)
        ax.title.set_fontsize(max(ax.title.get_fontsize(), FIG_TITLE_SIZE))
        ax.xaxis.label.set_fontsize(max(ax.xaxis.label.get_fontsize(), FIG_LABEL_SIZE))
        ax.yaxis.label.set_fontsize(max(ax.yaxis.label.get_fontsize(), FIG_LABEL_SIZE))
        nonempty_tick_labels = [label for label in [*ax.get_xticklabels(), *ax.get_yticklabels()] if label.get_text()]
        tick_floor = 7.8 if len(nonempty_tick_labels) > 18 else FIG_TICK_SIZE
        for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
            label.set_fontsize(max(label.get_fontsize(), tick_floor))
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(max(text.get_fontsize(), FIG_LEGEND_SIZE))
            legend_title = legend.get_title()
            if legend_title is not None:
                legend_title.set_fontsize(max(legend_title.get_fontsize(), FIG_LEGEND_SIZE))
        for gridline in [*ax.get_xgridlines(), *ax.get_ygridlines()]:
            gridline.set_color("black")
            gridline.set_alpha(0.09)
            gridline.set_linewidth(0.75)
        for line in ax.lines:
            color = color_to_hex(line.get_color())
            if color in OLD_GRID_COLORS:
                line.set_color("black")
                line.set_alpha(0.22)
            elif color in {"#333", "#333333"}:
                line.set_color("black")
                line.set_alpha(0.38)


def add_selection(
    rows: list[dict[str, object]],
    filename: str,
    title: str,
    status: str,
    role: str,
    rationale: str,
) -> None:
    rows.append(
        {
            "filename": filename,
            "title": title,
            "status": status,
            "role": role,
            "rationale": rationale,
        }
    )


TERM_LABELS = {
    "light": "Lighting",
    "model by light": "Model × lighting",
    "model": "Model",
    "scan": "Development-scan",
    "film": "Filmstock",
    "film by light": "Filmstock × lighting",
    "model by scan": "Model × scan",
    "model by film": "Model × filmstock",
    "light by scan": "Lighting × scan",
    "film by light by scan": "Filmstock × lighting × scan",
    "film by scan": "Filmstock × scan",
}


def pc_axis_label(component: str, pct: float) -> str:
    if component == "PC1":
        return f"PC1 score ({pct:.1f}%): high = blue/green brightness; low = warm highlights"
    if component == "PC2":
        return f"PC2 score ({pct:.1f}%): high = palette warmth/chroma; low = teal/blue shadows"
    return f"{component} score ({pct:.1f}% variance)"


def covariance_data_ellipse(points: np.ndarray, coverage: float = 0.80) -> tuple[np.ndarray, np.ndarray] | None:
    if points.shape[0] < 3:
        return None
    theta = np.linspace(0, 2 * np.pi, 180)
    circle = np.column_stack([np.cos(theta), np.sin(theta)])
    center = points.mean(axis=0)
    cov = np.cov(points, rowvar=False, ddof=1)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 0)
    transform = vecs @ np.diag(np.sqrt(vals * chi2.ppf(coverage, 2)))
    ellipse = circle @ transform.T + center
    return ellipse[:, 0], ellipse[:, 1]


def pca_by_group(scores: list[dict[str, str]], explained: list[dict[str, str]], group: str, filename: str, title: str) -> None:
    pc1 = f(explained[0]["explained_variance_ratio"]) * 100
    pc2 = f(explained[1]["explained_variance_ratio"]) * 100
    if group == "model_key":
        colors = MODEL_COLORS
        labels = MODEL_LABELS
    elif group == "light_key":
        colors = {"cool_ambient": "#4E79A7", "warm_practical": "#C43C39"}
        labels = {"cool_ambient": "Cool ambient", "warm_practical": "Warm practical"}
    elif group == "film_key":
        colors = {"portra400": "#4E79A7", "cinestill800t": "#D37230"}
        labels = {"portra400": "Portra 400", "cinestill800t": "CineStill 800T"}
    else:
        colors = {"clean_scan": "#59A14F", "pushed_scan": "#B07AA1"}
        labels = {"clean_scan": "Clean scan", "pushed_scan": "Pushed scan"}
    fig, ax = plt.subplots(figsize=(8.8, 6.25), dpi=180)
    for key in sorted(colors):
        pts = np.asarray([[f(r["PC1"]), f(r["PC2"])] for r in scores if r[group] == key])
        ax.scatter(pts[:, 0], pts[:, 1], s=28, alpha=0.62, color=colors[key], edgecolor="white", linewidth=0.3, label=labels[key])
        ell = covariance_data_ellipse(pts)
        if ell:
            ax.fill(*ell, color=colors[key], alpha=0.075)
            ax.plot(*ell, color=colors[key], linewidth=1.9)
    ax.axhline(0, color=REFERENCE_LINE_COLOR, linewidth=0.9)
    ax.axvline(0, color=REFERENCE_LINE_COLOR, linewidth=0.9)
    ax.set_xlabel(pc_axis_label("PC1", pc1), fontsize=8.8)
    ax.set_ylabel(pc_axis_label("PC2", pc2), fontsize=8.8)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(color=GRID_COLOR)
    save(fig, filename)


def pca_by_run(scores: list[dict[str, str]], explained: list[dict[str, str]]) -> None:
    pc1 = f(explained[0]["explained_variance_ratio"]) * 100
    pc2 = f(explained[1]["explained_variance_ratio"]) * 100
    runs = sorted({r["run_block"] for r in scores}, key=int)
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.8), dpi=180, sharex=True, sharey=True)
    for ax, run in zip(axes, runs):
        for model in ["chatgpt_image_2", "xai_grok_imagine"]:
            pts = np.asarray([[f(r["PC1"]), f(r["PC2"])] for r in scores if r["run_block"] == run and r["model_key"] == model])
            ax.scatter(pts[:, 0], pts[:, 1], s=24, alpha=0.7, color=MODEL_COLORS[model], edgecolor="white", linewidth=0.2, label=MODEL_LABELS[model])
        ax.set_title(f"Run block {run}")
        ax.axhline(0, color=REFERENCE_LINE_COLOR, linewidth=0.8)
        ax.axvline(0, color=REFERENCE_LINE_COLOR, linewidth=0.8)
        ax.grid(color=GRID_COLOR)
    axes[0].set_ylabel(pc_axis_label("PC2", pc2), fontsize=8)
    fig.supxlabel(pc_axis_label("PC1", pc1), fontsize=8, y=0.02)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("PCA Stability Across the Three Generation Runs", y=0.98)
    fig.subplots_adjust(bottom=0.18, top=0.82, wspace=0.08)
    save(fig, "v2_04_pca_by_run_block.png")


def plot_permanova(permanova: list[dict[str, str]]) -> None:
    rows = sorted(permanova, key=lambda r: f(r["partial_r2"]), reverse=True)
    fig, ax = plt.subplots(figsize=(8.3, 5.1), dpi=180)
    y = np.arange(len(rows))
    vals = [f(r["partial_r2"]) for r in rows]
    ax.barh(y, vals, color="#4E79A7", alpha=0.88)
    ax.set_yticks(y)
    ax.set_yticklabels([TERM_LABELS.get(r["term_label"], r["term_label"]) for r in rows], fontsize=8)
    ax.invert_yaxis()
    for i, r in enumerate(rows):
        ax.text(vals[i] + max(vals) * 0.02, i, f"p={f(r['permutation_p_value']):.3f}", va="center", fontsize=9)
    ax.set_xlabel("Partial R²: share of PC1-PC10 variation explained")
    ax.set_title("Blocked PERMANOVA on Standardized Image-Feature Scores")
    ax.grid(axis="x", color=GRID_COLOR)
    save(fig, "v2_05_blocked_permanova_partial_r2.png")


def plot_manova_vs_permanova(manova: list[dict[str, str]], permanova: list[dict[str, str]]) -> None:
    terms = ["light", "model by light", "model", "scan", "film", "film by light"]
    manova_lookup = {r["term_label"]: f(r["pillai_trace"]) for r in manova}
    per_lookup = {r["term_label"]: f(r["partial_r2"]) for r in permanova}
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), dpi=180)
    y = np.arange(len(terms))
    axes[0].barh(y, [manova_lookup.get(t, 0) for t in terms], color="#C43C39", alpha=0.86)
    axes[0].set_title("Blocked MANOVA\nPillai trace on PC1-PC10")
    axes[1].barh(y, [per_lookup.get(t, 0) for t in terms], color="#4E79A7", alpha=0.86)
    axes[1].set_title("Blocked PERMANOVA\nPartial R² on PC1-PC10")
    for ax in axes:
        ax.set_yticks(y)
        ax.set_yticklabels([TERM_LABELS.get(t, t) for t in terms], fontsize=8)
        ax.invert_yaxis()
        ax.grid(axis="x", color=GRID_COLOR)
    save(fig, "v2_06_manova_permanova_comparison.png")


def plot_distance_bars(distance_summary: list[dict[str, str]]) -> None:
    metrics = list(METRIC_LABELS)
    effects = list(EFFECT_LABELS)
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.1), dpi=180, sharex=True)
    axes = axes.reshape(-1)
    for ax, metric in zip(axes, metrics):
        x = np.arange(len(effects))
        width = 0.35
        for j, model in enumerate(["chatgpt_image_2", "xai_grok_imagine"]):
            means, errs = [], []
            for effect in effects:
                r = next(row for row in distance_summary if row["effect_type"] == effect and row["model_key"] == model and row["distance_metric"] == metric)
                means.append(f(r["mean_distance"]))
                errs.append(t_dist.ppf(0.975, int(r["n_pairs"]) - 1) * f(r["se_distance"]))
            ax.bar(x + (j - 0.5) * width, means, width, color=MODEL_COLORS[model], alpha=0.86, label=MODEL_LABELS[model])
            ax.errorbar(x + (j - 0.5) * width, means, yerr=errs, fmt="none", ecolor="#222", linewidth=0.8, capsize=3)
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xticks(x)
        ax.set_xticklabels([EFFECT_SHORT[e] for e in effects])
        ax.set_ylabel("Mean Euclidean distance across z-scored features")
        ax.grid(axis="y", color=GRID_COLOR)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Prompt Responsiveness by Model, Factor, and Feature Family", y=1.02)
    save(fig, "v2_07_responsiveness_distance_bars_ci.png")


def plot_distance_box(distance_rows: list[dict[str, str]]) -> None:
    effects = list(EFFECT_LABELS)
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.5), dpi=180, sharey=True)
    rng = np.random.default_rng(216)
    for ax, effect in zip(axes, effects):
        for j, model in enumerate(["chatgpt_image_2", "xai_grok_imagine"]):
            vals = np.asarray([f(r["distance_all_selected_features"]) for r in distance_rows if r["effect_type"] == effect and r["model_key"] == model])
            x = np.full(vals.shape, j + 1, dtype=float) + rng.normal(0, 0.045, size=vals.shape)
            ax.boxplot(vals, positions=[j + 1], widths=0.42, patch_artist=True, showfliers=False, boxprops={"facecolor": MODEL_COLORS[model], "alpha": 0.35}, medianprops={"color": "#222"})
            ax.scatter(x, vals, s=13, color=MODEL_COLORS[model], alpha=0.55, edgecolor="none")
        ax.set_title(EFFECT_SHORT[effect])
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["ChatGPT", "Grok"])
        ax.grid(axis="y", color=GRID_COLOR)
    axes[0].set_ylabel("All-feature Euclidean distance across z-scored features")
    fig.suptitle("Distribution of Matched Prompt-Response Distances", y=1.03)
    save(fig, "v2_08_distance_box_strip_by_effect.png")


def plot_paired_lines(distance_rows: list[dict[str, str]]) -> None:
    effect = "lighting_warm_minus_cool"
    rows = [r for r in distance_rows if r["effect_type"] == effect]
    keys = sorted({(r["run_id"], r["scene_key"], r.get("film_key", ""), r.get("scan_key", ""), r["replicate"]) for r in rows})
    idx = {(r["model_key"], r["run_id"], r["scene_key"], r.get("film_key", ""), r.get("scan_key", ""), r["replicate"]): f(r["distance_all_selected_features"]) for r in rows}
    fig, ax = plt.subplots(figsize=(6.6, 5.4), dpi=180)
    for key in keys:
        gpt = idx.get(("chatgpt_image_2", *key))
        grok = idx.get(("xai_grok_imagine", *key))
        if gpt is None or grok is None:
            continue
        color = "#888888" if grok >= gpt else "#C43C39"
        ax.plot([0, 1], [gpt, grok], color=color, alpha=0.45, linewidth=1)
        ax.scatter([0, 1], [gpt, grok], color=[MODEL_COLORS["chatgpt_image_2"], MODEL_COLORS["xai_grok_imagine"]], s=18, zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["ChatGPT Image 2", "Grok Imagine"])
    ax.set_ylabel("Lighting-response distance across z-scored features")
    ax.set_title("Matched Pairs for the Lighting Factor")
    ax.grid(axis="y", color=GRID_COLOR)
    save(fig, "v2_09_paired_lines_lighting_distance.png")


def plot_bootstrap_forest(boot_rows: list[dict[str, str]]) -> None:
    rows = [r for r in boot_rows if r["distance_metric"] == "distance_all_selected_features"]
    fig, ax = plt.subplots(figsize=(7.8, 4.5), dpi=180)
    y = np.arange(len(rows))
    means = [f(r["mean_difference_xai_minus_chatgpt"]) for r in rows]
    lo = [f(r["hierarchical_boot_ci95_low"]) for r in rows]
    hi = [f(r["hierarchical_boot_ci95_high"]) for r in rows]
    for i, r in enumerate(rows):
        ax.plot([lo[i], hi[i]], [i, i], color="#222", linewidth=2)
        ax.scatter([means[i]], [i], color="#C43C39", s=42, zorder=3)
        ax.text(hi[i] + 0.05, i, f"p={f(r['hierarchical_boot_p_two_sided']):.3f}", va="center", fontsize=9)
    ax.axvline(0, color=REFERENCE_LINE_COLOR, linewidth=1, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels([EFFECT_SHORT[r["effect_type"]] for r in rows])
    ax.set_xlabel("Grok minus ChatGPT all-feature distance across z-scored features")
    ax.set_title("Hierarchical Bootstrap Model Comparisons")
    ax.grid(axis="x", color=GRID_COLOR)
    save(fig, "v2_10_hierarchical_bootstrap_forest.png")


def plot_targeted_heatmap(target_rows: list[dict[str, str]]) -> None:
    keep_features = [
        ("lighting_warm_minus_cool", "local_lab_cov_l_b_abs_mean"),
        ("lighting_warm_minus_cool", "split_tone_distance_lab_ab"),
        ("lighting_warm_minus_cool", "highlight_minus_shadow_lab_b"),
        ("lighting_warm_minus_cool", "warmth_rgb_mean_r_minus_b"),
        ("lighting_warm_minus_cool", "lab_mean_b"),
        ("lighting_warm_minus_cool", "halation_index_ring_minus_background"),
        ("filmstock_cinestill_minus_portra", "fft_high_freq_power_share"),
        ("filmstock_cinestill_minus_portra", "edge_density_canny"),
        ("filmstock_cinestill_minus_portra", "halation_index_ring_minus_background"),
        ("scan_pushed_minus_clean", "texture_highpass_mad"),
        ("scan_pushed_minus_clean", "fft_high_mid_power_ratio"),
        ("scan_pushed_minus_clean", "shadow_compression_ratio"),
    ]
    vals, labels = [], []
    for effect, feature in keep_features:
        found = [r for r in target_rows if r["effect_type"] == effect and r["feature"] == feature]
        if not found:
            continue
        r = found[0]
        vals.append([f(r["mean_chatgpt_standardized"]), f(r["mean_xai_standardized"])])
        labels.append(f"{EFFECT_SHORT[effect]}: {clean_label(feature, 28)}")
    mat = np.asarray(vals)
    vmax = max(0.2, float(np.nanmax(np.abs(mat))))
    fig, ax = plt.subplots(figsize=(7.4, 7.0), dpi=180)
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["ChatGPT", "Grok"])
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=7.2)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8.5)
    ax.set_title("Targeted Photographic Feature Effects")
    fig.colorbar(im, ax=ax, shrink=0.78, label="High-minus-low feature change (SDs)")
    save(fig, "v2_11_targeted_feature_heatmap_curated.png")


def plot_screening_volcano(screen_rows: list[dict[str, str]]) -> None:
    colors = {
        "filmstock_cinestill_minus_portra": "#4E79A7",
        "lighting_warm_minus_cool": "#C43C39",
        "scan_pushed_minus_clean": "#59A14F",
    }
    fig, ax = plt.subplots(figsize=(8.6, 5.5), dpi=180)
    for effect, color in colors.items():
        sub = [r for r in screen_rows if r["effect_type"] == effect]
        x = [f(r["abs_standardized_model_gap"]) for r in sub]
        y = [-math.log10(max(f(r["bh_fdr_p_all_features"]), 1e-12)) for r in sub]
        ax.scatter(x, y, s=18, color=color, alpha=0.6, label=EFFECT_SHORT[effect], edgecolor="none")
    ax.axhline(-math.log10(0.05), color=REFERENCE_LINE_COLOR, linewidth=0.9, linestyle="--")
    ax.set_xlabel("Absolute model gap in SD-scaled feature effects")
    ax.set_ylabel("-log10(FDR-adjusted p)")
    ax.set_title("Exploratory Screening Across All 195 Image Features")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(color=GRID_COLOR)
    save(fig, "v2_12_all_feature_screening_volcano.png")


def plot_top_screening_lollipop(screen_rows: list[dict[str, str]], effect: str, filename: str) -> None:
    rows = [r for r in screen_rows if r["effect_type"] == effect][:18][::-1]
    fig, ax = plt.subplots(figsize=(8.2, 6.0), dpi=180)
    y = np.arange(len(rows))
    vals = [f(r["abs_standardized_model_gap"]) for r in rows]
    color = "#C43C39" if effect == "lighting_warm_minus_cool" else "#4E79A7"
    ax.barh(y, vals, color=color, alpha=0.86, edgecolor=(0, 0, 0, 0.28), linewidth=0.45)
    ax.set_yticks(y)
    ax.set_yticklabels([clean_label(r["feature"], 34) for r in rows], fontsize=7.2)
    ax.set_xlim(0, max(vals) * 1.08)
    ax.set_xlabel("Absolute model gap in SD-scaled feature effects")
    ax.set_title(f"Top Model-Differentiating Features: {EFFECT_SHORT[effect]}")
    ax.grid(axis="x", color=GRID_COLOR)
    save(fig, filename)


def plot_pls_vip(vip_rows: list[dict[str, str]]) -> None:
    targets = ["model_key", "light_key", "film_key", "scan_key"]
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.2), dpi=180)
    axes = axes.reshape(-1)
    for ax, target in zip(axes, targets):
        rows = [r for r in vip_rows if r["target"] == target][:10][::-1]
        y = np.arange(len(rows))
        vals = [f(r["vip"]) for r in rows]
        ax.barh(y, vals, color="#4E79A7", alpha=0.86)
        ax.axvline(1, color=REFERENCE_LINE_COLOR, linestyle="--", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([clean_label(r["feature"], 27) for r in rows], fontsize=6.8)
        ax.set_title(rows[-1]["target_label"] if rows else target)
        ax.set_xlabel("VIP score (unitless predictor importance)")
        ax.grid(axis="x", color=GRID_COLOR)
    fig.suptitle("PLS-DA VIP Features by Prediction Target", y=1.02)
    save(fig, "v2_15_pls_vip_top_features.png")


def plot_regularized_coefficients(coef_rows: list[dict[str, str]]) -> None:
    selected = [("model_key", "Ridge logistic"), ("light_key", "Ridge logistic")]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.4), dpi=180)
    for ax, (target, method) in zip(axes, selected):
        rows = [r for r in coef_rows if r["target"] == target and r["method"] == method][:12][::-1]
        vals = [f(r["coefficient"]) for r in rows]
        colors = ["#C43C39" if v > 0 else "#4E79A7" for v in vals]
        ax.barh(np.arange(len(rows)), vals, color=colors, alpha=0.86)
        ax.axvline(0, color=REFERENCE_LINE_COLOR, linewidth=0.9)
        ax.set_yticks(np.arange(len(rows)))
        ax.set_yticklabels([clean_label(r["feature"], 28) for r in rows], fontsize=6.8)
        ax.set_title(f"{rows[-1]['target_label']} ({method})" if rows else target)
        ax.grid(axis="x", color=GRID_COLOR)
    fig.suptitle("Regularized Classification Coefficients", y=1.03)
    save(fig, "v2_16_regularized_coefficients_model_lighting.png")


def plot_variance_components(feature_var: list[dict[str, str]]) -> None:
    components = ["condition_share", "run_block_share", "condition_by_run_share", "within_run_replicate_share"]
    labels = ["Prompt condition", "Run block", "Condition × run", "Within-run replicate"]
    colors = ["#4E79A7", "#F28E2B", "#59A14F", "#C43C39"]
    models = ["chatgpt_image_2", "xai_grok_imagine"]
    means = {
        model: [np.mean([f(r[comp]) for r in feature_var if r["model_key"] == model]) for comp in components]
        for model in models
    }
    fig, (ax, ax_leg, ax_delta) = plt.subplots(
        3,
        1,
        figsize=(9.0, 6.65),
        dpi=180,
        gridspec_kw={"height_ratios": [1.35, 0.22, 1.1], "hspace": 0.42},
    )

    y = np.arange(len(models))
    left = np.zeros(len(models))
    for idx, (comp, label, color) in enumerate(zip(components, labels, colors)):
        vals = np.asarray([means[model][idx] for model in models])
        ax.barh(y, vals, left=left, color=color, alpha=0.88, label=label, height=0.48)
        for row_idx, value in enumerate(vals):
            if value >= 0.055:
                ax.text(left[row_idx] + value / 2, row_idx, f"{value*100:.1f}%", ha="center", va="center", fontsize=8.4, color="white")
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_LABELS[m] for m in models])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Mean share of total standardized feature variance across 195 features")
    ax.set_title("Feature Variance Is Mostly Prompt Condition, Not Run Block")
    ax.grid(axis="x", color=GRID_COLOR)
    handles, legend_labels = ax.get_legend_handles_labels()
    ax_leg.axis("off")
    ax_leg.legend(handles, legend_labels, frameon=False, fontsize=7.6, ncol=4, loc="center")

    deltas = [(means["xai_grok_imagine"][idx] - means["chatgpt_image_2"][idx]) * 100 for idx in range(len(components))]
    yy = np.arange(len(components))
    max_abs_delta = max(abs(value) for value in deltas)
    x_pad = max(0.45, max_abs_delta * 0.18)
    ax_delta.axvline(0, color=REFERENCE_LINE_COLOR, linewidth=0.9)
    ax_delta.barh(yy, deltas, color=colors, alpha=0.86, height=0.52)
    for i, val in enumerate(deltas):
        if abs(val) >= 0.8:
            x_pos = val - 0.08 if val > 0 else val + 0.08
            ha = "right" if val > 0 else "left"
            text_color = "white"
        else:
            x_pos = val + (0.12 if val >= 0 else -0.12)
            ha = "left" if val >= 0 else "right"
            text_color = "black"
        ax_delta.text(x_pos, i, f"{val:+.1f} pp", va="center", ha=ha, fontsize=8.4, color=text_color)
    ax_delta.set_yticks(yy)
    ax_delta.set_yticklabels(labels, fontsize=7.4)
    ax_delta.set_xlim(min(deltas) - x_pad, max(deltas) + x_pad)
    ax_delta.set_xlabel("Grok minus ChatGPT share (percentage points)")
    ax_delta.set_title("Grok minus ChatGPT Variance-Share Difference by Component", fontsize=10)
    ax_delta.grid(axis="x", color=GRID_COLOR)
    path = FIG_DIR / "v2_17_feature_variance_components.png"
    style_figure(fig)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_feature_correlation_heatmap(image_rows: list[dict[str, str]]) -> None:
    features = [
        "warmth_rgb_mean_r_minus_b",
        "lab_mean_b",
        "saturation_mean",
        "highlight_saturation",
        "split_tone_distance_lab_ab",
        "highlight_minus_shadow_warmth",
        "highlight_minus_shadow_lab_b",
        "halation_ring_red_excess_mean",
        "hue_red_orange_share",
        "hue_teal_cyan_share",
        "fft_high_freq_log_power_mean",
        "edge_density_canny",
        "gradient_mag_mean",
        "palette_entropy_6",
        "palette_top2_lab_distance",
        "local_lab_cov_l_b_abs_mean",
    ]
    labels = [clean_label(name, 24) for name in features]
    data = np.asarray([[f(row[name]) for name in features] for row in image_rows], dtype=np.float64)
    corr = np.corrcoef(data, rowvar=False)
    fig, ax = plt.subplots(figsize=(8.6, 7.6), dpi=180)
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(np.arange(len(features)))
    ax.set_yticks(np.arange(len(features)))
    ax.set_xticklabels(labels, fontsize=6.4)
    ax.set_yticklabels(labels, fontsize=6.4)
    ax.xaxis.tick_top()
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
    for i in range(len(features)):
        for j in range(len(features)):
            color = "white" if abs(corr[i, j]) > 0.55 else "#222222"
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=4.8, color=color)
    ax.set_title("Correlation Matrix for Selected Image Features", fontsize=16, fontweight="semibold", pad=24)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.035)
    cbar.set_label("Pearson correlation r", fontsize=8.5)
    save(fig, "v2_24_existing_feature_correlation_heatmap.png")


def plot_seed_variation(seed_summary: list[dict[str, str]]) -> None:
    rows = [r for r in seed_summary if r["variation_metric"] == "mean_pairwise_seed_distance"]
    fig, ax = plt.subplots(figsize=(6.3, 4.5), dpi=180)
    x = np.arange(len(rows))
    means = [f(r["mean"]) for r in rows]
    err = [f(r["mean"]) - f(r["ci95_low"]) for r in rows]
    ax.bar(x, means, color=[MODEL_COLORS[r["model_key"]] for r in rows], alpha=0.86)
    ax.errorbar(x, means, yerr=err, fmt="none", ecolor="#222", capsize=4, linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[r["model_key"]] for r in rows])
    ax.set_ylabel("Mean exact-prompt distance across z-scored features")
    ax.set_title("Seed-to-Seed Variation Within Exact Prompt Conditions")
    ax.grid(axis="y", color=GRID_COLOR)
    save(fig, "v2_18_seed_to_seed_variation.png")


def plot_interaction_heatmap(interactions: list[dict[str, str]]) -> None:
    interaction_order = [
        "filmstock_by_lighting",
        "filmstock_by_scan",
        "lighting_by_scan",
        "filmstock_by_lighting_by_scan",
    ]
    labels = {
        "filmstock_by_lighting": "Filmstock × lighting",
        "filmstock_by_scan": "Filmstock × scan",
        "lighting_by_scan": "Lighting × scan",
        "filmstock_by_lighting_by_scan": "Filmstock × lighting × scan",
    }
    models = ["chatgpt_image_2", "xai_grok_imagine"]
    lookup = {(r["interaction_type"], r["model_key"]): f(r["mean_abs_standardized_interaction"]) for r in interactions}
    mat = np.asarray([[lookup[(interaction, model)] for model in models] for interaction in interaction_order])
    fig, ax = plt.subplots(figsize=(6.6, 4.3), dpi=180)
    im = ax.imshow(mat, cmap=MUTED_BLUE_CMAP, aspect="auto")
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels([MODEL_LABELS[m] for m in models])
    ax.set_yticks(np.arange(len(interaction_order)))
    ax.set_yticklabels([labels[i] for i in interaction_order], fontsize=7.5)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=9, color="black" if mat[i, j] < mat.max() * 0.84 else "white")
    ax.set_xticks(np.arange(-0.5, len(models), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(interaction_order), 1), minor=True)
    ax.grid(which="minor", color=(1, 1, 1, 0.82), linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title("Mean Absolute Factor Interaction Contrasts")
    cbar = fig.colorbar(im, ax=ax, fraction=0.052, pad=0.035)
    cbar.set_label("Mean abs. contrast (SDs)", fontsize=9)
    save(fig, "v2_23_existing_interaction_heatmap.png")


def plot_nonparametric_ordering(distance_summary: list[dict[str, str]]) -> None:
    rows = [r for r in distance_summary if r["distance_metric"] == "distance_all_selected_features"]
    effects = list(EFFECT_LABELS)
    models = ["chatgpt_image_2", "xai_grok_imagine"]
    fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=180)
    x = np.arange(len(effects))
    width = 0.36
    for j, model in enumerate(models):
        means, errs = [], []
        for effect in effects:
            r = next(row for row in rows if row["model_key"] == model and row["effect_type"] == effect)
            means.append(f(r["mean_distance"]))
            errs.append(t_dist.ppf(0.975, int(r["n_pairs"]) - 1) * f(r["se_distance"]))
        ax.bar(x + (j - 0.5) * width, means, width, color=MODEL_COLORS[model], alpha=0.86, label=MODEL_LABELS[model])
        ax.errorbar(x + (j - 0.5) * width, means, yerr=errs, fmt="none", ecolor="#222", linewidth=0.8, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([EFFECT_SHORT[e] for e in effects])
    ax.set_ylabel("Mean all-feature Euclidean distance across z-scored features")
    ax.set_title("Distance Ordering Used in Kruskal-Wallis and Friedman Tests")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", color=GRID_COLOR)
    save(fig, "v2_25_existing_nonparametric_ordering.png")


def plot_run_variation(run_rows: list[dict[str, str]]) -> None:
    effects = list(EFFECT_LABELS)
    models = ["chatgpt_image_2", "xai_grok_imagine"]
    fig, ax = plt.subplots(figsize=(8.6, 4.7), dpi=180)
    x = np.arange(len(effects))
    width = 0.35
    for j, model in enumerate(models):
        vals = []
        for effect in effects:
            r = next(row for row in run_rows if row["model_key"] == model and row["effect_type"] == effect and row["distance_metric"] == "distance_all_selected_features")
            vals.append(f(r["between_run_sd_of_means"]))
        ax.bar(x + (j - 0.5) * width, vals, width, color=MODEL_COLORS[model], alpha=0.86, label=MODEL_LABELS[model])
    ax.set_xticks(x)
    ax.set_xticklabels([EFFECT_SHORT[e] for e in effects])
    ax.set_ylabel("Between-run SD of mean Euclidean distance")
    ax.set_title("Run-Block Variation in Responsiveness")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", color=GRID_COLOR)
    save(fig, "v2_19_run_block_variation.png")


def plot_reliable_features(rel_rows: list[dict[str, str]]) -> None:
    rows = rel_rows[:20][::-1]
    fig, ax = plt.subplots(figsize=(8.8, 6.9), dpi=180)
    vals = [f(r["standardized_mean_run_effect"]) for r in rows]
    colors = ["#C43C39" if v > 0 else "#4E79A7" for v in vals]
    labels = [f"{EFFECT_SHORT[r['effect_type']]} | {clean_label(r['feature'], 28)}" for r in rows]
    ax.barh(np.arange(len(rows)), vals, color=colors, alpha=0.86)
    ax.axvline(0, color=REFERENCE_LINE_COLOR, linewidth=0.9)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(labels, fontsize=6.7)
    ax.set_xlabel("Mean high-minus-low feature effect across runs (SDs)")
    ax.set_title("Most Reliable Feature Effects Across All Three Runs")
    ax.grid(axis="x", color=GRID_COLOR)
    save(fig, "v2_20_cross_run_reliable_features.png")


def plot_outliers(outliers: list[dict[str, str]]) -> None:
    rows = outliers[:18][::-1]
    fig, ax = plt.subplots(figsize=(8.7, 6.3), dpi=180)
    vals = [f(r["robust_mahalanobis_sq_pc10"]) for r in rows]
    labels = [f"{'Grok' if r['model_key']=='xai_grok_imagine' else 'ChatGPT'} | {r['scene_key']} | {r['film_key']}" for r in rows]
    ax.barh(np.arange(len(rows)), vals, color="#B07AA1", alpha=0.86)
    ax.axvline(f(rows[0]["chi_square_975_cutoff"]), color=REFERENCE_LINE_COLOR, linestyle="--", linewidth=0.9)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(labels, fontsize=6.8)
    ax.set_xlabel("Robust Mahalanobis distance squared in PC1-PC10 space")
    ax.set_title("Robust Multivariate Outlier Diagnostics")
    ax.grid(axis="x", color=GRID_COLOR)
    save(fig, "v2_21_robust_outlier_diagnostics.png")


def contact_sheet(image_rows: list[dict[str, str]], scene: str, filename: str) -> None:
    rows = [
        r for r in image_rows
        if r["run_block"] == "1"
        and r["scene_key"] == scene
        and r["scan_key"] == "clean_scan"
        and r["replicate"] == "1"
    ]
    cols = [
        ("portra400", "cool_ambient"),
        ("portra400", "warm_practical"),
        ("cinestill800t", "cool_ambient"),
        ("cinestill800t", "warm_practical"),
    ]
    models = ["chatgpt_image_2", "xai_grok_imagine"]
    by_key = {(r["model_key"], r["film_key"], r["light_key"]): r for r in rows}
    if scene == "apartment":
        alternate_rows = [
            r for r in image_rows
            if r["scene_key"] == scene
            and r["model_key"] == "xai_grok_imagine"
            and r["film_key"] == "cinestill800t"
            and r["light_key"] == "cool_ambient"
            and r["scan_key"] == "clean_scan"
            and r["run_block"] == "3"
            and r["replicate"] == "2"
        ]
        if alternate_rows:
            by_key[("xai_grok_imagine", "cinestill800t", "cool_ambient")] = alternate_rows[0]
    thumb = 230
    top_h = 54
    left_w = 165
    canvas = Image.new("RGB", (left_w + len(cols) * thumb, top_h + len(models) * (thumb + 38)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for j, (film, light) in enumerate(cols):
        label = f"{film}\n{light}"
        draw.multiline_text((left_w + j * thumb + 8, 10), label, fill=(25, 25, 25), font=font, spacing=4)
    for i, model in enumerate(models):
        y = top_h + i * (thumb + 38)
        draw.text((8, y + thumb // 2 - 8), MODEL_LABELS[model], fill=(25, 25, 25), font=font)
        for j, (film, light) in enumerate(cols):
            r = by_key.get((model, film, light))
            if not r:
                continue
            im = Image.open(r["image_path"]).convert("RGB")
            im.thumbnail((thumb, thumb))
            canvas.paste(im, (left_w + j * thumb, y))
    canvas.save(FIG_DIR / filename)


def copy_existing_candidates(selection: list[dict[str, object]]) -> None:
    copies = []
    for src, dest, status, rationale in copies:
        shutil.copyfile(src, FIG_DIR / dest)
        add_selection(selection, dest, dest.replace("_", " ").replace(".png", ""), status, "Copied existing diagnostic", rationale)


def make_montages() -> None:
    paths = sorted(FIG_DIR.glob("v2_*.png"))
    thumb_w = 360
    label_h = 38
    cols = 3
    rows = math.ceil(len(paths) / cols)
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_w + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for i, path in enumerate(paths):
        im = Image.open(path).convert("RGB")
        im.thumbnail((thumb_w, thumb_w))
        x = (i % cols) * thumb_w
        y = (i // cols) * (thumb_w + label_h)
        sheet.paste(im, (x + (thumb_w - im.width) // 2, y))
        draw.text((x + 8, y + thumb_w + 6), path.name[:48], fill=(20, 20, 20), font=font)
    sheet.save(ASSET_DIR / "v2_candidate_figure_montage.png")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)
    for path in FIG_DIR.glob("*.png"):
        path.unlink()

    scores = read_csv(ANALYSIS_DIR / "pca_image_scores.csv")
    explained = read_csv(ANALYSIS_DIR / "pca_explained_variance.csv")
    image_rows = read_csv(ANALYSIS_DIR / "image_features.csv")
    distance_rows = read_csv(ANALYSIS_DIR / "responsiveness_distances.csv")
    distance_summary = read_csv(ANALYSIS_DIR / "responsiveness_distance_summary.csv")
    manova = read_csv(STATS_DIR / "manova_pillai_permutation.csv")
    targeted = read_csv(STATS_DIR / "targeted_feature_tests.csv")
    screening = read_csv(STATS_DIR / "all_feature_model_screening.csv")
    interactions = read_csv(STATS_DIR / "interaction_contrast_summary.csv")
    seed_summary = read_csv(STATS_DIR / "seed_variation_model_summary.csv")
    run_variation = read_csv(STATS_DIR / "run_block_variation_summary.csv")
    permanova = read_csv(ADV_DIR / "blocked_permanova_pc10.csv")
    boot = read_csv(ADV_DIR / "hierarchical_bootstrap_distance_tests.csv")
    vip = read_csv(ADV_DIR / "pls_vip_top_features.csv")
    regcoef = read_csv(ADV_DIR / "regularized_top_coefficients.csv")
    feature_var = read_csv(ADV_DIR / "image_feature_variance_components.csv")
    reliability = read_csv(ADV_DIR / "cross_run_feature_reliability.csv")
    outliers = read_csv(ADV_DIR / "robust_mahalanobis_outliers.csv")

    selection: list[dict[str, object]] = []

    pca_by_group(scores, explained, "model_key", "v2_01_pca_by_model.png", "PCA of 195 Standardized Image Features by Model")
    add_selection(selection, "v2_01_pca_by_model.png", "PCA by model", "Main", "PCA", "Readable overview of model overlap and separation.")
    pca_by_group(scores, explained, "light_key", "v2_02_pca_by_lighting.png", "PCA of 195 Standardized Image Features by Lighting")
    add_selection(selection, "v2_02_pca_by_lighting.png", "PCA by lighting", "Main", "PCA", "Strong visual evidence that lighting dominates the feature space.")
    pca_by_group(scores, explained, "film_key", "v2_03_pca_by_filmstock.png", "PCA of 195 Standardized Image Features by Filmstock")
    add_selection(selection, "v2_03_pca_by_filmstock.png", "PCA by filmstock", "Appendix", "PCA", "Useful secondary evidence; less visually separated than lighting.")
    pca_by_run(scores, explained)
    add_selection(selection, "v2_04_pca_by_run_block.png", "PCA by run block", "Appendix", "Run blocking", "Assesses run stability without overloading the main narrative.")
    plot_permanova(permanova)
    add_selection(selection, "v2_05_blocked_permanova_partial_r2.png", "Blocked PERMANOVA", "Main", "Advanced multivariate inference", "Cleanly ranks factor importance.")
    plot_manova_vs_permanova(manova, permanova)
    add_selection(selection, "v2_06_manova_permanova_comparison.png", "MANOVA vs PERMANOVA", "Appendix", "Robustness", "Good robustness comparison, but somewhat redundant with PERMANOVA.")
    plot_distance_bars(distance_summary)
    add_selection(selection, "v2_07_responsiveness_distance_bars_ci.png", "Responsiveness distances", "Main", "Primary response", "Central response-variable plot.")
    plot_distance_box(distance_rows)
    add_selection(selection, "v2_08_distance_box_strip_by_effect.png", "Distance distributions", "Appendix", "Primary response", "Shows spread and outliers; useful but busy.")
    plot_paired_lines(distance_rows)
    add_selection(selection, "v2_09_paired_lines_lighting_distance.png", "Paired lighting lines", "Main", "Model-by-lighting interaction", "Directly shows matched lighting-pair pattern.")
    plot_bootstrap_forest(boot)
    add_selection(selection, "v2_10_hierarchical_bootstrap_forest.png", "Hierarchical bootstrap forest", "Main", "Robust inference", "Concise and interpretable model-comparison result.")
    plot_targeted_heatmap(targeted)
    add_selection(selection, "v2_11_targeted_feature_heatmap_curated.png", "Curated targeted feature heatmap", "Main", "Feature interpretation", "Best compact feature-level evidence.")
    plot_screening_volcano(screening)
    add_selection(selection, "v2_12_all_feature_screening_volcano.png", "All-feature screening volcano", "Appendix", "Exploratory screening", "Important audit trail, but visually dense.")
    plot_top_screening_lollipop(screening, "lighting_warm_minus_cool", "v2_13_top_lighting_screening_features.png")
    add_selection(selection, "v2_13_top_lighting_screening_features.png", "Top lighting screening features", "Main", "Feature interpretation", "Strong local covariance and split-tone evidence.")
    plot_top_screening_lollipop(screening, "filmstock_cinestill_minus_portra", "v2_14_top_filmstock_screening_features.png")
    add_selection(selection, "v2_14_top_filmstock_screening_features.png", "Top filmstock screening features", "Appendix", "Feature interpretation", "Supports subtler filmstock result.")
    plot_pls_vip(vip)
    add_selection(selection, "v2_15_pls_vip_top_features.png", "PLS VIP features", "Main", "Predictive validation", "Shows which features classify model, lighting, film, and scan.")
    plot_regularized_coefficients(regcoef)
    add_selection(selection, "v2_16_regularized_coefficients_model_lighting.png", "Regularized coefficients", "Appendix", "Shrinkage models", "Useful but technical; better as appendix.")
    plot_variance_components(feature_var)
    add_selection(selection, "v2_17_feature_variance_components.png", "Feature variance components", "Main", "Variance decomposition", "Shows that prompt condition accounts for far more feature variance than run-block variation.")
    plot_seed_variation(seed_summary)
    add_selection(selection, "v2_18_seed_to_seed_variation.png", "Seed-to-seed variation", "Main", "Replication", "Clear, readable replication result.")
    plot_run_variation(run_variation)
    add_selection(selection, "v2_19_run_block_variation.png", "Run-block variation", "Appendix", "Replication", "Useful supplement to seed variation.")
    plot_reliable_features(reliability)
    add_selection(selection, "v2_20_cross_run_reliable_features.png", "Cross-run reliable features", "Appendix", "Reliability", "Strong but label-heavy; keep for appendix.")
    plot_outliers(outliers)
    add_selection(selection, "v2_21_robust_outlier_diagnostics.png", "Robust outlier diagnostics", "Appendix", "Robustness", "Useful diagnostic but not core narrative.")
    plot_feature_correlation_heatmap(image_rows)
    add_selection(selection, "v2_24_existing_feature_correlation_heatmap.png", "Selected feature correlation heatmap", "Appendix", "Collinearity diagnostic", "Shows correlation structure among feature families and motivates PCA, PLS, and shrinkage methods.")
    plot_interaction_heatmap(interactions)
    add_selection(selection, "v2_23_existing_interaction_heatmap.png", "Interaction heatmap", "Main", "Supplementary diagnostic", "Compact evidence for nonadditive factor interactions.")
    plot_nonparametric_ordering(distance_summary)
    add_selection(selection, "v2_25_existing_nonparametric_ordering.png", "Nonparametric ordering plot", "Appendix", "Supplementary diagnostic", "Supports assumption-light sensitivity analysis.")

    for scene, filename in [
        ("corner_store", "v2_26_contact_sheet_corner_store.png"),
        ("apartment", "v2_27_contact_sheet_apartment.png"),
        ("backstage", "v2_28_contact_sheet_backstage.png"),
    ]:
        contact_sheet(image_rows, scene, filename)
        add_selection(selection, filename, f"{scene.replace('_', ' ').title()} contact sheet", "Main" if scene == "corner_store" else "Appendix", "Example images", "Visual proof-of-concept for the prompt manipulations.")

    copy_existing_candidates(selection)

    write_csv(ASSET_DIR / "figure_selection.csv", selection)
    make_montages()
    print(FIG_DIR)
    print(ASSET_DIR / "figure_selection.csv")
    print(ASSET_DIR / "v2_candidate_figure_montage.png")


if __name__ == "__main__":
    main()
