#!/usr/bin/env python3
"""Pool the three filmstock benchmark runs into analysis-ready feature tables.

This script is the pooled companion to `analyze_filmstock_dataset.py`. It keeps
the exact same image-feature extraction logic, but adds `run_id` as an explicit
blocking factor so repeated generations of the same prompt can be modeled as
seed-to-seed variation rather than treated as unrelated images.
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

import numpy as np
from scipy.stats import t, ttest_1samp, wilcoxon
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import analyze_filmstock_dataset as base


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = Path(os.environ.get("FILMSTOCK_BENCHMARK_DIR", REPO_ROOT / "data" / "generated"))
RUN_DIRS_ENV = os.environ.get("FILMSTOCK_RUN_DIRS", "")
RUN_DIRS = (
    [Path(path) for path in RUN_DIRS_ENV.split(os.pathsep) if path]
    if RUN_DIRS_ENV
    else [
        BENCHMARK_DIR / "filmstock_20260503_005006_f42a60",
        BENCHMARK_DIR / "filmstock_20260503_085439_ff0729",
        BENCHMARK_DIR / "filmstock_20260503_114446_63c69b",
    ]
)

POOLED_DIR = Path(os.environ.get("FILMSTOCK_POOLED_DIR", REPO_ROOT / "data" / "pooled_three_run_analysis"))
ANALYSIS_DIR = Path(os.environ.get("FILMSTOCK_ANALYSIS_DIR", POOLED_DIR / "analysis"))
PLOTS_DIR = ANALYSIS_DIR / "plots"

ID_COLUMNS = [
    "run_id",
    "run_block",
    "source_variant_id",
    *base.ID_COLUMNS,
]

FACTOR_COLUMNS = [
    *base.FACTOR_COLUMNS,
    "run_block_2",
    "run_block_3",
    "within_run_replicate_2",
]

MAIN_EFFECTS = {
    name: {
        **spec,
        "match": ["run_id", *spec["match"]],
    }
    for name, spec in base.MAIN_EFFECTS.items()
}


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def add_pooled_factor_columns(row: dict[str, object]) -> None:
    base.add_factor_columns(row)
    row["run_block_2"] = int(row["run_block"] == 2)
    row["run_block_3"] = int(row["run_block"] == 3)
    row["within_run_replicate_2"] = int(row["replicate"] == 2)


def clean_and_extract() -> tuple[list[dict[str, object]], list[str], dict[str, object]]:
    feature_rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []
    feature_names: list[str] = []
    manifest_count = 0

    for run_block, run_dir in enumerate(RUN_DIRS, start=1):
        run_id = run_dir.name
        manifest_rows = read_manifest(run_dir / "manifest.csv")
        manifest_count += len(manifest_rows)

        for row_number, src in enumerate(manifest_rows, start=1):
            image_path = Path(src["image_path"])
            source_variant_id = src.get("variant_id", "")
            pooled_variant_id = f"run{run_block}_{source_variant_id}"
            qc_record: dict[str, object] = {
                "run_id": run_id,
                "run_block": run_block,
                "row_number": row_number,
                "source_variant_id": source_variant_id,
                "variant_id": pooled_variant_id,
                "image_path": str(image_path),
                "manifest_status": src.get("status", ""),
                "file_exists": image_path.exists(),
                "readable": False,
            }

            if src.get("status") != "success" or not image_path.exists():
                qc_rows.append(qc_record)
                continue

            feats, qc = base.extract_image_features(image_path)
            if not feature_names:
                feature_names = list(feats.keys())

            row: dict[str, object] = {
                "run_id": run_id,
                "run_block": run_block,
                "source_variant_id": source_variant_id,
            }
            for col in base.ID_COLUMNS:
                row[col] = src.get(col, "")
            row["variant_id"] = pooled_variant_id
            row["width"] = int(src["width"])
            row["height"] = int(src["height"])
            row["replicate"] = int(src["replicate"])
            add_pooled_factor_columns(row)
            row.update(feats)
            feature_rows.append(row)

            qc_record.update(qc)
            qc_record["readable"] = True
            qc_rows.append(qc_record)

    base.write_csv(ANALYSIS_DIR / "qc_images.csv", qc_rows)
    qc_report = {
        "pooled_dir": str(POOLED_DIR),
        "run_dirs": [str(path) for path in RUN_DIRS],
        "run_ids": [path.name for path in RUN_DIRS],
        "manifest_rows": manifest_count,
        "feature_rows": len(feature_rows),
        "missing_files": sum(1 for row in qc_rows if not row["file_exists"]),
        "unreadable_files": sum(1 for row in qc_rows if row["file_exists"] and not row["readable"]),
        "non_success_manifest_rows": sum(1 for row in qc_rows if row["manifest_status"] != "success"),
        "unexpected_size_count": sum(1 for row in qc_rows if row.get("readable") and not row.get("is_expected_size")),
        "non_square_count": sum(1 for row in qc_rows if row.get("readable") and not row.get("is_square")),
        "images_with_nan_features": sum(1 for row in qc_rows if row.get("has_nan_features")),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    (ANALYSIS_DIR / "qc_report.json").write_text(json.dumps(qc_report, indent=2))
    return feature_rows, feature_names, qc_report


def summarize_conditions(
    rows: list[dict[str, object]],
    feature_names: list[str],
    include_run: bool = False,
) -> list[dict[str, object]]:
    group_keys = [
        "model_key",
        "provider",
        "api_model",
        "scene_key",
        "film_key",
        "light_key",
        "scan_key",
        "condition_id",
    ]
    if include_run:
        group_keys = ["run_id", "run_block", *group_keys]

    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in group_keys)].append(row)

    out: list[dict[str, object]] = []
    for key, items in sorted(groups.items()):
        record: dict[str, object] = {group_keys[i]: key[i] for i in range(len(group_keys))}
        record["n_images"] = len(items)
        for feature in feature_names:
            vals = [float(item[feature]) for item in items if not math.isnan(float(item[feature]))]
            record[f"{feature}_mean"] = base.safe_float(statistics.fmean(vals)) if vals else float("nan")
            record[f"{feature}_sd"] = base.safe_float(statistics.stdev(vals)) if len(vals) > 1 else float("nan")
        out.append(record)
    return out


def row_index(rows: list[dict[str, object]], keys: list[str]) -> dict[tuple[object, ...], dict[str, object]]:
    return {tuple(row[key] for key in keys): row for row in rows}


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
            base_record: dict[str, object] = {
                "run_id": low_row["run_id"],
                "run_block": low_row["run_block"],
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
                        **base_record,
                        "feature": feature,
                        "effect_value": base.safe_float(float(high_row[feature]) - float(low_row[feature])),
                    }
                )
    return effects


def condition_value_map(
    rows: list[dict[str, object]],
) -> dict[tuple[str, str, int, str, str, str, str, str], dict[str, object]]:
    out: dict[tuple[str, str, int, str, str, str, str, str], dict[str, object]] = {}
    for row in rows:
        key = (
            str(row["model_key"]),
            str(row["run_id"]),
            int(row["replicate"]),
            str(row["scene_key"]),
            str(row["film_key"]),
            str(row["light_key"]),
            str(row["scan_key"]),
            str(row["variant_id"]),
        )
        out[key] = row
    return out


def exact_condition_map(
    rows: list[dict[str, object]],
) -> dict[tuple[str, str, int, str, str, str, str], dict[str, object]]:
    return {
        (
            str(row["model_key"]),
            str(row["run_id"]),
            int(row["replicate"]),
            str(row["scene_key"]),
            str(row["film_key"]),
            str(row["light_key"]),
            str(row["scan_key"]),
        ): row
        for row in rows
    }


def interaction_effects(rows: list[dict[str, object]], feature_names: list[str]) -> list[dict[str, object]]:
    values = exact_condition_map(rows)
    models = sorted({str(row["model_key"]) for row in rows})
    run_ids = sorted({str(row["run_id"]) for row in rows})
    run_block_by_id = {str(row["run_id"]): int(row["run_block"]) for row in rows}
    replicates = sorted({int(row["replicate"]) for row in rows})
    scenes = sorted({str(row["scene_key"]) for row in rows})
    films = ["portra400", "cinestill800t"]
    lights = ["cool_ambient", "warm_practical"]
    scans = ["clean_scan", "pushed_scan"]
    out: list[dict[str, object]] = []

    for model in models:
        for run_id in run_ids:
            for replicate in replicates:
                for scene in scenes:
                    common = {
                        "model_key": model,
                        "run_id": run_id,
                        "run_block": run_block_by_id[run_id],
                        "scene_key": scene,
                        "replicate": replicate,
                    }

                    for scan in scans:
                        keys = [(model, run_id, replicate, scene, film, light, scan) for film in films for light in lights]
                        if all(key in values for key in keys):
                            for feature in feature_names:
                                val = (
                                    float(values[(model, run_id, replicate, scene, "cinestill800t", "warm_practical", scan)][feature])
                                    - float(values[(model, run_id, replicate, scene, "portra400", "warm_practical", scan)][feature])
                                    - float(values[(model, run_id, replicate, scene, "cinestill800t", "cool_ambient", scan)][feature])
                                    + float(values[(model, run_id, replicate, scene, "portra400", "cool_ambient", scan)][feature])
                                )
                                out.append({**common, "effect_type": "filmstock_by_lighting", "effect_family": "two_factor_interaction", "scan_key": scan, "feature": feature, "effect_value": base.safe_float(val), "n_condition_means": 4, "n_images_used": 4})

                    for light in lights:
                        keys = [(model, run_id, replicate, scene, film, light, scan) for film in films for scan in scans]
                        if all(key in values for key in keys):
                            for feature in feature_names:
                                val = (
                                    float(values[(model, run_id, replicate, scene, "cinestill800t", light, "pushed_scan")][feature])
                                    - float(values[(model, run_id, replicate, scene, "portra400", light, "pushed_scan")][feature])
                                    - float(values[(model, run_id, replicate, scene, "cinestill800t", light, "clean_scan")][feature])
                                    + float(values[(model, run_id, replicate, scene, "portra400", light, "clean_scan")][feature])
                                )
                                out.append({**common, "effect_type": "filmstock_by_scan", "effect_family": "two_factor_interaction", "light_key": light, "feature": feature, "effect_value": base.safe_float(val), "n_condition_means": 4, "n_images_used": 4})

                    for film in films:
                        keys = [(model, run_id, replicate, scene, film, light, scan) for light in lights for scan in scans]
                        if all(key in values for key in keys):
                            for feature in feature_names:
                                val = (
                                    float(values[(model, run_id, replicate, scene, film, "warm_practical", "pushed_scan")][feature])
                                    - float(values[(model, run_id, replicate, scene, film, "cool_ambient", "pushed_scan")][feature])
                                    - float(values[(model, run_id, replicate, scene, film, "warm_practical", "clean_scan")][feature])
                                    + float(values[(model, run_id, replicate, scene, film, "cool_ambient", "clean_scan")][feature])
                                )
                                out.append({**common, "effect_type": "lighting_by_scan", "effect_family": "two_factor_interaction", "film_key": film, "feature": feature, "effect_value": base.safe_float(val), "n_condition_means": 4, "n_images_used": 4})

                    keys = [(model, run_id, replicate, scene, film, light, scan) for film in films for light in lights for scan in scans]
                    if all(key in values for key in keys):
                        for feature in feature_names:
                            val = 0.0
                            for film in films:
                                for light in lights:
                                    for scan in scans:
                                        sign = 1.0
                                        sign *= 1.0 if film == "cinestill800t" else -1.0
                                        sign *= 1.0 if light == "warm_practical" else -1.0
                                        sign *= 1.0 if scan == "pushed_scan" else -1.0
                                        val += sign * float(values[(model, run_id, replicate, scene, film, light, scan)][feature])
                            out.append({**common, "effect_type": "filmstock_by_lighting_by_scan", "effect_family": "three_factor_interaction", "feature": feature, "effect_value": base.safe_float(val), "n_condition_means": 8, "n_images_used": 8})
    return out


def responsiveness_distances(rows: list[dict[str, object]], feature_names: list[str]) -> list[dict[str, object]]:
    used_features = [feature for feature in base.EFFECT_FEATURES if feature in feature_names]
    color_features = [feature for feature in base.COLOR_RESPONSE_FEATURES if feature in feature_names]
    contrast_features = [feature for feature in base.CONTRAST_TEXTURE_FEATURES if feature in feature_names]
    structure_features = [feature for feature in base.STRUCTURE_FEATURES if feature in feature_names]
    z_features: list[str] = []
    for feature in [*used_features, *color_features, *contrast_features, *structure_features]:
        if feature not in z_features:
            z_features.append(feature)
    z_by_variant, _ = base.zscore_rows(rows, z_features)
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
                return base.safe_float(np.sqrt(np.sum(diff * diff)))

            out.append(
                {
                    "run_id": low_row["run_id"],
                    "run_block": low_row["run_block"],
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


def model_comparison_tests(distance_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    distance_cols = [
        "distance_all_selected_features",
        "distance_color_features",
        "distance_contrast_texture_features",
        "distance_structure_features",
    ]
    out: list[dict[str, object]] = []
    for effect_type in sorted({str(row["effect_type"]) for row in distance_rows}):
        effect_rows = [row for row in distance_rows if row["effect_type"] == effect_type]
        match_keys = sorted({
            (
                str(row["run_id"]),
                str(row["scene_key"]),
                str(row.get("film_key", "")),
                str(row.get("light_key", "")),
                str(row.get("scan_key", "")),
                str(row["replicate"]),
            )
            for row in effect_rows
        })
        indexed = {
            (
                str(row["model_key"]),
                str(row["run_id"]),
                str(row["scene_key"]),
                str(row.get("film_key", "")),
                str(row.get("light_key", "")),
                str(row.get("scan_key", "")),
                str(row["replicate"]),
            ): row
            for row in effect_rows
        }
        for metric in distance_cols:
            diffs = []
            for key in match_keys:
                gpt = indexed.get(("chatgpt_image_2", *key))
                xai = indexed.get(("xai_grok_imagine", *key))
                if gpt and xai:
                    diffs.append(float(xai[metric]) - float(gpt[metric]))
            arr = np.asarray(diffs, dtype=np.float64)
            if arr.size < 2:
                continue
            mean_diff = float(arr.mean())
            sd_diff = float(arr.std(ddof=1))
            se_diff = sd_diff / math.sqrt(arr.size)
            t_res = ttest_1samp(arr, popmean=0.0)
            try:
                w_res = wilcoxon(arr)
                wilcoxon_stat = base.safe_float(w_res.statistic)
                wilcoxon_p = base.safe_float(w_res.pvalue)
            except ValueError:
                wilcoxon_stat = float("nan")
                wilcoxon_p = float("nan")
            crit = t.ppf(0.975, df=arr.size - 1)
            out.append(
                {
                    "effect_type": effect_type,
                    "distance_metric": metric,
                    "comparison": "xai_grok_imagine_minus_chatgpt_image_2",
                    "n_paired_conditions": int(arr.size),
                    "mean_difference": base.safe_float(mean_diff),
                    "median_difference": base.safe_float(np.median(arr)),
                    "sd_difference": base.safe_float(sd_diff),
                    "se_difference": base.safe_float(se_diff),
                    "ci95_low": base.safe_float(mean_diff - crit * se_diff),
                    "ci95_high": base.safe_float(mean_diff + crit * se_diff),
                    "paired_t_statistic": base.safe_float(t_res.statistic),
                    "paired_t_p_value": base.safe_float(t_res.pvalue),
                    "cohens_dz": base.safe_float(mean_diff / sd_diff) if sd_diff > 0 else float("nan"),
                    "wilcoxon_statistic": wilcoxon_stat,
                    "wilcoxon_p_value": wilcoxon_p,
                }
            )
    return out


def run_pca(rows: list[dict[str, object]], feature_names: list[str]) -> None:
    pca_features = [feature for feature in feature_names if not feature.startswith("hue_hist_12bin_")]
    mat = base.numeric_matrix(rows, pca_features)
    z = StandardScaler().fit_transform(mat)
    pca = PCA(n_components=min(10, z.shape[0], z.shape[1]))
    scores = pca.fit_transform(z)

    score_rows: list[dict[str, object]] = []
    for i, row in enumerate(rows):
        record = {col: row[col] for col in ID_COLUMNS if col in row and col != "prompt"}
        for j in range(scores.shape[1]):
            record[f"PC{j + 1}"] = base.safe_float(scores[i, j])
        score_rows.append(record)
    base.write_csv(ANALYSIS_DIR / "pca_image_scores.csv", score_rows)

    loading_rows = []
    for j in range(scores.shape[1]):
        for i, feature in enumerate(pca_features):
            loading_rows.append({"component": f"PC{j + 1}", "feature": feature, "loading": base.safe_float(pca.components_[j, i])})
    base.write_csv(ANALYSIS_DIR / "pca_loadings.csv", loading_rows)

    explained_rows = [
        {
            "component": f"PC{i + 1}",
            "explained_variance_ratio": base.safe_float(ratio),
            "cumulative_explained_variance_ratio": base.safe_float(np.sum(pca.explained_variance_ratio_[: i + 1])),
        }
        for i, ratio in enumerate(pca.explained_variance_ratio_)
    ]
    base.write_csv(ANALYSIS_DIR / "pca_explained_variance.csv", explained_rows)

    ellipse_rows = base.pca_confidence_ellipses(
        score_rows,
        ["model_key", "model_key|film_key", "model_key|light_key", "model_key|run_id"],
    )
    base.write_csv(ANALYSIS_DIR / "pca_confidence_ellipses.csv", ellipse_rows)

    old_plots_dir = base.PLOTS_DIR
    base.PLOTS_DIR = PLOTS_DIR
    try:
        base.make_pca_plot(score_rows, ellipse_rows)
    finally:
        base.PLOTS_DIR = old_plots_dir


def write_analysis_readme(qc_report: dict[str, object]) -> None:
    text = f"""# Pooled Three-Run Analysis Outputs

Created: `{qc_report['created_at']}`

This folder pools the three complete 96-image filmstock benchmark runs into one
288-image dataset. `run_id` is preserved as a blocking factor, and the original
within-run `replicate` value is preserved as a nested repeat for each prompt.

## Pooled Runs

{chr(10).join(f"- `{run_id}`" for run_id in qc_report["run_ids"])}

## Primary Tables

- `image_features.csv`: one row per image with `run_id`, `run_block`, design factors, and numeric image features.
- `condition_features.csv`: condition means pooled across the three runs.
- `condition_features_by_run.csv`: condition means within each run block.
- `effect_pairs_long.csv`: blocked high-minus-low main effects and factorial interaction contrasts for every feature.
- `responsiveness_distances.csv`: blocked standardized prompt-response distances.
- `model_comparison_tests.csv`: paired model-comparison tests matching on `run_id`, scene, prompt factors, and replicate.
- `pca_image_scores.csv`, `pca_loadings.csv`, `pca_explained_variance.csv`, `pca_confidence_ellipses.csv`: PCA outputs for the pooled feature matrix.
- `qc_images.csv` and `qc_report.json`: data-cleaning checks.

## Cleaning Result

- Manifest rows: `{qc_report['manifest_rows']}`
- Feature rows: `{qc_report['feature_rows']}`
- Missing files: `{qc_report['missing_files']}`
- Unreadable files: `{qc_report['unreadable_files']}`
- Unexpected image size: `{qc_report['unexpected_size_count']}`
- Non-square images: `{qc_report['non_square_count']}`

## Blocking Strategy

Main effects are paired within `run_id` and within-run `replicate`. This means a
CineStill-minus-Portra contrast only compares images from the same model, same
scene, same lighting condition, same scan condition, same generation run, and
same within-run replicate. The same blocking logic is used for model comparison
tests and the downstream statistical analysis.
"""
    (ANALYSIS_DIR / "README_ANALYSIS.md").write_text(text)


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    rows, feature_names, qc_report = clean_and_extract()
    base.write_csv(ANALYSIS_DIR / "image_features.csv", rows, [*ID_COLUMNS, *FACTOR_COLUMNS, *feature_names])
    base.write_csv(ANALYSIS_DIR / "feature_dictionary.csv", base.feature_dictionary(feature_names))

    condition_rows = summarize_conditions(rows, feature_names, include_run=False)
    condition_by_run_rows = summarize_conditions(rows, feature_names, include_run=True)
    base.write_csv(ANALYSIS_DIR / "condition_features.csv", condition_rows)
    base.write_csv(ANALYSIS_DIR / "condition_features_by_run.csv", condition_by_run_rows)

    main_effect_rows = paired_main_effects(rows, feature_names)
    interaction_rows = interaction_effects(rows, feature_names)
    all_effect_rows = [*main_effect_rows, *interaction_rows]
    base.write_csv(ANALYSIS_DIR / "effect_pairs_long.csv", all_effect_rows)
    base.write_csv(ANALYSIS_DIR / "effect_summary_by_model.csv", base.summarize_effects(all_effect_rows))

    distance_rows = responsiveness_distances(rows, feature_names)
    base.write_csv(ANALYSIS_DIR / "responsiveness_distances.csv", distance_rows)
    base.write_csv(ANALYSIS_DIR / "responsiveness_distance_summary.csv", base.summarize_distances(distance_rows))
    base.write_csv(ANALYSIS_DIR / "model_comparison_tests.csv", model_comparison_tests(distance_rows))

    run_pca(rows, feature_names)
    write_analysis_readme(qc_report)

    print(json.dumps({
        "analysis_dir": str(ANALYSIS_DIR),
        "image_feature_rows": len(rows),
        "feature_count": len(feature_names),
        "condition_rows": len(condition_rows),
        "condition_by_run_rows": len(condition_by_run_rows),
        "effect_rows": len(all_effect_rows),
        "responsiveness_distance_rows": len(distance_rows),
        "qc": qc_report,
    }, indent=2))


if __name__ == "__main__":
    main()
