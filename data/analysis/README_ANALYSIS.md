# Pooled Three-Run Analysis Outputs

Created: `2026-05-03T15:02:33-04:00`

This folder pools the three complete 96-image filmstock benchmark runs into one
288-image dataset. `run_id` is preserved as a blocking factor, and the original
within-run `replicate` value is preserved as a nested repeat for each prompt.

## Pooled Runs

- `filmstock_20260503_005006_f42a60`
- `filmstock_20260503_085439_ff0729`
- `filmstock_20260503_114446_63c69b`

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

- Manifest rows: `288`
- Feature rows: `288`
- Missing files: `0`
- Unreadable files: `0`
- Unexpected image size: `0`
- Non-square images: `0`

## Blocking Strategy

Main effects are paired within `run_id` and within-run `replicate`. This means a
CineStill-minus-Portra contrast only compares images from the same model, same
scene, same lighting condition, same scan condition, same generation run, and
same within-run replicate. The same blocking logic is used for model comparison
tests and the downstream statistical analysis.
