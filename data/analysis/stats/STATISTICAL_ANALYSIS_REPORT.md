# Pooled Statistical Analysis Report

This folder contains the pooled inferential analysis for the three-run filmstock responsiveness benchmark. The analysis uses 288 generated images: three independent generation runs, two models, three scenes, two film stocks, two lighting conditions, two scan conditions, and two within-run replicates.

The key design change from the first analysis is that `run_id` is now treated as a blocking factor. Main-effect contrasts are paired within run and within replicate, regression models include run-block fixed effects, nonparametric repeated-measures tests block by run-scene-replicate, and the permutation MANOVA permutes residuals within run blocks.

## Model Comparison on Total Responsiveness

- CineStill minus Portra: ChatGPT mean 7.294, Grok mean 7.259, paired difference -0.035, 95% CI [-0.719, 0.648], p = 0.919.
- Warm practical minus cool ambient: ChatGPT mean 10.335, Grok mean 8.906, paired difference -1.429, 95% CI [-2.189, -0.668], p = 0.000.
- Pushed scan minus clean scan: ChatGPT mean 7.226, Grok mean 6.843, paired difference -0.383, 95% CI [-1.072, 0.306], p = 0.271.

## Targeted Filmstock and Lighting Features

- Warm practical minus cool ambient on Split-tone Lab distance: model difference -12.4935, p = 0.000, Holm p = 0.000.
- Warm practical minus cool ambient on Highlight-shadow Lab b split: model difference -7.4181, p = 0.000, Holm p = 0.000.
- Warm practical minus cool ambient on Lab yellow minus blue: model difference -2.8649, p = 0.000, Holm p = 0.002.
- Pushed scan minus clean scan on Lab yellow minus blue: model difference -1.0842, p = 0.109, Holm p = 1.000.
- CineStill minus Portra on Split-tone Lab distance: model difference -0.9172, p = 0.440, Holm p = 1.000.
- CineStill minus Portra on Local a-b covariance: model difference -0.7366, p = 0.740, Holm p = 1.000.

`all_feature_model_screening.csv` screens every extracted feature, not just the pre-registered targeted set. Use that table as the exploratory layer for deciding which richer image descriptors tell the clearest story in the final writeup.

## PCA and Blocked MANOVA-Style Testing

- light: Pillai trace 0.645, blocked permutation p = 0.001.
- model by light: Pillai trace 0.233, blocked permutation p = 0.001.
- film by light: Pillai trace 0.124, blocked permutation p = 0.001.
- film: Pillai trace 0.123, blocked permutation p = 0.001.
- model: Pillai trace 0.120, blocked permutation p = 0.001.
- scan: Pillai trace 0.095, blocked permutation p = 0.001.

## Regression, WLS, and Robust Estimation

- xAI model: beta -0.035, 95% CI [-0.711, 0.641], p = 0.919.
- xAI by lighting effect: beta -1.394, 95% CI [-2.395, -0.392], p = 0.007.
- xAI by scan effect: beta -0.348, 95% CI [-1.293, 0.597], p = 0.470.
- Run block 2: beta -0.607, 95% CI [-1.116, -0.098], p = 0.020.
- Run block 3: beta -0.580, 95% CI [-1.104, -0.056], p = 0.030.

The regression table reports OLS with HC3 robust standard errors, two-stage WLS, and Huber IRLS robust regression. Run blocks are included directly as fixed effects, so the model comparison is not simply pooling all generations as if they came from one batch.

## Seed-to-Seed and Run-to-Run Variation

- ChatGPT Image 2: mean pairwise exact-prompt seed distance 4.709, 95% CI [4.336, 5.082].
- Grok Imagine: mean pairwise exact-prompt seed distance 4.257, 95% CI [3.955, 4.560].

For prompt-response distances, the run-level descriptive variance checks are:

- ChatGPT Image 2, CineStill minus Portra: between-run SD of means 0.084; pooled within-run SD 1.909.
- ChatGPT Image 2, Warm practical minus cool ambient: between-run SD of means 0.510; pooled within-run SD 2.032.
- ChatGPT Image 2, Pushed scan minus clean scan: between-run SD of means 0.208; pooled within-run SD 1.705.
- Grok Imagine, CineStill minus Portra: between-run SD of means 0.694; pooled within-run SD 2.142.
- Grok Imagine, Warm practical minus cool ambient: between-run SD of means 0.557; pooled within-run SD 2.430.
- Grok Imagine, Pushed scan minus clean scan: between-run SD of means 0.918; pooled within-run SD 2.260.

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
