# Advanced Statistical Suite

This suite extends the pooled 288-image analysis beyond the original project pass. It adds blocked permutation PERMANOVA, hierarchical bootstrap inference, fixed-effect variance decomposition, image-feature variance components, PLS-DA, LASSO/ridge logistic classification, cross-run feature reliability, and robust multivariate outlier diagnostics.

## Data Used

- Images: `288`
- Numeric image features: `195`
- Prompt-response distance rows: `432`
- Advanced output folder: `data/generated/pooled_three_run_analysis/analysis/stats/advanced_suite`

## Blocked PERMANOVA

The PERMANOVA uses PC1-PC10 from all image features and permutes residuals within `run_id` blocks.

- light: partial R2 0.118, blocked permutation p = 0.001.
- model by light: partial R2 0.039, blocked permutation p = 0.001.
- model: partial R2 0.037, blocked permutation p = 0.001.
- scan: partial R2 0.023, blocked permutation p = 0.001.
- film: partial R2 0.022, blocked permutation p = 0.001.
- film by light: partial R2 0.013, blocked permutation p = 0.001.

## Hierarchical Bootstrap Model Comparisons

The bootstrap resamples run blocks, then matched prompt contrasts within sampled runs.

- CineStill minus Portra: mean Grok minus ChatGPT -0.035, hierarchical bootstrap 95% CI [-0.954, 0.883], p = 0.949.
- Warm practical minus cool ambient: mean Grok minus ChatGPT -1.429, hierarchical bootstrap 95% CI [-2.190, -0.594], p = 0.001.
- Pushed scan minus clean scan: mean Grok minus ChatGPT -0.383, hierarchical bootstrap 95% CI [-1.405, 0.750], p = 0.474.

## PLS and Shrinkage Predictive Checks

- Model: best PLS-DA AUC 0.996 with 6 components.
- Filmstock: best PLS-DA AUC 0.835 with 6 components.
- Lighting: best PLS-DA AUC 0.997 with 3 components.
- Scan: best PLS-DA AUC 0.757 with 7 components.

Top regularized classification results:

- Lighting using Ridge logistic: AUC 0.996, accuracy 0.972.
- Lighting using LASSO logistic: AUC 0.995, accuracy 0.958.
- Model using Ridge logistic: AUC 0.993, accuracy 0.965.
- Model using LASSO logistic: AUC 0.991, accuracy 0.955.
- Filmstock using Ridge logistic: AUC 0.840, accuracy 0.747.
- Filmstock using LASSO logistic: AUC 0.792, accuracy 0.750.

## Cross-Run Feature Reliability

These are the strongest feature effects that keep the same direction across all three runs.

- chatgpt_image_2, Warm practical minus cool ambient on local_lab_cov_l_b_abs_mean: standardized effect 2.008, same sign in 3/3 runs.
- chatgpt_image_2, Warm practical minus cool ambient on local_lab_cov_l_b_p90_abs: standardized effect 1.985, same sign in 3/3 runs.
- chatgpt_image_2, Warm practical minus cool ambient on highlight_saturation: standardized effect 1.900, same sign in 3/3 runs.
- chatgpt_image_2, Warm practical minus cool ambient on split_tone_distance_lab_ab: standardized effect 1.866, same sign in 3/3 runs.
- chatgpt_image_2, Warm practical minus cool ambient on local_lab_corr_l_b_abs_mean: standardized effect 1.846, same sign in 3/3 runs.
- chatgpt_image_2, Warm practical minus cool ambient on highlight_lab_b: standardized effect 1.843, same sign in 3/3 runs.
- chatgpt_image_2, Warm practical minus cool ambient on local_lab_cov_l_b_sd: standardized effect 1.839, same sign in 3/3 runs.
- chatgpt_image_2, Warm practical minus cool ambient on highlight_warmth_rgb_r_minus_b: standardized effect 1.836, same sign in 3/3 runs.

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
