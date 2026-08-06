# Robust-XAI Framework Paper Notes

Date: 2026-06-30

Use this extension in the methodology and results sections as a robustness-aware explainability audit for SparseGuard-NIDS.

## Methodology Addition

Add a subsection after the main architecture/training/evaluation methodology:

`Robustness-Aware Explainability Evaluation`

Describe the framework as follows:

- Existing finalized SparseGuard-NIDS predictions are kept fixed.
- Gradient-input and KernelSHAP are compared to quantify cross-method explanation agreement.
- KernelSHAP top-ranked features are used as a sparse adversarial feature budget.
- A targeted PGD attack at epsilon 0.10 and 8 steps perturbs only the top-20 explanation-ranked features.
- Clean and adversarial attributions are compared using top-k Jaccard overlap, cosine similarity, and semantic-group attribution drift.

## Results To Discuss

- XAI method agreement: Spearman rho 0.6898, p-value 7.8246e-10.
- Top-20 Gradient-input vs KernelSHAP feature overlap: Jaccard 0.6000.
- Attribution stability under sparse attack: top-20 clean-vs-adversarial Jaccard 0.6433.
- Mean attribution cosine similarity under attack: 0.8377.
- Attack-to-benign evasion at epsilon 0.10: 0.0176.
- Largest attribution drift occurred in `protocol_semantics`, followed by `port_addressing`, `rate_dynamics`, `unknown_numeric`, and `byte_volume`.

## Paper Claim Supported

SparseGuard-NIDS is evaluated beyond standard predictive metrics by linking adversarial robustness to interpretable feature-group behavior. This supports a stronger Q1-SCI framing because the evaluation studies not only whether the model works, but also how its explanations behave under adversarial pressure.

## Primary Files

- `IMPLEMENTATION/ROBUST_XAI_FRAMEWORK/results/robust_xai_framework_summary.json`
- `IMPLEMENTATION/ROBUST_XAI_FRAMEWORK/results/methodology_extension_results_index.csv`
- `IMPLEMENTATION/ROBUST_XAI_FRAMEWORK/results/xai_method_topk_agreement.csv`
- `IMPLEMENTATION/ROBUST_XAI_FRAMEWORK/results/attribution_stability_under_attack.csv`
- `IMPLEMENTATION/ROBUST_XAI_FRAMEWORK/results/semantic_group_attribution_drift.csv`
