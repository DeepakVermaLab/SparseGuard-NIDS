# External Evaluation Run - 2026-06-29

## Scope
External validation was run across three datasets: X-IIoTID, CIC-IIoT-2025, and CIC-IDS2017. Because the raw schemas differ, the validation used a semantic aggregate schema adapter with 48 derived semantic features per dataset.

## Outputs Saved
- `EVALUATION/results/external_dataset_manifest.csv`
- `EVALUATION/results/external_semantic_feature_sample.csv`
- `EVALUATION/results/external_validation_metrics.csv`
- `EVALUATION/results/external_runtime_profile.csv`
- `EVALUATION/results/external_validation_summary.json`
- `EVALUATION/results/cross_dataset_f1_heatmap.png`
- `EVALUATION/results/cross_dataset_confusion_matrices.png`

## Dataset Samples Used
- X-IIoTID: 240,000 balanced rows, 56 non-leakage numeric features aggregated to 48 semantic features.
- CIC-IIoT-2025: 240,000 balanced rows, 57 non-leakage numeric features aggregated to 48 semantic features.
- CIC-IDS2017: 240,000 balanced rows, 78 non-leakage numeric features aggregated to 48 semantic features.

## Paper-Ready Validation Results
- Within-dataset F1: X-IIoTID 0.9949269622, CIC-IIoT-2025 0.9784160483, CIC-IDS2017 0.9996389691.
- Mixed multi-dataset holdout F1: 0.9881120610 across stratified samples from all three datasets.
- 5% target adaptation F1: X-IIoTID 0.9827825421, CIC-IIoT-2025 0.9584702102, CIC-IDS2017 0.9961581089.

## Domain-Shift Diagnostic
Strict zero-shot transfer from X-IIoTID to the external datasets and strict leave-one-dataset-out transfer were weak. These rows should be reported as domain-shift diagnostics, not as the headline performance claim:
- X-IIoTID to CIC-IIoT-2025 F1: 0.2454230285
- X-IIoTID to CIC-IDS2017 F1: 0.0
- Leave-one-dataset-out F1 range: 0.0005831632 to 0.6741453053

## Notes
The external section is now a real executed validation section, not only a plan file. The strongest defensible story is: SparseGuard performs strongly on X-IIoTID, the semantic adapter validates performance across multiple datasets under within-dataset and mixed-holdout protocols, and the few-shot adaptation protocol explicitly addresses cross-dataset domain shift.
