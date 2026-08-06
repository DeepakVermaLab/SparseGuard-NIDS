# Paper Assets Run - 2026-06-29

## Scope
Paper asset planning was refreshed after the main experiment, ablation, robustness, and external evaluation runs.

## Outputs Saved
- `PAPER_ASSETS/results/paper_asset_plan.json`

## Tables Now Supported By Results
- Main X-IIoTID performance table.
- Ablation table with four variants below the main model.
- Robustness table for FGSM epsilon sweep.
- External validation table for within-dataset, mixed multi-dataset holdout, few-shot target adaptation, zero-shot transfer, and leave-one-dataset-out diagnostics.
- Runtime table using stage timings, epoch timings, ablation timings, robustness timings, and external validation runtime profile.

## Figures Now Supported By Results
- Training curves.
- Held-out confusion matrix and ROC/PR curves.
- Ablation F1 bar plot.
- FGSM evasion curve.
- Cross-dataset F1 heatmap.
- Cross-dataset confusion matrix panel.
