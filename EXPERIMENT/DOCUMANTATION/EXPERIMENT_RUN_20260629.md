# Main Experiment Run - 2026-06-29

## Scope
SparseGuard-NIDS main model was trained on leakage-safe X-IIoTID splits in Google Colab with L4 GPU and High-RAM runtime.

## Outputs Saved
- `EXPERIMENT/results/sparseguard_best.pt`
- `EXPERIMENT/results/training_history.csv`
- `EXPERIMENT/results/training_curves.png`
- `EXPERIMENT/results/training_summary.json`
- `EXPERIMENT/results/test_metrics.json`
- `EXPERIMENT/results/test_prediction_sample.csv`
- `EXPERIMENT/results/test_confusion_matrix.png`
- `EXPERIMENT/results/test_roc_pr_curves.png`

## Training Summary
- Device: CUDA
- Epochs completed: 54
- Best validation F1: 0.9970548456
- Semantic groups used: byte volume, flow timing, packet volume, port/addressing, protocol semantics, rate dynamics, statistical flags, unknown numeric.

## Held-Out Test Metrics
- Accuracy: 0.9968745395
- Balanced accuracy: 0.9968415084
- Precision: 0.9979372833
- Recall: 0.9956315889
- F1: 0.9967831028
- MCC: 0.9937465462
- ROC-AUC: 0.9998646885
- Average precision: 0.9998712202
- Brier score: 0.0025629195
- Confusion matrix: `[[83488, 163], [346, 78859]]`

## Notes
The held-out result is the main paper headline result for the base-paper dataset. It is above the ablation variants saved in the ablation section.
