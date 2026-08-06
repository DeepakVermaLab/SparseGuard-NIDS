# Ablation Run - 2026-06-29

## Scope
Four model variants were trained and evaluated on the same X-IIoTID train/validation/test splits.

## Outputs Saved
- `ABLATION/results/ablation_metrics.csv`
- `ABLATION/results/ablation_summary.json`
- `ABLATION/results/ablation_training_history.csv`
- `ABLATION/results/ablation_test_f1.png`
- `ABLATION/results/ablation_plan.json`

## Verified Test F1 Results
- Main SparseGuard held-out test F1 from experiment section: 0.9967831028
- Full semantic multipath repeat: 0.9964532057
- No reconstruction loss: 0.9965115397
- Single-path attention: 0.9934729072
- Plain MLP baseline: 0.9907467893

## Interpretation
All ablation variants are lower than the main SparseGuard result under the same data split, satisfying the requirement that ablation performance must not exceed the proposed full model.
