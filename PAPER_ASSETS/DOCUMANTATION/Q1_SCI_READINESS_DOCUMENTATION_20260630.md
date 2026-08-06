# Q1 SCI Readiness Documentation

Date: 2026-06-30

This section adds methodology-level validation evidence for a high-impact Q1 SCI submission target.

## Readiness Gate

Status: **PASS for methodology evidence package**

Scope note: this indicates that the implementation now contains the validation evidence expected for a high-impact submission. Final acceptance still depends on manuscript writing, journal fit, editorial scope, and reviewer response.

## Main Held-Out Statistical Reliability

- Bootstrap F1 mean: 0.996759
- Bootstrap F1 95% CI: [0.996287, 0.997203]

## Added Methodology Evidence

- Formal threat model and sparse adaptive attacker formulation.
- Main model bootstrap confidence intervals.
- Repeated-seed multi-dataset validation.
- Leave-one-dataset-out domain-shift stress testing.
- Few-shot target-domain adaptation with paired significance tests.
- Calibration reliability bins and threshold sensitivity curves.
- Adaptive adversary budget/evasion analysis.
- Complexity-performance Pareto analysis.
- Integration with Robust-XAI attribution stability evidence.

## Primary Result Files

- `q1_sci_readiness_gate.csv`
- `q1_sci_readiness_summary.json`
- `main_model_bootstrap_confidence_intervals.csv`
- `calibration_reliability_bins.csv`
- `threshold_sensitivity_metrics.csv`
- `repeated_domain_generalization_metrics.csv`
- `repeated_domain_generalization_confidence_intervals.csv`
- `paired_domain_shift_significance_tests.csv`
- `adaptive_adversary_budget_summary.csv`
- `complexity_performance_pareto_table.csv`
