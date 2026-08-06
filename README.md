# SparseGuard-NIDS Implementation

SparseGuard-NIDS is a semantics-aware multi-path Network Intrusion Detection System (NIDS) implementation prepared as a GitHub-style research repository for paper submission and reproducibility.

The finalized version is documented as `SparseGuard-NIDS Implementation v1.0 Final`.

## Method

SparseGuard-NIDS separates protocol, flow timing, packet volume, byte volume, rate dynamics, addressing, statistical, and unknown numeric evidence into independent semantic branches. These branches are fused with attention, and the pipeline adds reconstruction and semantic-consistency outputs to support robustness against sparse attribution-guided attacks.

## Datasets

- `X-IIoTID`: base dataset.
- `CIC-IIoT-2025`: external IIoT validation.
- `CIC-IDS2017`: external NIDS validation.

Large raw datasets are not stored in this repository. Configure local or Google Drive dataset paths before running full experiments.

## Repository Structure

```text
SparseGuard-NIDS/
├── src/                         # Core SparseGuard pipeline code
├── scripts/                     # Project-level run helpers
├── configs/                     # Dataset and experiment configuration
├── notebooks/                   # Full project / controller notebooks
├── EDA/                         # Dataset inspection and exploratory results
├── PREPROCESSING/               # Cleaning, encoding, scaling, split creation
├── EXPERIMENT/                  # Main SparseGuard training and test outputs
├── ABLATION/                    # Component-removal study assets
├── EVALUATION/                  # External and cross-dataset validation assets
├── XAI/                         # Attribution and semantic-group explanation assets
├── ROBUSTNESS/                  # Adversarial and reconstruction robustness assets
├── ROBUST_XAI_FRAMEWORK/        # Robustness-aware XAI audit assets
├── Q1_VALIDATION/               # Q1 SCI readiness validation assets
├── PROFILING/                   # Runtime, memory, FLOPs, and energy-proxy assets
├── PAPER_ASSETS/                # Paper-ready tables, final audits, and version markers
└── logs/                        # Root log notes
```

Each major stage follows this documentation format:

```text
<STAGE>/
├── scripts/                     # Python entry point for the stage
├── notebooks/                   # Google Colab notebook for the stage
├── results/                     # CSV, JSON, PNG, checkpoint, or table outputs
└── DOCUMANTATION/               # Run notes and manuscript/reproducibility notes
```

Note: the folder name `DOCUMANTATION` is preserved from the finalized Drive package so existing paths stay consistent.

## Setup

Create a Python environment and install dependencies:

```bash
cd /users/
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Expose the source package for the lightweight wrapper scripts:

```bash
export PYTHONPATH="$PWD/src/sparseguard:$PYTHONPATH"
export SPARSEGUARD_ROOT="$PWD"
```

If running full data-dependent stages, also point the code to the dataset root:

```bash
export SPARSEGUARD_DATASET_ROOT="/users/"
```

## Run Commands

Run individual available stage scripts:

```bash
python EDA/scripts/run_eda.py
python EXPERIMENT/scripts/train_sparseguard.py
python ABLATION/scripts/run_ablation.py
python EVALUATION/scripts/run_external_validation.py
```

Run the project-level helper:

```bash
python scripts/run_all.py
```

Important: `scripts/run_all.py` expects every stage script to exist locally. If a stage script is missing from this GitHub upload package, run that stage from its notebook or restore the script from the finalized Drive package.

## Notebook Run Order

Official experiments were designed to run from Google Colab with Drive mounted. Use this order:

1. `notebooks/00_FULL_PROJECT_CONTROL_COLAB.ipynb`
2. `notebooks/00_RUN_ALL_COLAB.ipynb`
3. `EDA/notebooks/01_EDA.ipynb`
4. `PREPROCESSING/notebooks/02_PREPROCESSING.ipynb`
5. `EXPERIMENT/notebooks/03_TRAIN_SPARSEGUARD.ipynb`
6. `ABLATION/notebooks/04_ABLATION.ipynb`
7. `ROBUSTNESS/notebooks/05_ROBUSTNESS_EVAL.ipynb`
8. `EVALUATION/notebooks/06_EXTERNAL_VALIDATION.ipynb`
9. `ROBUST_XAI_FRAMEWORK/notebooks/01_ROBUST_XAI_FRAMEWORK_COLAB.ipynb`
10. `Q1_VALIDATION/notebooks/01_Q1_SCI_VALIDATION_COLAB.ipynb`

After each notebook run, save outputs under that stage's `results/` folder and write or update the run note under `DOCUMANTATION/`.

## Outputs and Logs

- Stage outputs belong in `results/`.
- Human-readable run notes belong in `DOCUMANTATION/`.
- Final paper tables and audit files belong in `PAPER_ASSETS/`.
- The Drive root `logs` folder was empty when this package was prepared; see `logs/README.md`.

## Final Version Evidence

Finalization and paper-readiness files:

- `PAPER_ASSETS/DOCUMANTATION/IMPLEMENTATION_VERSION_v1.0_FINAL.md`
- `PAPER_ASSETS/DOCUMANTATION/PAPER_ASSETS_FINAL_AUDIT_20260630.md`
- `PAPER_ASSETS/DOCUMANTATION/Q1_SCI_READINESS_DOCUMENTATION_20260630.md`
- `PAPER_ASSETS/results/FINAL_COMPLETION_AUDIT_20260630.md`
- `PAPER_ASSETS/results/IMPLEMENTATION_VERSION_v1.0_FINAL.json`

## GitHub Notes

The `.gitignore` excludes Python caches, notebook checkpoints, environment files, and model checkpoint files such as `*.pt` and `*.pth`.

If you want to publish model checkpoints, use Git LFS or attach them as release assets instead of committing them directly.
