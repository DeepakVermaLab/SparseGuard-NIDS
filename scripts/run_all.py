import subprocess, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
for s in ['EDA/scripts/run_eda.py','PREPROCESSING/scripts/run_preprocessing.py','EXPERIMENT/scripts/train_sparseguard.py','ABLATION/scripts/run_ablation.py','ROBUSTNESS/scripts/run_adversarial_robustness.py','EVALUATION/scripts/run_external_validation.py','PAPER_ASSETS/scripts/build_paper_tables.py']:
    print('RUN', s); subprocess.run(['python', str(ROOT/s)], check=True)
