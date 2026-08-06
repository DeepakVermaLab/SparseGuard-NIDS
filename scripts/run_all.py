import pathlib
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]

STAGE_SCRIPTS = [
    "EDA/scripts/run_eda.py",
    "EXPERIMENT/scripts/train_sparseguard.py",
    "ABLATION/scripts/run_ablation.py",
    "EVALUATION/scripts/run_external_validation.py",
]


def main() -> None:
    for script in STAGE_SCRIPTS:
        path = ROOT / script
        if not path.exists():
            print(f"SKIP missing script: {script}", file=sys.stderr)
            continue
        print(f"RUN {script}")
        subprocess.run([sys.executable, str(path)], check=True)


if __name__ == "__main__":
    main()
