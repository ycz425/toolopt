import json
from pathlib import Path

RUNS_ROOT = Path("experiments/runs")


def write_run_info(run_dir: Path, info: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=2) + "\n")


def read_run_info(run_dir: Path) -> dict:
    return json.loads((run_dir / "run_info.json").read_text())
