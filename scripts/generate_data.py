import json
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from app.data.generator import generate_trajectories
from app.data.task_bank import build_task_bank
from app.data.labeled_task import LabeledTask
from app.data.versions import DATASET_ROOT, next_version_dir
from app.environment.environment import Environment
from app.tools.executor import ToolExecutor
from app.tools.registry_builder import build_registry

VAL_SIZE = 0.1
TEST_SIZE = 0.2
SPLIT_SEED = 42


def group_key(task: LabeledTask) -> str:
    """Identifies the underlying task: its tool calls, ignoring how the description is worded."""
    return json.dumps([action.model_dump() for action in task.expected_actions], sort_keys=True)


def grouped_split(
    tasks: list[LabeledTask], val_size: float, test_size: float, seed: int = 42
) -> tuple[list[LabeledTask], list[LabeledTask], list[LabeledTask]]:
    """Splits tasks into train/val/test so every wording of the same underlying task lands in the same split.

    Groups are split within each template, keeping every split stratified by template. The val and
    test fractions are approximate, since they're counted in groups rather than tasks. A template
    with very few groups may get none in val or test.
    """
    rng = random.Random(seed)
    by_template: dict[str, dict[str, list[LabeledTask]]] = defaultdict(lambda: defaultdict(list))
    for task in tasks:
        by_template[task.template][group_key(task)].append(task)

    train, val, test = [], [], []
    for template in sorted(by_template):
        groups = by_template[template]
        keys = sorted(groups)
        rng.shuffle(keys)
        n_test = round(len(keys) * test_size)
        n_val = round(len(keys) * val_size)
        for i, key in enumerate(keys):
            split = test if i < n_test else val if i < n_test + n_val else train
            split.extend(groups[key])
    return train, val, test


def write_jsonl(path: Path, records: list[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for record in records:
            f.write(record.model_dump_json() + "\n")
    print(f"Wrote {len(records)} records to {path}")


def main():
    version_dir = next_version_dir(DATASET_ROOT)
    environment = Environment(build_registry(), ToolExecutor())

    train_tasks, val_tasks, test_tasks = grouped_split(build_task_bank(), VAL_SIZE, TEST_SIZE, seed=SPLIT_SEED)

    # Test tasks get no trajectories: the benchmark runs the model on them live.
    train_trajectories = generate_trajectories(environment, train_tasks)
    val_trajectories = generate_trajectories(environment, val_tasks)

    write_jsonl(version_dir / "train_tasks.jsonl", train_tasks)
    write_jsonl(version_dir / "val_tasks.jsonl", val_tasks)
    write_jsonl(version_dir / "test_tasks.jsonl", test_tasks)
    write_jsonl(version_dir / "train_trajectories.jsonl", train_trajectories)
    write_jsonl(version_dir / "val_trajectories.jsonl", val_trajectories)

    failed = [t.task.task_id for t in train_trajectories + val_trajectories if not t.success]
    if failed:
        print(f"{len(failed)} trajectories failed: {failed}")

    # Records how this version was built, so training runs and benchmark results can be traced back to it.
    manifest = {
        "version": version_dir.name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "val_size": VAL_SIZE,
        "test_size": TEST_SIZE,
        "split_seed": SPLIT_SEED,
        "counts": {
            "train_tasks": len(train_tasks),
            "val_tasks": len(val_tasks),
            "test_tasks": len(test_tasks),
            "train_trajectories": len(train_trajectories),
            "val_trajectories": len(val_trajectories),
            "failed_trajectories": len(failed),
        },
        "tasks_per_template": {
            split: dict(sorted(Counter(t.template for t in tasks).items()))
            for split, tasks in [("train", train_tasks), ("val", val_tasks), ("test", test_tasks)]
        },
    }
    manifest_path = version_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
