"""Builds a new dataset version from the task bank: a grouped 4-way task split, SFT demonstrations, and scripted DPO pairs.

    python -m scripts.generate_data

Writes datasets/vN/:
  sft_train_tasks / dpo_train_tasks / val_tasks / test_tasks .jsonl
  sft_train_trajectories.jsonl, val_trajectories.jsonl                    one random valid path per task (cost-unaware)
  sft_train_cheapest_trajectories.jsonl, val_cheapest_trajectories.jsonl  the cheapest path per task (reference model)
  dpo_pairs.jsonl (from dpo_train tasks), dpo_val_pairs.jsonl (from val tasks)
  manifest.json
"""
import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from app.data.generator import run_path
from app.data.labeled_task import LabeledTask
from app.data.preference import PreferencePair
from app.data.task_bank import build_task_bank
from app.data.trajectory import Trajectory
from app.data.versions import DATASET_ROOT, next_version_dir
from app.environment.environment import Environment
from app.environment.state import State
from app.tools.executor import ToolExecutor
from app.tools.registry_builder import build_registry

SEED = 42

# Fractions of underlying tasks per split (counted in groups, per template); the rest is sft_train.
SPLIT_FRACTIONS = {"test": 0.20, "val": 0.05, "dpo_train": 0.30}

# SFT demonstrations: None picks uniformly among a task's valid paths. A number p picks the cheapest path with
# probability p and otherwise a random pricier one, which sets how cost-wasteful the SFT baseline is.
SFT_P_OPTIMAL: float | None = None

# DPO pairs.
MIN_COST_GAP = 0.001             # a pricier valid path must cost at least this much more to make a cost pair
MAX_PAIRS_PER_TASK = 3           # so tasks with many paths don't dominate
DPO_HELDOUT_TOOLS = frozenset({"send_email", "send_short_email"})  # one cost family kept out of DPO to test transfer

_COST_TOL = 1e-9


def seeded_rng(*keys: str) -> random.Random:
    """A random generator seeded from SEED and the given keys (stable across runs, unlike Python's hash())."""
    digest = hashlib.sha256(":".join([str(SEED), *keys]).encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def task_rng(task_id: str, purpose: str) -> random.Random:
    """Seeded from the task itself, so each task's choices don't depend on any other task."""
    return seeded_rng(purpose, task_id)


def group_key(task: LabeledTask) -> str:
    """Identifies the underlying task by its cheapest path's tool calls, ignoring how the description is worded."""
    return json.dumps([action.model_dump() for action in task.expected_actions], sort_keys=True)


def grouped_split(
    tasks: list[LabeledTask], fractions: dict[str, float], no_dpo_templates: frozenset[str] = frozenset()
) -> dict[str, list[LabeledTask]]:
    """Splits tasks so every wording of the same underlying task lands in the same split.

    Groups are split within each template, keeping every split stratified by template. Fractions are approximate,
    since they're counted in groups rather than tasks; whatever isn't assigned goes to "sft_train". Each template is
    shuffled with its own seed, so adding or removing a template never changes how the others are split.
    Templates in no_dpo_templates get no dpo_train share (it goes to sft_train instead).
    """
    by_template: dict[str, dict[str, list[LabeledTask]]] = defaultdict(lambda: defaultdict(list))
    for task in tasks:
        by_template[task.template][group_key(task)].append(task)

    splits: dict[str, list[LabeledTask]] = {name: [] for name in [*fractions, "sft_train"]}
    for template in sorted(by_template):
        groups = by_template[template]
        keys = sorted(groups)
        seeded_rng("split", template).shuffle(keys)
        start = 0
        for name, fraction in fractions.items():
            if name == "dpo_train" and template in no_dpo_templates:
                continue
            count = round(len(keys) * fraction)
            for key in keys[start:start + count]:
                splits[name].extend(groups[key])
            start += count
        for key in keys[start:]:
            splits["sft_train"].extend(groups[key])
    return splits


def run_all_paths(environment: Environment, task: LabeledTask) -> tuple[list[Trajectory], list[Trajectory]]:
    """Runs every valid and invalid path of a task and checks the template got them right.

    Valid paths must succeed with the expected answer, invalid paths must fail, and expected_actions must be the
    cheapest valid path. Returns (valid trajectories with the cheapest first, invalid trajectories).
    """
    valid = [run_path(environment, task.task, path) for path in task.valid_paths]
    invalid = [run_path(environment, task.task, path) for path in task.invalid_paths]
    for i, trajectory in enumerate(valid):
        answer = trajectory.steps[-1].action.args.get("answer")
        if not trajectory.success or answer != task.expected_answer:
            raise ValueError(f"{task.task.task_id}: valid path {i} doesn't succeed with the expected answer "
                             f"(success={trajectory.success}, answer={answer!r}, expected={task.expected_answer!r})")
    for i, trajectory in enumerate(invalid):
        if trajectory.success:
            raise ValueError(f"{task.task.task_id}: invalid path {i} unexpectedly succeeds")
    cheapest = min(t.total_cost for t in valid)
    if valid[0].total_cost > cheapest + _COST_TOL:
        raise ValueError(f"{task.task.task_id}: expected_actions costs {valid[0].total_cost} but a valid path costs {cheapest}")
    return valid, invalid


def choose_sft_path(task: LabeledTask, valid: list[Trajectory]) -> Trajectory:
    rng = task_rng(task.task.task_id, "sft")
    if SFT_P_OPTIMAL is None or len(valid) == 1:
        return rng.choice(valid)
    pricier = [t for t in valid[1:] if t.total_cost > valid[0].total_cost + _COST_TOL]
    if not pricier or rng.random() < SFT_P_OPTIMAL:
        return valid[0]
    return rng.choice(pricier)


def first_divergence(a: Trajectory, b: Trajectory) -> int | None:
    for i, (step_a, step_b) in enumerate(zip(a.steps, b.steps)):
        if step_a.action != step_b.action:
            return i
    return None


def build_pairs(task: LabeledTask, valid: list[Trajectory], invalid: list[Trajectory]) -> list[PreferencePair]:
    """Pairs the cheapest valid path against each pricier valid path (cost pairs) and each invalid path (correctness
    pairs), at the first step where they differ. Keeps at most MAX_PAIRS_PER_TASK, sampled with the task's own seed."""
    base = valid[0]
    candidates: dict[tuple[int, str], PreferencePair] = {}
    for other, kind in [*((t, "cost") for t in valid[1:]), *((t, "correctness") for t in invalid)]:
        if kind == "cost" and other.total_cost - base.total_cost < MIN_COST_GAP - _COST_TOL:
            continue
        i = first_divergence(base, other)
        if i is None:
            continue
        chosen, rejected = base.steps[i].action, other.steps[i].action
        if chosen.tool_name in DPO_HELDOUT_TOOLS or rejected.tool_name in DPO_HELDOUT_TOOLS:
            continue
        key = (i, rejected.model_dump_json())
        candidates.setdefault(key, PreferencePair(
            task_id=task.task.task_id,
            template=task.template,
            kind=kind,
            state=State(task=base.task, available_tools=base.available_tools, history=base.steps[:i]),
            chosen=chosen,
            rejected=rejected,
            chosen_path_cost=base.total_cost,
            rejected_path_cost=other.total_cost,
        ))
    pairs = [candidates[key] for key in sorted(candidates)]
    if len(pairs) > MAX_PAIRS_PER_TASK:
        pairs = task_rng(task.task.task_id, "dpo").sample(pairs, MAX_PAIRS_PER_TASK)
    return pairs


def write_jsonl(path: Path, records: list[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for record in records:
            f.write(record.model_dump_json() + "\n")
    print(f"Wrote {len(records)} records to {path}")


def main():
    version_dir = next_version_dir(DATASET_ROOT)
    environment = Environment(build_registry(), ToolExecutor())

    tasks = build_task_bank()
    runs = {t.task.task_id: run_all_paths(environment, t) for t in tasks}
    # Templates whose tasks all have a single valid path and no failing attempt can never produce a DPO pair, so they
    # get no dpo_train share; their tasks still feed SFT, validation, and the benchmark.
    pairable = {t.template for t in tasks if len(t.valid_paths) > 1 or t.invalid_paths}
    no_dpo_templates = frozenset({t.template for t in tasks} - pairable)
    splits = grouped_split(tasks, SPLIT_FRACTIONS, no_dpo_templates)

    # SFT demonstrations: one random valid path per task (the cost-unaware teacher), plus the cheapest path for the
    # reference model.
    demos: dict[str, list[Trajectory]] = {}
    for split in ["sft_train", "val"]:
        demos[split] = [choose_sft_path(t, runs[t.task.task_id][0]) for t in splits[split]]
        demos[f"{split}_cheapest"] = [runs[t.task.task_id][0][0] for t in splits[split]]

    # Scripted DPO pairs: training pairs from the dpo_train pool, validation pairs from the val tasks.
    dpo_pairs = [p for t in splits["dpo_train"] for p in build_pairs(t, *runs[t.task.task_id])]
    dpo_val_pairs = [p for t in splits["val"] for p in build_pairs(t, *runs[t.task.task_id])]

    for split in ["sft_train", "dpo_train", "val", "test"]:
        write_jsonl(version_dir / f"{split}_tasks.jsonl", splits[split])
    write_jsonl(version_dir / "sft_train_trajectories.jsonl", demos["sft_train"])
    write_jsonl(version_dir / "val_trajectories.jsonl", demos["val"])
    write_jsonl(version_dir / "sft_train_cheapest_trajectories.jsonl", demos["sft_train_cheapest"])
    write_jsonl(version_dir / "val_cheapest_trajectories.jsonl", demos["val_cheapest"])
    write_jsonl(version_dir / "dpo_pairs.jsonl", dpo_pairs)
    write_jsonl(version_dir / "dpo_val_pairs.jsonl", dpo_val_pairs)

    optimal_cost = {task_id: valid[0].total_cost for task_id, (valid, _) in runs.items()}
    sft = demos["sft_train"]
    sft_excess = [d.total_cost - optimal_cost[d.task.task_id] for d in sft]

    # Records how this version was built, so training runs and benchmark results can be traced back to it.
    manifest = {
        "version": version_dir.name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "split_fractions": SPLIT_FRACTIONS,
        "sft_p_optimal": SFT_P_OPTIMAL,
        "dpo": {"min_cost_gap": MIN_COST_GAP, "max_pairs_per_task": MAX_PAIRS_PER_TASK, "heldout_tools": sorted(DPO_HELDOUT_TOOLS),
                "templates_without_dpo_share": sorted(no_dpo_templates)},
        "counts": {
            **{f"{split}_tasks": len(splits[split]) for split in ["sft_train", "dpo_train", "val", "test"]},
            "tasks_with_alternative_paths": sum(len(t.valid_paths) > 1 for t in tasks),
            "tasks_with_invalid_paths": sum(bool(t.invalid_paths) for t in tasks),
            "dpo_pairs": dict(Counter(p.kind for p in dpo_pairs)),
            "dpo_val_pairs": dict(Counter(p.kind for p in dpo_val_pairs)),
        },
        "sft_demos": {
            "optimal_fraction": sum(e <= _COST_TOL for e in sft_excess) / len(sft) if sft else None,
            "avg_cost": sum(d.total_cost for d in sft) / len(sft) if sft else None,
            "avg_excess_cost": sum(sft_excess) / len(sft) if sft else None,
        },
        "tasks_per_template": {
            split: dict(sorted(Counter(t.template for t in splits[split]).items()))
            for split in ["sft_train", "dpo_train", "val", "test"]
        },
        "dpo_pairs_per_template": {
            template: dict(Counter(p.kind for p in dpo_pairs if p.template == template))
            for template in sorted({p.template for p in dpo_pairs})
        },
    }
    manifest_path = version_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {manifest_path}")
    print(json.dumps({"counts": manifest["counts"], "sft_demos": manifest["sft_demos"]}, indent=2))


if __name__ == "__main__":
    main()
