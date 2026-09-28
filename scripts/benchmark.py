"""Benchmarks the base model, or a trained LoRA run, on a dataset version's held-out test tasks.

    python -m scripts.benchmark                                  # base model, latest dataset version
    python -m scripts.benchmark --run experiments/runs/<run>     # trained adapter, on the dataset it was trained on
    python -m scripts.benchmark --run ... --dataset v3 --limit 50
"""
import argparse
import asyncio
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from app.agents.agent import Agent
from app.agents.local_policy import LocalPolicy
from app.data.labeled_task import LabeledTask
from app.data.versions import resolve_version_dir
from app.environment.environment import Environment
from app.evaluation.benchmark import run_benchmark, summarize
from app.tools.executor import ToolExecutor
from app.tools.registry_builder import build_registry
from app.training.config import LoRATrainingConfig
from app.training.lora import load_tokenizer

BASE_RESULTS_DIR = Path("experiments/benchmarks")


def load_test_tasks(dataset_dir: Path) -> list[LabeledTask]:
    with (dataset_dir / "test_tasks.jsonl").open() as f:
        return [LabeledTask.model_validate_json(line) for line in f if line.strip()]


async def main():
    parser = argparse.ArgumentParser(description="Benchmark the base model or a trained run on held-out test tasks.")
    parser.add_argument("--run", type=Path, help="Training run directory (experiments/runs/<run>). Omit to benchmark the base model.")
    parser.add_argument("--dataset", help="Dataset version to test on, e.g. v2 (default: the run's dataset, or the latest)")
    parser.add_argument("--limit", type=int, help="Only run a fixed random sample of N test tasks (to save judge calls)")
    args = parser.parse_args()

    run_info = json.loads((args.run / "run_info.json").read_text()) if args.run else None
    dataset_dir = resolve_version_dir(args.dataset or (run_info["dataset_version"] if run_info else None))
    if run_info and dataset_dir.name != run_info["dataset_version"]:
        print(f"WARNING: run was trained on {run_info['dataset_version']} but is being tested on {dataset_dir.name}; "
              "the test set may overlap its training data.")

    config = LoRATrainingConfig()
    model = AutoModelForCausalLM.from_pretrained(config.base_model, torch_dtype=torch.bfloat16, device_map="auto")
    if args.run:
        model = PeftModel.from_pretrained(model, str(args.run / "adapter"))
    model.eval()
    tokenizer = load_tokenizer(config=config)

    agent = Agent(Environment(build_registry(), ToolExecutor()), LocalPolicy(model, tokenizer))

    labeled_tasks = load_test_tasks(dataset_dir)
    if args.limit is not None and args.limit < len(labeled_tasks):
        # The test file is ordered by template, so sample instead of slicing. The fixed seed means every
        # model benchmarked with the same --limit sees the same tasks.
        labeled_tasks = random.Random(0).sample(labeled_tasks, args.limit)
    model_name = f"{config.base_model} + {args.run}" if args.run else config.base_model
    print(f"Running {len(labeled_tasks)} {dataset_dir.name} test tasks against {model_name}...")

    results = await run_benchmark(agent, labeled_tasks)
    summary = summarize(results)
    print(json.dumps(summary, indent=2))

    template_of = {t.task.task_id: t.template for t in labeled_tasks}
    by_template = defaultdict(list)
    for result in results:
        by_template[template_of[result.task_id]].append(result)

    output_path = (
        args.run / f"benchmark_{dataset_dir.name}.json" if args.run
        else BASE_RESULTS_DIR / f"base_{dataset_dir.name}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump({
            "model": config.base_model,
            "run": str(args.run) if args.run else None,
            "dataset_version": dataset_dir.name,
            "num_tasks": len(labeled_tasks),
            "limit": args.limit,
            "created": datetime.now().isoformat(timespec="seconds"),
            "summary": summary,
            "summary_by_template": {k: summarize(v) for k, v in sorted(by_template.items())},
            "results": [{"template": template_of[r.task_id], **r.model_dump()} for r in results],
        }, f, indent=2)
    print(f"Saved full results to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
