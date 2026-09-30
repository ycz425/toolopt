"""Benchmarks the base model, or a trained LoRA run, on a dataset version's held-out test tasks.

Two stages, so the model and the judge can run on different machines and each can resume after an interruption:
  run       runs the model on each task and saves its final state   (needs the model; no internet)
  evaluate  scores the saved states, calling the Gemini judge       (needs internet; no model)
  all       both, one after the other (default)

    python -m scripts.benchmark                                          # base model, latest dataset, both stages
    python -m scripts.benchmark --run experiments/runs/<run> --limit 450  # trained adapter, on its own dataset
    python -m scripts.benchmark --run experiments/runs/<run> --limit 450 --stage evaluate

Outputs go to experiments/benchmarks/<run name, or "base">/<dataset>_<n450|all>/:
  info.json        what was benchmarked
  states.jsonl     stage 1: one final state per task
  results.jsonl    stage 2: one BenchmarkResult per task
  summary.json     stage 2: overall and per-template summary
"""
import argparse
import asyncio
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from app.agents.agent import Agent
from app.data.labeled_task import LabeledTask
from app.data.versions import resolve_version_dir
from app.environment.environment import Environment
from app.evaluation.benchmark import evaluate_benchmark, run_benchmark, summarize
from app.tools.executor import ToolExecutor
from app.tools.registry_builder import build_registry
from app.training.config import LoRATrainingConfig

BENCHMARKS_ROOT = Path("experiments/benchmarks")


def load_test_tasks(dataset_dir: Path, limit: int | None) -> list[LabeledTask]:
    with (dataset_dir / "test_tasks.jsonl").open() as f:
        labeled_tasks = [LabeledTask.model_validate_json(line) for line in f if line.strip()]
    if limit is not None and limit < len(labeled_tasks):
        # The test file is ordered by template, so sample instead of slicing. The fixed seed means every
        # model benchmarked with the same --limit sees the same tasks, in both stages.
        labeled_tasks = random.Random(0).sample(labeled_tasks, limit)
    return labeled_tasks


def build_agent(base_model: str, run: Path | None) -> Agent:
    # Imported here so the evaluate stage doesn't need torch or the model.
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    from app.agents.local_policy import LocalPolicy
    from app.training.lora import load_tokenizer

    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16, device_map="auto")
    if run:
        model = PeftModel.from_pretrained(model, str(run / "adapter"))
    model.eval()
    tokenizer = load_tokenizer(config=LoRATrainingConfig())
    return Agent(Environment(build_registry(), ToolExecutor()), LocalPolicy(model, tokenizer))


async def main():
    parser = argparse.ArgumentParser(description="Benchmark the base model or a trained run on held-out test tasks.")
    parser.add_argument("--stage", choices=["run", "evaluate", "all"], default="all",
                        help="run: model -> saved final states; evaluate: saved states -> scores; all: both (default)")
    parser.add_argument("--run", type=Path, help="Training run directory (experiments/runs/<run>). Omit to benchmark the base model.")
    parser.add_argument("--dataset", help="Dataset version to test on, e.g. v2 (default: the run's dataset, or the latest)")
    parser.add_argument("--limit", type=int, help="Only use a fixed random sample of N test tasks (to save time and judge calls)")
    args = parser.parse_args()

    run_info = json.loads((args.run / "run_info.json").read_text()) if args.run else None
    dataset_dir = resolve_version_dir(args.dataset or (run_info["dataset_version"] if run_info else None))
    if run_info and dataset_dir.name != run_info["dataset_version"]:
        print(f"WARNING: run was trained on {run_info['dataset_version']} but is being tested on {dataset_dir.name}; "
              "the test set may overlap its training data.")

    labeled_tasks = load_test_tasks(dataset_dir, args.limit)
    base_model = LoRATrainingConfig().base_model
    # experiments/benchmarks/<run name, or "base">/<dataset>_<n450|all>/
    out_dir = BENCHMARKS_ROOT / (args.run.name if args.run else "base") / f"{dataset_dir.name}_{f'n{args.limit}' if args.limit else 'all'}"
    states_path, results_path = out_dir / "states.jsonl", out_dir / "results.jsonl"

    if args.stage in ("run", "all"):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "info.json").write_text(json.dumps({
            "model": base_model,
            "run": str(args.run) if args.run else None,
            "dataset_version": dataset_dir.name,
            "limit": args.limit,
            "num_tasks": len(labeled_tasks),
            "created": datetime.now().isoformat(timespec="seconds"),
        }, indent=2) + "\n")
        model_name = f"{base_model} + {args.run}" if args.run else base_model
        print(f"Stage 1: running {len(labeled_tasks)} {dataset_dir.name} test tasks against {model_name}...")
        await run_benchmark(build_agent(base_model, args.run), labeled_tasks, states_path)
        print(f"Saved final states to {states_path}")

    if args.stage in ("evaluate", "all"):
        if not states_path.exists():
            raise FileNotFoundError(f"{states_path} not found; run --stage run first with the same --run/--dataset/--limit")
        print(f"Stage 2: evaluating {states_path}...")
        results = await evaluate_benchmark(labeled_tasks, states_path, results_path)

        template_of = {t.task.task_id: t.template for t in labeled_tasks}
        by_template = defaultdict(list)
        for result in results:
            by_template[template_of[result.task_id]].append(result)
        summary = {
            "summary": summarize(results),
            "summary_by_template": {k: summarize(v) for k, v in sorted(by_template.items())},
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary["summary"], indent=2))
        print(f"Saved results to {results_path} and summary to {out_dir / 'summary.json'}")


if __name__ == "__main__":
    asyncio.run(main())
