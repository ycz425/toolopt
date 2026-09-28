import asyncio
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from app.agents.agent import Agent
from app.agents.local_policy import LocalPolicy
from app.data.task_bank import build_task_bank
from app.environment.environment import Environment
from app.evaluation.benchmark import run_benchmark, summarize
from app.tools.executor import ToolExecutor
from app.tools.registry_builder import build_registry
from app.training.config import LoRATrainingConfig
from app.training.lora import load_tokenizer

OUTPUT_PATH = Path("experiments/benchmarks/zero_shot_qwen.json")


async def main():
    config = LoRATrainingConfig()

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model, torch_dtype=torch.bfloat16, device_map="auto",
    )
    tokenizer = load_tokenizer(config=config)

    registry = build_registry()
    executor = ToolExecutor()
    environment = Environment(registry, executor)
    agent = Agent(environment, LocalPolicy(model, tokenizer))

    cases = build_task_bank()
    print(f"Running {len(cases)} benchmark cases against zero-shot {config.base_model}...")

    results = await run_benchmark(agent, cases)
    summary = summarize(results)

    print(json.dumps(summary, indent=2))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w") as f:
        json.dump({
            "summary": summary,
            "results": [r.model_dump() for r in results],
        }, f, indent=2)
    print(f"Saved full results to {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
