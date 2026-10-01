# toolopt

Teaching a small LLM to use tools. A modular agent framework, a synthetic data pipeline, LoRA fine-tuning of
Qwen2.5-1.5B-Instruct, and a two-stage benchmark harness with an LLM judge.

## Results

450 held-out test tasks (dataset `v1`), base model vs. LoRA fine-tuned:

| | Base Qwen2.5-1.5B-Instruct | LoRA fine-tuned |
|---|---|---|
| Task success | 0% | **98.2%** |
| Tool selection accuracy | 0.34 | **1.00** |
| Argument accuracy | 0.37 | **0.99** |
| Tool calls per task | 9.8 | 3.1 |

The judge scored the fine-tuned model at 93.3%. Of its 30 rejected tasks, 22 had exactly the expected tool calls and
answer and were judge errors (the judge prompt has since been fixed), giving 442/450 = 98.2%. The 8 real failures
were reasoning over tool outputs (comparisons, rounding), misspelled entity names in arguments, and one transcription
error.

Training: LoRA rank 16 on q/k/v/o projections (4.4M trainable parameters, 0.28%), ~10.7K step-level examples from
3.5K tasks, 3 epochs, ~50 minutes on one H100.

Test tasks are unseen, but come from the same templates and phrasings as the training tasks.

## How it works

- **Agent framework** (`app/tools`, `app/environment`, `app/agents`): 26 tools behind a registry and executor. The
  environment runs an agent loop in which a policy picks one tool call per step, as structured JSON, until it calls
  `finish` or reaches 10 steps.
- **Task bank** (`app/data/task_bank.py`): 33 templates generate tasks with random values and 247 phrasings, each
  with its expected answer and expected tool calls.
- **Data generation** (`scripts/generate_data.py`): runs each task's expected actions through the real tools to get
  trajectories, then splits train/val/test, stratified by template and grouped by underlying task so no task
  appears in two splits. Each run writes a new dataset version.
- **Training** (`scripts/train.py`): each trajectory becomes one example per step (task + history → next action),
  and only the action is trained on.
- **Benchmark** (`scripts/benchmark.py`): stage 1 runs the model on test tasks and saves final states (GPU, no
  internet needed). Stage 2 scores them with a Gemini judge plus tool and argument accuracy, cost and latency. Both
  stages resume after interruption.

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
echo "GEMINI_API_KEY=..." > .env    # only needed for benchmark stage 2 (the judge)
```

## Usage

```bash
# 1. Generate a dataset version -> datasets/vN/
python -m scripts.generate_data

# 2. Train a LoRA adapter on the latest version -> experiments/runs/<timestamp>_vN/
python -m scripts.train                       # or --dataset v1

# 3. Benchmark (use the same --limit for every model you compare)
python -m scripts.benchmark --run experiments/runs/<run> --limit 450 --stage run        # model -> final states
python -m scripts.benchmark --run experiments/runs/<run> --limit 450 --stage evaluate   # judge -> scores
python -m scripts.benchmark --limit 450                                                  # base model, both stages
```

## Outputs

```
datasets/vN/
  manifest.json                         how the version was built (split sizes, seed, per-template counts)
  train_tasks.jsonl  val_tasks.jsonl  test_tasks.jsonl
  train_trajectories.jsonl  val_trajectories.jsonl

experiments/runs/<timestamp>_vN/
  run_info.json                         dataset version, config, counts, best checkpoint
  adapter/                              best LoRA adapter (load with PeftModel.from_pretrained)
  checkpoint-*/

experiments/benchmarks/<run name, or base>/<vN>_<n450|all>/
  info.json  states.jsonl  results.jsonl  summary.json
```
