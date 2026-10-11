# toolopt

Teaching a small LLM to use tools, and to use them cost-effectively. A modular agent framework with priced tools, a
synthetic data pipeline, LoRA fine-tuning of Qwen2.5-1.5B-Instruct (SFT, then DPO toward cheaper tool choices), and a
two-stage benchmark harness with an LLM judge.

## Results

First experiment: SFT on the original single-path dataset (26 tools, one valid path per task, no cost
alternatives). 450 held-out test tasks, base model vs. LoRA fine-tuned:

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

- **Agent framework** (`app/tools`, `app/environment`, `app/agents`): 35 tools behind a registry and executor. Each
  tool documents its price (`ToolPricing`: a base price, optionally plus a price per unit of one argument); the
  model sees prices in its prompt and `Tool.cost()` charges exactly that. The environment runs an agent loop in which
  a policy picks one tool call per step, as structured JSON, until it calls `finish` or reaches 10 steps.
- **Cost families**: 10 pairs of overlapping tools where the cheaper choice depends on the task (see below).
- **Task bank** (`app/data/task_bank.py`): 50 templates generate tasks with random values and phrasings. Each task
  lists every valid path (the cheapest is `expected_actions`) and plausible failing cheap attempts.
- **Data generation** (`scripts/generate_data.py`): validates every path, splits tasks four ways, and writes SFT
  demonstrations and scripted DPO preference pairs. See [Dataset generation](#dataset-generation).
- **Training** (`scripts/train.py`): each trajectory becomes one example per step (task + history → next action),
  and only the action is trained on. `--demos random` (default) trains the cost-unaware SFT model; `--demos cheapest`
  trains the reference model on optimal paths.
- **Benchmark** (`scripts/benchmark.py`): stage 1 runs the model on test tasks and saves final states (GPU, no
  internet needed). Stage 2 scores them with a Gemini judge plus tool and argument accuracy, cost and latency. Both
  stages resume after interruption.

### Cost families

| Family | Cheaper tool | Pricier tool | Cheaper choice depends on |
|---|---|---|---|
| Calculator | `calculator` 0.01 | `scientific_calculator` 0.03 | Function needed (arithmetic and `** 0.5` vs trig/log/exp) |
| Weather | `weather` 0.02 per call | `weather_batch` 0.01 + 0.012 per city | Number of cities |
| Stocks | `stock_price` 0.02 per call | `stock_prices` 0.01 + 0.012 per ticker | Number of tickers |
| Statistics | `statistics` 0.01 per stat | `describe_numbers` 0.015 | Number of statistics |
| Primes | `small_prime_checker` 0.004 (≤ 9,999) | `prime_checker` 0.01 | Size of the number |
| Conversion | `currency_converter` 0.004 (currencies) | `converter` 0.01 | The units |
| Distance | `distance_between` 0.02 (100-pair table) | `geo_distance` 0.05 (any pair) | Whether the pair is in the table |
| Files | `file_write` 0.02 (+ `file_read` 0.01) | `file_write_verified` 0.025 | Whether the task asks to confirm |
| Calendar | `calendar_create` 0.02 (+ `calendar_list` 0.01) | `calendar_create_verified` 0.025 | Whether the task asks to confirm |
| Email | `send_short_email` 0.02 (body ≤ 100 chars) | `send_email` 0.05 | Body length |

## Dataset generation

Everything comes from the task bank. `◆` marks a branching point. Numbers are from a full `generate_data` run.

```
35 priced tools · 10 overlapping cost families
│
build_task_bank()
│   50 templates × random values × phrasings, seeded
│     8 single-tool with a cost alternative
│     13 single-path (no alternative)
│     6 multi-lookup · 19 chain / multi-family
│     4 mixes (2-5 unrelated lookups)
│   crossover values balanced:
│     small/large primes, table/non-table city pairs,
│     short/long emails, one/several cities or tickers
│
◆ BRANCH A ─ per task: every way to solve it
│            (_labeled; run + checked by generate_data)
│   ├── valid paths     18,328
│   │     sorted by real cost; cheapest = expected_actions,
│   │     rest = alternative_paths
│   └── failing paths    2,120
│         a cheap tool outside its coverage,
│         swapped into the cheapest path
│   checks: valid paths succeed with the expected answer,
│           failing paths fail, cheapest comes first
│
│   = 7,680 tasks
│     5,802 (76%) with >1 valid path · 1,728 with a failing path
│
◆ BRANCH B ─ grouped_split: which split does each task go to?
│            grouped by underlying task (all wordings together),
│            stratified by template, each template seeded separately;
│            single-path templates get no dpo_train share
│
├── sft_train ── 3,757 tasks (49%)
│   │   → sft_train_tasks.jsonl
│   │
│   ◆ BRANCH C ─ which valid path is demonstrated?
│   ├── one random valid path (seeded per task)
│   │     → sft_train_trajectories.jsonl
│   │       11,182 step examples · 59% optimal · ~14% excess cost
│   │     → SFT, cost-unaware (run A)
│   └── the cheapest path
│         → sft_train_cheapest_trajectories.jsonl (100% optimal)
│         → SFT reference (run C)
│
├── dpo_train ── 1,996 tasks (26%)
│   │   → dpo_train_tasks.jsonl
│   │
│   ◆ BRANCH D ─ what is the cheapest path compared against?
│   │            (at the first step where they differ)
│   ├── a pricier valid path (≥ 0.001 more)
│   │     → cost pairs         2,234
│   ├── a failing cheap attempt
│   │     → correctness pairs    586
│   └── filters: email family held out, max 3 pairs per task
│         → dpo_pairs.jsonl (2,820 pairs)
│         → DPO on run A (run B)
│
├── val ── 386 tasks (5%)
│   │   → val_tasks.jsonl
│   │
│   ◆ BRANCH E ─ what is each validation set used for?
│   ├── one random valid path
│   │     → val_trajectories.jsonl (1,190 step examples)
│   │     → validation loss, run A
│   ├── the cheapest path
│   │     → val_cheapest_trajectories.jsonl
│   │     → validation loss, run C
│   └── pairs (same rules as branch D)
│         → dpo_val_pairs.jsonl
│           468 pairs: 375 cost, 93 correctness
│         → DPO validation
│
└── test ── 1,541 tasks (20%)
        → test_tasks.jsonl
        no trajectories or pairs; all paths kept,
        so each task's optimal cost is known
        → benchmark for runs A, B and C
```

Guarantees:
- **No leakage**: no task, description, or underlying task appears in two splits.
- **Reproducible**: everything is seeded per task or per template with stable hashes; reruns are byte-identical
  (apart from measured latency).
- **Correct by construction**: every demonstration and pair is a path that was run and checked, and every answer is
  consistent with the tool outputs on every valid path.

Settings (top of `scripts/generate_data.py`, plus `_BASE_COUNTS` in the task bank):

| Branch | Decides | Setting |
|---|---|---|
| A | Valid and failing paths of each task | The cost-family helpers in `task_bank.py`; `MAX_VALID_PATHS = 8` |
| B | Which split each task goes to | `SPLIT_FRACTIONS` (test 0.20, val 0.05, dpo_train 0.30, rest sft_train) |
| C | Which path SFT demonstrates | `SFT_P_OPTIMAL` (None = uniform over valid paths) |
| D | Cost vs correctness pairs, and which are kept | `MIN_COST_GAP = 0.001`, `MAX_PAIRS_PER_TASK = 3`, `DPO_HELDOUT_TOOLS` |
| E | How validation tasks are reused | Fixed: one validation set per model kind |

DPO pair balance (chosen tool over rejected tool):

| Family | Cheaper wins | Pricier wins |
|---|---|---|
| Weather (single vs batch) | 148 | 225 |
| Stocks (single vs batch) | 140 | 258 |
| Statistics (single vs describe) | 279 | 165 |
| Files (plain vs verified) | 66 | 61 |
| Calendar (plain vs verified) | 66 | 60 |
| Calculator vs scientific | 249 | 141 (correctness) |
| Small vs regular prime | 254 | 121 (correctness) |
| Currency vs general converter | 144 | 131 (correctness) |
| Distance table vs geo | 144 | 171 (correctness) |

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
python -m scripts.train                       # cost-unaware SFT (random-path demos); add --dataset v1 to pick a version
python -m scripts.train --demos cheapest      # reference SFT on cheapest-path demos

# 3. Benchmark (use the same --limit for every model you compare)
python -m scripts.benchmark --run experiments/runs/<run> --limit 450 --stage run        # model -> final states
python -m scripts.benchmark --run experiments/runs/<run> --limit 450 --stage evaluate   # judge -> scores
python -m scripts.benchmark --limit 450                                                  # base model, both stages
```

DPO training on `dpo_pairs.jsonl` is the next step and doesn't have a script yet.

## Outputs

```
datasets/vN/                            ~174 MB
  manifest.json                         how the version was built (settings, split sizes, pair counts, per-template counts)
  sft_train_tasks.jsonl  dpo_train_tasks.jsonl  val_tasks.jsonl  test_tasks.jsonl
  sft_train_trajectories.jsonl          random-path SFT demos
  sft_train_cheapest_trajectories.jsonl cheapest-path SFT demos (reference model)
  val_trajectories.jsonl  val_cheapest_trajectories.jsonl
  dpo_pairs.jsonl  dpo_val_pairs.jsonl  scripted DPO preference pairs

experiments/runs/<timestamp>_vN[_cheapest]/
  run_info.json                         dataset version, demos used, config, counts, best checkpoint
  adapter/                              best LoRA adapter (load with PeftModel.from_pretrained)
  checkpoint-*/

experiments/benchmarks/<run name, or base>/<vN>_<n450|all>/
  info.json  states.jsonl  results.jsonl  summary.json
```
