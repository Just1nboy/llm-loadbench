# llm-loadbench

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.1%2B-EE4C2C)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> A reproducible CLI for benchmarking LLM loading strategies on consumer hardware.

`llm-loadbench` runs five different model-loading strategies — **standard**, **mmap**, **lazy**, **streaming**, **cached** — against any HuggingFace causal LM, repeats each one with warmup + measured iterations, and emits a CSV of timing/memory metrics plus comparison charts and statistical significance tests.

It exists because the question *"is mmap actually faster than `from_pretrained` on a 16 GB laptop?"* doesn't have a simple answer — it depends on disk, page cache state, model format (safetensors vs. bin), and how much RAM you've already given to your IDE. This tool measures it on the actual hardware you care about.

---

## Background

This tool was extracted from a thesis on memory-efficient LLM loading. The notebook prototype validated that meta-device init combined with PyTorch 2.1's `assign=True` flag for `load_state_dict` lets you swap weights into a model without paying the cost of random parameter initialization — which is what makes the `mmap` and `cached` strategies competitive with (or faster than) the default `from_pretrained` baseline.

If you're interested in the underlying technique, the relevant code is in [`llm_loadbench/metrics.py`](llm_loadbench/metrics.py) (`init_model_on_meta`, `load_and_tie`).

---

## Strategies

| Strategy    | What it does                                                                          | When it wins                                          |
|-------------|---------------------------------------------------------------------------------------|-------------------------------------------------------|
| `standard`  | Default `AutoModelForCausalLM.from_pretrained`. The baseline.                         | Reference point.                                      |
| `mmap`      | Meta-device init + `safetensors` mmap (or `torch.load(mmap=True)`).                   | Repeated loads of the same model — OS page cache.     |
| `lazy`      | Init with empty weights, materialize on first forward pass via pre-hook.              | When *load* time matters more than *first-token* time. |
| `streaming` | Per-block weight loading via forward hooks; releases each block after it computes.    | Models too large to hold in RAM at once.              |
| `cached`    | First call uses `from_pretrained`, then caches the state dict in memory for re-use.   | Benchmarks/serving with frequent reloads.             |

All strategies use the same `BaseLoader` interface (`load_model`, `generate`, `cleanup`) so adding a sixth is straightforward — drop a file in [`llm_loadbench/loaders/`](llm_loadbench/loaders/) and register it in [`runner.py`](llm_loadbench/runner.py).

---

## Install

Requires **Python 3.10+** and **PyTorch 2.1+** (the `assign=True` flag on `load_state_dict` is mandatory — earlier versions don't have it).

```bash
git clone https://github.com/Just1nboy/llm-loadbench.git
cd llm-loadbench
pip install -e .
```

This registers `llm-loadbench` as a console script. CUDA is auto-detected; pass `--device cpu` to force CPU.

---

## Quick start

```bash
# Inspect your hardware
llm-loadbench info

# Smoke test (1 strategy, fast)
llm-loadbench run --strategy mmap --iterations 2 --warmup 1

# Full benchmark
llm-loadbench run --strategy all --iterations 10 --warmup 3

# Generate report from latest results
llm-loadbench report --input results/latest.csv
```

---

## Commands

### `run` — execute the benchmark

```bash
llm-loadbench run \
  --strategy all \
  --model EleutherAI/gpt-neo-125m \
  --iterations 10 \
  --warmup 3 \
  --prompt "The future of AI is" \
  --max-new-tokens 50
```

| Flag                | Default                                      | Meaning                                              |
|---------------------|----------------------------------------------|------------------------------------------------------|
| `--strategy`        | `all`                                        | Comma-separated list, or `all`. e.g. `mmap,cached`.  |
| `--model`           | `EleutherAI/gpt-neo-125m`                    | Any HuggingFace causal LM repo ID.                   |
| `--iterations`      | `10`                                         | Measured iterations per strategy.                    |
| `--warmup`          | `3`                                          | Warmup iterations (discarded from results).          |
| `--prompt`          | `"The future of artificial intelligence is"` | Prompt fed to `generate()`.                          |
| `--max-new-tokens`  | `50`                                         | Tokens generated per iteration.                      |
| `--output-dir`      | `results`                                    | CSV output directory.                                |
| `--device`          | auto                                         | `cuda` or `cpu`. Auto-detected from `torch.cuda`.    |

Output:
- `results/results_<model>_<YYYYMMDD_HHMMSS>.csv` — timestamped run.
- `results/latest.csv` — copy of the most recent run, used by `report`.

### `report` — summarize a CSV

```bash
llm-loadbench report --input results/latest.csv
```

Prints a summary table (mean ± std per strategy) and an efficiency table (memory/load improvement vs. the `standard` baseline) to the terminal, then writes three PNGs alongside the CSV:

- `comparison_<model>.png` — bar charts for load time, peak memory, TTFT, throughput.
- `efficiency_<model>.png` — heatmap of memory and load-time improvement vs. baseline.
- `iterations_<model>.png` — line plot of load time across iterations (catches caching/page-cache warmup effects).

### `info` — hardware report

```bash
llm-loadbench info
```

Prints Python version, PyTorch version, OS, CPU, RAM, and GPU (if CUDA is available). Useful to record alongside results so you know what hardware produced them.

---

## CSV schema

Each row in the output CSV is one measured iteration. Columns:

| Column               | Type   | Meaning                                                       |
|----------------------|--------|---------------------------------------------------------------|
| `strategy`           | str    | One of `standard`, `mmap`, `lazy`, `streaming`, `cached`.     |
| `model_name`         | str    | HuggingFace repo ID.                                          |
| `load_time_s`        | float  | Wall time of `loader.load_model()` in seconds.                |
| `peak_memory_mb`     | float  | Process RSS peak during load + generate, in MB.               |
| `ttft_s`             | float  | Time to first generated token, in seconds.                    |
| `throughput_tps`     | float  | Tokens-per-second during the full generation.                 |
| `total_inference_s`  | float  | Wall time for the full `max_new_tokens` generation.           |
| `initial_memory_mb`  | float  | Process RSS just before this iteration started.               |
| `memory_delta_mb`    | float  | `peak_memory_mb − initial_memory_mb`.                         |

Mean ± std across iterations (and 95% CI for the summary table) are computed in [`analysis.py`](llm_loadbench/analysis.py) using `scipy.stats`.

---

## Sample output

From a `gpt-neo-125m` run on RTX 4060 Laptop GPU (16 GB RAM, Windows 11, 10 measured iterations):

```
======================================================================
SUMMARY STATISTICS
======================================================================
strategy        model  load_time_s_mean  load_time_s_std  peak_memory_mb_mean  ttft_s_mean  throughput_tps_mean
standard gpt-neo-125m            2.1032           0.0276            2027.0004       0.0215              78.9423
    lazy gpt-neo-125m            1.5129           0.0113            1996.7555       1.1236              77.7182

======================================================================
EFFICIENCY METRICS (relative to baseline)
======================================================================
       model strategy  η_mem (%)  η_load (%)  memory_mb  load_time_s
gpt-neo-125m     lazy       1.49       28.07     1996.8        1.513
gpt-neo-125m standard       0.00        0.00     2027.0        2.103
```

The numbers above are partial (two strategies). What the table shows: `lazy` cuts load time by ~28% but pushes the cost into TTFT (1.12s vs. 0.02s), since real weight loading happens on the first forward pass. That's the kind of trade-off this benchmark surfaces.

---

## Project layout

```
llm-loadbench/
├── llm_loadbench/
│   ├── __init__.py
│   ├── config.py          # ExperimentConfig dataclass
│   ├── metrics.py         # BenchmarkMetrics, MetricsCollector, weight-loading helpers
│   ├── runner.py          # BenchmarkRunner — orchestrates warmup/measured loops
│   ├── analysis.py        # StatisticalAnalyzer + matplotlib charts
│   ├── cli.py             # Click entrypoint (run / report / info)
│   └── loaders/
│       ├── base.py        # BaseLoader (shared generate())
│       ├── standard.py
│       ├── mmap.py
│       ├── lazy.py
│       ├── streaming.py
│       └── cached.py
├── pyproject.toml
├── README.md
├── LICENSE
└── results/               # Gitignored — CSV + PNG outputs land here.
```

---

## Hardware notes

- The default model (`gpt-neo-125m`) is ~500 MB in fp32 and runs comfortably on 8 GB of RAM. Larger models (`gpt-neo-1.3B`, `gpt-neo-2.7B`) are listed in the original notebook but commented out — they need 16 GB+ to benchmark all five strategies in one session, since `cached` keeps a copy of the state dict resident.
- Peak memory is measured via `psutil` process RSS, **not** CUDA memory — so for GPU runs the numbers reflect host-side allocations (state dicts, caches, mmap'd file pages), which is what most strategies are actually trying to optimize.
- If you OOM on `--strategy all`, run subsets: `--strategy standard,mmap,streaming` keeps RAM low; `--strategy cached` last keeps its big resident copy isolated.

---

## License

MIT — see [LICENSE](LICENSE).
