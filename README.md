# llm-loadbench

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.1%2B-EE4C2C)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A CLI tool for benchmarking LLM loading strategies, measuring how long each one takes, how much memory it uses, and how fast it generates text afterwards.

This project was extracted from a thesis on memory-efficient LLM loading. The original prototype was a Jupyter notebook, repackaged here as a reusable command-line tool that runs from the terminal and produces CSV output and comparison charts.

## What it does

Given a HuggingFace model id, the tool runs five loading strategies against it: the default `from_pretrained`, memory-mapped loading via safetensors, lazy loading where weights only materialise on the first forward pass, weight streaming where each transformer block loads and frees during inference, and an in-memory state-dict cache. Each strategy runs for a configurable number of warmup and measured iterations, and the per-iteration metrics are written to a CSV.

The `report` command then reads that CSV and produces a summary table, an efficiency comparison against the baseline, and three PNG charts.

## Why it exists

The default load path in transformers is fine, but it does a couple of things that may be undesirable. It allocates random parameters before overwriting them with real weights, which costs both time and a temporary memory spike. It also does not benefit much from the OS page cache on repeated loads of the same file, because the bytes are copied into a fresh tensor each time.

Some of the strategies in this benchmark sidestep that, by initialising on the meta device first and then using PyTorch 2.1's `assign=True` flag in `load_state_dict` to swap real tensors in without the random-init step. Whether that actually helps in practice depends on disk speed, available RAM, and how warm the page cache is, which is the question the benchmark is designed to answer.

## Strategies

| name | what it does |
|------|--------------|
| standard | Default `AutoModelForCausalLM.from_pretrained`, used as the baseline. |
| mmap | Meta-device init, weights memory-mapped from safetensors or `torch.load(mmap=True)`. |
| lazy | Init with empty weights, real weights load on the first forward pass via a pre-hook. |
| streaming | Per-block weight loading via forward hooks, weights released after each block runs. |
| cached | First call uses `from_pretrained` and caches the state dict, later calls re-hydrate from cache. |

All five share the same `BaseLoader` interface, so adding a sixth is a matter of dropping a file in `llm_loadbench/loaders/` and registering it in the runner.

## Install

Requires Python 3.10 or newer, and PyTorch 2.1 or newer, because `assign=True` did not exist before that release.

```
git clone https://github.com/Just1nboy/llm-loadbench.git
cd llm-loadbench
pip install -e .
```

This puts `llm-loadbench` on the PATH as a console command. CUDA is auto-detected, so `--device cpu` is only needed to force CPU.

## Quick start

```
llm-loadbench info
llm-loadbench run --strategy mmap --iterations 2 --warmup 1
llm-loadbench run --strategy all --iterations 10 --warmup 3
llm-loadbench report --input results/latest.csv
```

`info` prints a hardware report, useful for recording alongside results to track which machine produced them. The two `run` examples are a quick smoke test and a full benchmark, in that order. `report` reads the latest CSV and writes the charts.

## Commands

### run

```
llm-loadbench run \
  --strategy all \
  --model EleutherAI/gpt-neo-125m \
  --iterations 10 \
  --warmup 3 \
  --prompt "The future of AI is" \
  --max-new-tokens 50
```

| flag | default | what it does |
|------|---------|--------------|
| `--strategy` | `all` | Comma-separated list of names, or `all`. |
| `--model` | `EleutherAI/gpt-neo-125m` | Any HuggingFace causal LM repo id. |
| `--iterations` | `10` | Measured iterations per strategy. |
| `--warmup` | `3` | Warmup iterations, discarded from results. |
| `--prompt` | `"The future of artificial intelligence is"` | Prompt fed to `generate`. |
| `--max-new-tokens` | `50` | Tokens generated per iteration. |
| `--output-dir` | `results` | Where the CSVs land. |
| `--device` | auto | `cuda` or `cpu`, auto-detected when omitted. |

Each run produces two files, a timestamped CSV at `results_<model>_<YYYYMMDD_HHMMSS>.csv` and a copy at `latest.csv`, so the report command does not need to know the exact timestamp.

### report

```
llm-loadbench report --input results/latest.csv
```

Prints a summary table with mean and standard deviation per strategy, an efficiency table comparing each strategy against `standard`, and writes three PNGs next to the CSV. The PNGs are bar charts for the four metrics, an efficiency heatmap, and a line plot of load time across iterations, which is the chart that surfaces page-cache warmup effects.

### info

```
llm-loadbench info
```

Prints Python version, PyTorch version, OS, CPU, RAM, and GPU info if available. There are no flags, the output is a flat dump.

## CSV schema

One row per measured iteration. Columns:

| column | type | meaning |
|--------|------|---------|
| strategy | str | One of the five strategy names. |
| model_name | str | The HuggingFace repo id. |
| load_time_s | float | Wall time spent inside `loader.load_model()`. |
| peak_memory_mb | float | Process RSS peak during load and generate, in MB. |
| ttft_s | float | Time to first generated token. |
| throughput_tps | float | Tokens per second across the full generation. |
| total_inference_s | float | Wall time for the full `max_new_tokens` generation. |
| initial_memory_mb | float | Process RSS just before the iteration started. |
| memory_delta_mb | float | `peak_memory_mb - initial_memory_mb`. |

The summary table also reports 95% confidence intervals computed with `scipy.stats`.

## Sample output

A run of standard and lazy on `gpt-neo-125m`, RTX 4060 Laptop, 16 GB RAM, 10 iterations:

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

That is only two strategies, so it is not the full picture, it was just what the original notebook had finished running. The interesting part is the trade-off lazy is making, it cuts load time by 28%, but pushes that cost into TTFT, from 0.02s up to 1.12s. Whether that is a win depends on whether cold start or first-token latency matters more for the use case.

## Project layout

```
llm-loadbench/
├── llm_loadbench/
│   ├── __init__.py
│   ├── config.py
│   ├── metrics.py
│   ├── runner.py
│   ├── analysis.py
│   ├── cli.py
│   └── loaders/
│       ├── base.py
│       ├── standard.py
│       ├── mmap.py
│       ├── lazy.py
│       ├── streaming.py
│       └── cached.py
├── pyproject.toml
├── README.md
├── LICENSE
└── results/
```

`results/` is gitignored, so a fresh clone will not bring along somebody else's CSVs.

## Hardware notes

The default `gpt-neo-125m` is roughly 500 MB in fp32 and runs comfortably on 8 GB of RAM. Larger models like `gpt-neo-1.3B` are listed in the original notebook but commented out, because running all five strategies at once on a 16 GB machine gets tight, especially with `cached` keeping a full state-dict copy resident.

Peak memory is measured via `psutil` process RSS, not CUDA memory, so for GPU runs the numbers reflect host-side allocations like state dicts, caches, and mmap pages, which is what most of these strategies are trying to optimise in the first place.

If `--strategy all` runs out of memory, splitting it helps. Running `--strategy standard,mmap,streaming` first and `--strategy cached` separately keeps the heavy resident copy isolated.

## License

MIT, see [LICENSE](LICENSE).
