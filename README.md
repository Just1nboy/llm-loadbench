# llm-loadbench

A CLI tool for benchmarking LLM loading strategies — extracted from a thesis on memory-efficient model loading. Compares five approaches (standard, mmap, lazy, streaming, cached) against GPT-Neo-class models, measuring load time, peak memory, time-to-first-token, and throughput across repeated iterations with statistical confidence intervals. Originally built to validate whether techniques like memory mapping and weight streaming meaningfully reduce cold-start cost on consumer hardware (RTX 4060, 16 GB RAM).

## Install

Requires Python 3.10+ and PyTorch 2.1+ (the `assign=True` flag on `load_state_dict` is mandatory for meta-device init).

```bash
git clone https://github.com/Just1nboy/llm-loadbench.git
cd llm-loadbench
pip install -e .
```

This registers `llm-loadbench` as a console script. CUDA is auto-detected — pass `--device cpu` to force CPU.

## Usage

### Run a benchmark

Run all five strategies against `gpt-neo-125m` with 10 measured iterations and 3 warmup iterations:

```bash
llm-loadbench run \
  --strategy all \
  --model EleutherAI/gpt-neo-125m \
  --iterations 10 \
  --warmup 3 \
  --prompt "The future of AI is"
```

Run only selected strategies (comma-separated, no spaces):

```bash
llm-loadbench run --strategy mmap,cached
```

Results are written to `./results/results_<model>_<timestamp>.csv` and copied to `./results/latest.csv`.

### Generate a report

Print a summary table to the terminal and save comparison charts:

```bash
llm-loadbench report --input results/latest.csv
```

Charts (`comparison_<model>.png`, `efficiency_<model>.png`, `iterations_<model>.png`) are saved next to the CSV.

### Inspect hardware

Detected RAM, CPU, and GPU before launching a long run:

```bash
llm-loadbench info
```

## Sample output

```
======================================================================
SUMMARY STATISTICS
======================================================================
strategy        model  load_time_s_mean  load_time_s_std  peak_memory_mb_mean  peak_memory_mb_std  ttft_s_mean  ttft_s_std  throughput_tps_mean  throughput_tps_std
standard gpt-neo-125m            2.1032           0.0276            2027.0004             26.9389       0.0215      0.0027              78.9423              5.3129
    lazy gpt-neo-125m            1.5129           0.0113            1996.7555              0.4882       1.1236      0.0186              77.7182              5.0021

======================================================================
EFFICIENCY METRICS (relative to baseline)
======================================================================
       model strategy  η_mem (%)  η_load (%)  memory_mb  load_time_s
gpt-neo-125m     lazy       1.49       28.07     1996.8        1.513
gpt-neo-125m standard       0.00        0.00     2027.0        2.103
```

The CSV contains one row per iteration with columns: `strategy`, `model_name`, `load_time_s`, `peak_memory_mb`, `ttft_s`, `throughput_tps`, `total_inference_s`, `initial_memory_mb`, `memory_delta_mb`. The summary table aggregates these with mean ± std across iterations.

## Strategies

| Strategy    | Idea                                                                                |
|-------------|-------------------------------------------------------------------------------------|
| `standard`  | Default `AutoModelForCausalLM.from_pretrained` — the baseline.                      |
| `mmap`      | Meta-device init + `safetensors`/`torch.load(mmap=True)` for OS-page-cached loads.  |
| `lazy`      | Empty weights at load, real weights pulled in on first forward pass.                |
| `streaming` | Per-block weight loading via forward hooks; releases after each block computes.     |
| `cached`    | First call caches the state dict; subsequent calls re-hydrate via `assign=True`.    |

Both `safetensors` and `pytorch_model.bin` weight formats are supported automatically (`resolve_weight_path` tries safetensors first, falls back to the legacy bin file).

## License

MIT
