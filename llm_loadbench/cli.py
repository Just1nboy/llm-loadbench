import os
import platform
import shutil
from datetime import datetime

import click
import pandas as pd
import psutil
import torch

from .analysis import StatisticalAnalyzer, create_comparison_plots
from .config import ExperimentConfig
from .runner import BenchmarkRunner

ALL_STRATEGIES = ["standard", "mmap", "lazy", "streaming", "cached"]


def _parse_strategies(value: str):
    if value == "all":
        return ALL_STRATEGIES
    requested = [s.strip() for s in value.split(",") if s.strip()]
    unknown = [s for s in requested if s not in ALL_STRATEGIES]
    if unknown:
        raise click.BadParameter(
            f"Unknown strategy: {', '.join(unknown)}. "
            f"Valid: {', '.join(ALL_STRATEGIES)}, all"
        )
    return requested


@click.group()
@click.version_option()
def main():
    """llm-loadbench: benchmark LLM loading strategies."""


@main.command()
@click.option("--strategy", default="all",
              help="Comma-separated strategies, or 'all'. "
                   f"Choices: {', '.join(ALL_STRATEGIES)}")
@click.option("--model", default="EleutherAI/gpt-neo-125m",
              help="HuggingFace model ID.")
@click.option("--iterations", default=10, type=int,
              help="Measured iterations per strategy.")
@click.option("--warmup", default=3, type=int,
              help="Warmup iterations (discarded).")
@click.option("--prompt", default="The future of artificial intelligence is",
              help="Input prompt for inference.")
@click.option("--max-new-tokens", default=50, type=int,
              help="Tokens to generate per iteration.")
@click.option("--output-dir", default="results",
              help="Directory for CSV results.")
@click.option("--device", default=None,
              help="cuda or cpu. Auto-detected if omitted.")
def run(strategy, model, iterations, warmup, prompt, max_new_tokens,
        output_dir, device):
    """Run benchmark suite and save results CSV."""
    strategies = _parse_strategies(strategy)
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    config = ExperimentConfig(
        models=[model],
        warmup_iterations=warmup,
        measured_iterations=iterations,
        input_prompt=prompt,
        max_new_tokens=max_new_tokens,
        device=resolved_device,
        output_dir=output_dir,
    )

    runner = BenchmarkRunner(config)
    df = runner.run_all(strategies=strategies)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_model = model.split("/")[-1]
    out_path = os.path.join(output_dir, f"results_{short_model}_{timestamp}.csv")
    latest_path = os.path.join(output_dir, "latest.csv")

    df.to_csv(out_path, index=False)
    shutil.copyfile(out_path, latest_path)

    click.echo(f"\nResults saved:")
    click.echo(f"  {out_path}")
    click.echo(f"  {latest_path}")


@main.command()
@click.option("--input", "input_path", default="results/latest.csv",
              help="Path to results CSV.")
@click.option("--output-dir", default=None,
              help="Directory for chart PNGs (defaults to CSV's directory).")
def report(input_path, output_dir):
    """Print summary table and save comparison charts."""
    if not os.path.exists(input_path):
        raise click.ClickException(f"Results file not found: {input_path}")

    df = pd.read_csv(input_path)
    if df.empty:
        raise click.ClickException(f"Results file is empty: {input_path}")

    out_dir = output_dir or os.path.dirname(os.path.abspath(input_path))
    os.makedirs(out_dir, exist_ok=True)

    analyzer = StatisticalAnalyzer(df)

    click.echo("=" * 70)
    click.echo("SUMMARY STATISTICS")
    click.echo("=" * 70)
    summary = analyzer.summary_table()
    display_cols = ["strategy", "model"]
    for m in ["load_time_s", "peak_memory_mb", "ttft_s", "throughput_tps"]:
        for suffix in ("_mean", "_std"):
            col = m + suffix
            if col in summary.columns:
                display_cols.append(col)
    with pd.option_context("display.max_columns", None,
                           "display.width", 200,
                           "display.float_format", lambda x: f"{x:.4f}"):
        click.echo(summary[display_cols].to_string(index=False))

    click.echo("\n" + "=" * 70)
    click.echo("EFFICIENCY METRICS (relative to baseline)")
    click.echo("=" * 70)
    eff = analyzer.efficiency_metrics()
    if eff.empty:
        click.echo("No 'standard' baseline runs found — skipping efficiency metrics.")
    else:
        click.echo(eff.to_string(index=False))

    click.echo("\nGenerating charts...")
    paths = create_comparison_plots(df, out_dir)
    for p in paths:
        click.echo(f"  saved {p}")


@main.command()
def info():
    """Print detected hardware (RAM, CPU, GPU)."""
    report = {
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "os": f"{platform.system()} {platform.release()}",
        "cpu": platform.processor() or "Unknown",
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "ram_available_gb": round(psutil.virtual_memory().available / (1024 ** 3), 2),
    }
    if torch.cuda.is_available():
        report["gpu"] = torch.cuda.get_device_name(0)
        report["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2
        )
        report["cuda_version"] = torch.version.cuda
    else:
        report["gpu"] = "None (CPU only)"

    if os.path.exists("/content"):
        report["environment"] = "Google Colab"
    elif os.path.exists("/kaggle"):
        report["environment"] = "Kaggle"
    else:
        report["environment"] = "Local"

    click.echo("Hardware Report")
    click.echo("-" * 40)
    for k, v in report.items():
        click.echo(f"  {k}: {v}")


if __name__ == "__main__":
    main()
