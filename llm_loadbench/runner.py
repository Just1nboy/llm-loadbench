import time
from dataclasses import asdict
from typing import List

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .config import ExperimentConfig
from .loaders import (
    CachedLoader,
    LazyLoader,
    MmapLoader,
    StandardLoader,
    StreamingLoader,
)
from .metrics import BenchmarkMetrics, MetricsCollector


class BenchmarkRunner:
    LOADER_CLASSES = {
        "standard": StandardLoader,
        "mmap": MmapLoader,
        "lazy": LazyLoader,
        "streaming": StreamingLoader,
        "cached": CachedLoader,
    }

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.collector = MetricsCollector()
        self.all_results: List[BenchmarkMetrics] = []

    def run_single_experiment(self, strategy_name, model_name):
        loader_cls = self.LOADER_CLASSES[strategy_name]
        loader = loader_cls(model_name, self.config.device)

        initial_mem = self.collector.get_memory_mb()
        self.collector.start_tracking()

        _, _, load_time = loader.load_model()
        self.collector.update_peak()

        output_text, ttft, throughput, total_inf = loader.generate(
            self.config.input_prompt, self.config.max_new_tokens,
        )
        self.collector.update_peak()
        peak_mem = self.collector.get_peak_memory_mb()

        metrics = BenchmarkMetrics(
            strategy=strategy_name, model_name=model_name,
            load_time_s=load_time, peak_memory_mb=peak_mem,
            ttft_s=ttft, throughput_tps=throughput,
            total_inference_s=total_inf, initial_memory_mb=initial_mem,
            memory_delta_mb=peak_mem - initial_mem,
        )
        loader.cleanup()
        return metrics

    def run_strategy_benchmark(self, strategy_name, model_name):
        short_model = model_name.split('/')[-1]
        print(f"\n{'='*60}")
        print(f"  Strategy: {strategy_name} | Model: {short_model}")
        print(f"{'='*60}")

        print(f"  Running {self.config.warmup_iterations} warmup iterations...")
        for i in range(self.config.warmup_iterations):
            try:
                _ = self.run_single_experiment(strategy_name, model_name)
                print(f"    Warmup {i+1} complete")
            except Exception as e:
                print(f"    Warmup {i+1} failed: {e}")
            MetricsCollector.clear_state()

        results = []
        print(f"  Running {self.config.measured_iterations} measured iterations...")
        for i in tqdm(range(self.config.measured_iterations), desc=f"  {strategy_name}"):
            try:
                metrics = self.run_single_experiment(strategy_name, model_name)
                results.append(metrics)
            except Exception as e:
                print(f"    Iteration {i+1} failed: {e}")
            MetricsCollector.clear_state()

        if results:
            avg_load = np.mean([r.load_time_s for r in results])
            avg_mem = np.mean([r.peak_memory_mb for r in results])
            avg_ttft = np.mean([r.ttft_s for r in results])
            avg_tps = np.mean([r.throughput_tps for r in results])
            print(f"  Results: Load={avg_load:.2f}s | Mem={avg_mem:.0f}MB | "
                  f"TTFT={avg_ttft:.3f}s | TPS={avg_tps:.1f}")
        else:
            print(f"  WARNING: All iterations failed for {strategy_name}!")

        self.all_results.extend(results)
        return results

    def run_all(self, strategies=None):
        if strategies is None:
            strategies = list(self.LOADER_CLASSES.keys())

        print("\n" + "#" * 60)
        print("#  STARTING FULL BENCHMARK SUITE")
        print(f"#  Strategies: {strategies}")
        print(f"#  Models: {[m.split('/')[-1] for m in self.config.models]}")
        print(f"#  Iterations: {self.config.warmup_iterations}W + {self.config.measured_iterations}M")
        print("#" * 60)

        total_start = time.perf_counter()
        for model_name in self.config.models:
            CachedLoader.reset_cache()
            for strategy in strategies:
                try:
                    self.run_strategy_benchmark(strategy, model_name)
                except Exception as e:
                    print(f"\n  FATAL ERROR [{strategy}/{model_name}]: {e}")
                    import traceback
                    traceback.print_exc()

        total_time = time.perf_counter() - total_start
        print(f"\n\nBenchmark suite completed in {total_time/60:.1f} minutes.")
        print(f"Total measurements collected: {len(self.all_results)}")
        return self.to_dataframe()

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(r) for r in self.all_results])
