import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats


class StatisticalAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def summary_table(self) -> pd.DataFrame:
        metrics = ["load_time_s", "peak_memory_mb", "ttft_s", "throughput_tps"]
        rows = []
        for (strategy, model), group in self.df.groupby(["strategy", "model_name"]):
            row = {"strategy": strategy, "model": model.split('/')[-1]}
            for m in metrics:
                vals = group[m].dropna()
                if len(vals) > 0:
                    mean = vals.mean()
                    std = vals.std()
                    ci = stats.t.interval(
                        0.95, len(vals)-1, loc=mean, scale=stats.sem(vals)
                    ) if len(vals) > 1 else (mean, mean)
                    row[f"{m}_mean"] = mean
                    row[f"{m}_std"] = std
                    row[f"{m}_ci95_lo"] = ci[0]
                    row[f"{m}_ci95_hi"] = ci[1]
            rows.append(row)
        return pd.DataFrame(rows)

    def efficiency_metrics(self) -> pd.DataFrame:
        rows = []
        for model, mg in self.df.groupby("model_name"):
            bl = mg[mg["strategy"] == "standard"]
            if bl.empty:
                continue
            M_bl, T_bl = bl["peak_memory_mb"].mean(), bl["load_time_s"].mean()
            for strat, sg in mg.groupby("strategy"):
                M_o, T_o = sg["peak_memory_mb"].mean(), sg["load_time_s"].mean()
                rows.append({
                    "model": model.split('/')[-1], "strategy": strat,
                    "η_mem (%)": round((M_bl - M_o) / M_bl * 100, 2),
                    "η_load (%)": round((T_bl - T_o) / T_bl * 100, 2),
                    "memory_mb": round(M_o, 1), "load_time_s": round(T_o, 3),
                })
        return pd.DataFrame(rows)

    def pairwise_ttests(self, metric="load_time_s") -> pd.DataFrame:
        results = []
        for model, mg in self.df.groupby("model_name"):
            bl = mg[mg["strategy"] == "standard"][metric].values
            if len(bl) < 2:
                continue
            for strat in mg["strategy"].unique():
                if strat == "standard":
                    continue
                sv = mg[mg["strategy"] == strat][metric].values
                if len(sv) < 2:
                    continue
                t_stat, p_val = stats.ttest_ind(bl, sv)
                results.append({
                    "model": model.split('/')[-1],
                    "comparison": f"standard vs {strat}",
                    "metric": metric,
                    "t_statistic": round(t_stat, 4),
                    "p_value": round(p_val, 6),
                    "significant_005": p_val < 0.05,
                    "significant_001": p_val < 0.01,
                })
        return pd.DataFrame(results)


def create_comparison_plots(df, output_dir="results"):
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = {
        "standard": "#4C72B0", "mmap": "#55A868",
        "lazy": "#C44E52", "streaming": "#8172B3", "cached": "#CCB974",
    }
    models = df["model_name"].unique()
    strategies = df["strategy"].unique()

    metrics_to_plot = [
        ("load_time_s", "Load Time (s)"),
        ("peak_memory_mb", "Peak Memory (MB)"),
        ("ttft_s", "Time to First Token (s)"),
        ("throughput_tps", "Throughput (tokens/s)"),
    ]

    saved_paths = []

    for model_name in models:
        mdf = df[df["model_name"] == model_name]
        short = model_name.split('/')[-1]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Benchmark Results: {short}", fontsize=16, fontweight="bold")

        for ax, (metric, label) in zip(axes.flat, metrics_to_plot):
            means, stds, labels_list, clrs = [], [], [], []
            for s in strategies:
                vals = mdf[mdf["strategy"] == s][metric].dropna()
                if len(vals) > 0:
                    means.append(vals.mean())
                    stds.append(vals.std())
                    labels_list.append(s)
                    clrs.append(colors.get(s, "#999"))
            x = np.arange(len(labels_list))
            bars = ax.bar(x, means, yerr=stds, capsize=5,
                         color=clrs, edgecolor="white", alpha=0.9)
            ax.set_xticks(x)
            ax.set_xticklabels(labels_list, rotation=25, ha="right")
            ax.set_ylabel(label)
            ax.set_title(label, fontsize=12)
            for bar, m in zip(bars, means):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                       f"{m:.2f}", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        path = os.path.join(output_dir, f"comparison_{short}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(path)

    eff = StatisticalAnalyzer(df).efficiency_metrics()
    if not eff.empty:
        for model_short in eff["model"].unique():
            me = eff[eff["model"] == model_short]
            fig, ax = plt.subplots(figsize=(8, 5))
            pivot = me.set_index("strategy")[["η_mem (%)", "η_load (%)"]]
            im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, fontsize=11)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index, fontsize=11)
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    v = pivot.values[i, j]
                    ax.text(j, i, f"{v:.1f}%", ha="center", va="center",
                           fontsize=12, fontweight="bold",
                           color="white" if abs(v) > 20 else "black")
            plt.colorbar(im, ax=ax, label="Efficiency (%)")
            ax.set_title(f"Efficiency vs Baseline — {model_short}", fontsize=14)
            plt.tight_layout()
            path = os.path.join(output_dir, f"efficiency_{model_short}.png")
            plt.savefig(path, dpi=150)
            plt.close(fig)
            saved_paths.append(path)

    for model_name in models:
        mdf = df[df["model_name"] == model_name]
        short = model_name.split('/')[-1]
        fig, ax = plt.subplots(figsize=(12, 5))
        for s in strategies:
            vals = mdf[mdf["strategy"] == s]["load_time_s"].values
            if len(vals) > 0:
                ax.plot(range(1, len(vals)+1), vals, marker="o", label=s,
                       alpha=0.8, color=colors.get(s, "#999"))
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Load Time (s)")
        ax.set_title(f"Load Time Across Iterations — {short}", fontsize=14)
        ax.legend()
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        plt.tight_layout()
        path = os.path.join(output_dir, f"iterations_{short}.png")
        plt.savefig(path, dpi=150)
        plt.close(fig)
        saved_paths.append(path)

    return saved_paths
