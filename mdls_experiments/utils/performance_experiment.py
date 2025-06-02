"""Performance experiment for drift detection methods (DDMs)."""

import random
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns

from river import drift


DDMS = {
    "JSWIN": drift.JSWIN(alpha=0.45),
    "KSWIN": drift.KSWIN(alpha=0.001),
    "ADWIN": drift.ADWIN(delta=0.002),
    "PH": drift.PageHinkley(delta=0.002, min_instances=30),
}


def compare_time_execution(n_iterations=10_000):
    """Compare the execution time of different drift detection methods."""
    results = {}

    for name, ddm in DDMS.items():
        np.random.seed(42)
        random.seed(42)
        total_time = 0.0
        for _ in range(n_iterations):
            start_time = time.perf_counter()
            ddm.update(np.random.uniform(0, 1))
            end_time = time.perf_counter()
            total_time += end_time - start_time

        avg_time = total_time / n_iterations
        results[name] = avg_time

    return results


def plot_execution_times(res):
    """Plot the average execution times of the drift detection methods."""
    sns.set_theme(style="whitegrid")
    df = pd.DataFrame.from_dict(res, orient="index", columns=["time"])
    df = df.sort_values(by="time", ascending=False)

    ax = df.plot(kind="barh", legend=False, figsize=(7, 5))
    plt.xlabel("Average time (seconds)")
    plt.title(
        "Average execution time per update call of JSWIN compared to other Drift Detectors"
    )
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.grid(axis="y", visible=False)
    ax.set_xlim(0, df["time"].max() * 1.1)

    color = ax.patches[0].get_facecolor()
    for i, v in enumerate(df["time"]):
        ax.text(v + 10e-7, i, f"{10_000*v:.2f}", va="center", fontsize=10, color=color)

    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    ax.xaxis.offsetText.set_va("bottom")
    ax.xaxis.offsetText.set_x(1.05)
    ax.xaxis.offsetText.set_fontsize(10)

    plt.tight_layout()
    plt.show()
