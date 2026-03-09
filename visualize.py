#!/usr/bin/env python3
"""
Benchmark Showdown Visualizer
Load two JSON result files and generate comparison charts + winner summary.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Friendly display names
# ---------------------------------------------------------------------------
FRIENDLY_NAMES = {
    "cpu_single_core_fibonacci": "CPU Single-Core\n(Fibonacci)",
    "cpu_multi_core": "CPU Multi-Core\n(Parallel)",
    "numpy_matmul": "NumPy\n(Matrix Multiply)",
    "sklearn_random_forest": "Random Forest\n(Training)",
    "sklearn_logistic_regression": "Logistic Reg.\n(Training)",
    "pytorch_cpu_nn": "PyTorch CPU\n(Neural Net)",
    "gpu_matmul": "GPU\n(Matrix Multiply)",
    "pytorch_gpu_nn": "PyTorch GPU\n(Neural Net)",
    "pandas_operations": "Pandas\n(Data Processing)",
}

COLORS = {
    "machine1": "#3498db",
    "machine2": "#e74c3c",
    "faster": "#2ecc71",
    "slower": "#e74c3c",
    "tie": "#95a5a6",
}


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def get_machine_label(result: dict) -> str:
    return result.get("machine_name", result["system_info"].get("platform", "Unknown"))


def extract_times(results: dict) -> dict[str, float]:
    times = {}
    for key, val in results["benchmarks"].items():
        if isinstance(val, dict) and "error" not in val:
            t = val.get("time_seconds")
            if t is not None:
                times[key] = t
    return times


def common_benchmarks(t1: dict, t2: dict) -> list[str]:
    """Return benchmark keys present in both, in a stable order."""
    order = list(FRIENDLY_NAMES.keys())
    common = set(t1.keys()) & set(t2.keys())
    ordered = [k for k in order if k in common]
    # any extra keys not in FRIENDLY_NAMES
    ordered += sorted(common - set(ordered))
    return ordered


def friendly(key: str) -> str:
    return FRIENDLY_NAMES.get(key, key.replace("_", " ").title())


# ---------------------------------------------------------------------------
# Chart 1: Side-by-side bar comparison
# ---------------------------------------------------------------------------
def plot_comparison(
    keys: list[str],
    t1: dict,
    t2: dict,
    label1: str,
    label2: str,
    output_dir: Path,
):
    fig, ax = plt.subplots(figsize=(max(10, len(keys) * 1.4), 6))

    x = np.arange(len(keys))
    width = 0.35

    vals1 = [t1[k] for k in keys]
    vals2 = [t2[k] for k in keys]

    bars1 = ax.bar(x - width / 2, vals1, width, label=label1, color=COLORS["machine1"])
    bars2 = ax.bar(x + width / 2, vals2, width, label=label2, color=COLORS["machine2"])

    ax.set_ylabel("Time (seconds) — lower is better")
    ax.set_title(f"Office Benchmark Showdown\n{label1}  vs  {label2}")
    ax.set_xticks(x)
    ax.set_xticklabels([friendly(k) for k in keys], fontsize=8)
    ax.legend()

    # value labels on bars
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.annotate(
            f"{h:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    fig.tight_layout()
    path = output_dir / "comparison_times.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"   Saved: {path}")


# ---------------------------------------------------------------------------
# Chart 2: Speedup ratios
# ---------------------------------------------------------------------------
def plot_speedup(
    keys: list[str],
    t1: dict,
    t2: dict,
    label1: str,
    label2: str,
    output_dir: Path,
):
    ratios = [t1[k] / t2[k] if t2[k] > 0 else 1.0 for k in keys]
    labels = [friendly(k).replace("\n", " ") for k in keys]

    fig, ax = plt.subplots(figsize=(10, max(4, len(keys) * 0.6)))

    colors = [
        COLORS["faster"] if r > 1.05 else COLORS["slower"] if r < 0.95 else COLORS["tie"]
        for r in ratios
    ]

    y = np.arange(len(keys))
    bars = ax.barh(y, ratios, color=colors)

    ax.axvline(x=1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(f"Speedup ratio (>1 = {label2} faster, <1 = {label1} faster)")
    ax.set_title(f"Speedup: {label1} time / {label2} time")

    for bar, ratio in zip(bars, ratios):
        w = bar.get_width()
        ax.text(
            w + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{ratio:.2f}x",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    fig.tight_layout()
    path = output_dir / "speedup_ratios.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"   Saved: {path}")


# ---------------------------------------------------------------------------
# Chart 3: Winner scoreboard
# ---------------------------------------------------------------------------
def plot_scoreboard(
    keys: list[str],
    t1: dict,
    t2: dict,
    label1: str,
    label2: str,
    output_dir: Path,
):
    wins1, wins2, ties = 0, 0, 0
    details = []
    for k in keys:
        ratio = t1[k] / t2[k] if t2[k] > 0 else 1.0
        name = friendly(k).replace("\n", " ")
        if ratio > 1.05:
            wins2 += 1
            details.append(f"  {name}: {label2} ({ratio:.1f}x faster)")
        elif ratio < 0.95:
            wins1 += 1
            details.append(f"  {name}: {label1} ({1/ratio:.1f}x faster)")
        else:
            ties += 1
            details.append(f"  {name}: TIE")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")

    # title
    ax.text(0.5, 0.92, "FINAL SCORE", ha="center", va="top",
            fontsize=28, fontweight="bold", transform=ax.transAxes)

    # scores
    ax.text(0.25, 0.72, str(wins1), ha="center", va="center",
            fontsize=60, fontweight="bold", color=COLORS["machine1"],
            transform=ax.transAxes)
    ax.text(0.5, 0.72, "vs", ha="center", va="center",
            fontsize=24, color="#666", transform=ax.transAxes)
    ax.text(0.75, 0.72, str(wins2), ha="center", va="center",
            fontsize=60, fontweight="bold", color=COLORS["machine2"],
            transform=ax.transAxes)

    # names
    ax.text(0.25, 0.52, label1, ha="center", va="center",
            fontsize=13, color=COLORS["machine1"], fontweight="bold",
            transform=ax.transAxes)
    ax.text(0.75, 0.52, label2, ha="center", va="center",
            fontsize=13, color=COLORS["machine2"], fontweight="bold",
            transform=ax.transAxes)
    if ties:
        ax.text(0.5, 0.52, f"({ties} tie{'s' if ties != 1 else ''})",
                ha="center", va="center", fontsize=11, color="#999",
                transform=ax.transAxes)

    # champion
    if wins1 > wins2:
        champ = label1
    elif wins2 > wins1:
        champ = label2
    else:
        champ = "IT'S A DRAW!"
    ax.text(0.5, 0.38, f"CHAMPION: {champ}",
            ha="center", va="center", fontsize=16, fontweight="bold",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0", edgecolor="#ccc"))

    # details
    detail_text = "\n".join(details)
    ax.text(0.5, 0.05, detail_text, ha="center", va="bottom",
            fontsize=8, family="monospace", transform=ax.transAxes,
            linespacing=1.5)

    fig.tight_layout()
    path = output_dir / "winner_summary.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"   Saved: {path}")


# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------
def print_summary(
    keys: list[str],
    t1: dict,
    t2: dict,
    label1: str,
    label2: str,
):
    wins1, wins2 = 0, 0
    print()
    print("=" * 55)
    print("   OFFICE SHOWDOWN RESULTS")
    print("=" * 55)
    print(f"   {label1}  vs  {label2}")
    print("-" * 55)

    for k in keys:
        name = friendly(k).replace("\n", " ")
        ratio = t1[k] / t2[k] if t2[k] > 0 else 1.0
        if ratio > 1.05:
            winner = label2
            speed = f"{ratio:.1f}x faster"
            wins2 += 1
        elif ratio < 0.95:
            winner = label1
            speed = f"{1/ratio:.1f}x faster"
            wins1 += 1
        else:
            winner = "TIE"
            speed = ""
        padding = 28 - len(name)
        print(f"   {name}{' ' * max(1, padding)}{winner} {speed}")

    print("-" * 55)
    print(f"   FINAL: {label1}: {wins1}  |  {label2}: {wins2}")
    if wins1 > wins2:
        print(f"   CHAMPION: {label1}!")
    elif wins2 > wins1:
        print(f"   CHAMPION: {label2}!")
    else:
        print("   IT'S A DRAW!")
    print("=" * 55)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compare two benchmark result files and generate charts."
    )
    parser.add_argument("file1", help="First machine JSON results")
    parser.add_argument("file2", help="Second machine JSON results")
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for output PNGs (default: current dir)",
    )
    args = parser.parse_args()

    r1 = load_results(args.file1)
    r2 = load_results(args.file2)

    label1 = get_machine_label(r1)
    label2 = get_machine_label(r2)

    t1 = extract_times(r1)
    t2 = extract_times(r2)

    keys = common_benchmarks(t1, t2)
    if not keys:
        print("No common benchmarks found between the two files.")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n   Comparing: {label1} vs {label2}")
    print(f"   Common benchmarks: {len(keys)}")

    plot_comparison(keys, t1, t2, label1, label2, output_dir)
    plot_speedup(keys, t1, t2, label1, label2, output_dir)
    plot_scoreboard(keys, t1, t2, label1, label2, output_dir)
    print_summary(keys, t1, t2, label1, label2)


if __name__ == "__main__":
    main()
