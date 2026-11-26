"""Plot whatever result files exist in results/."""
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "results"


def sim():
    for path in glob.glob(f"{R}/sim_hitrate*.csv"):
        rows = list(csv.DictReader(open(path)))
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.bar([r["map"] for r in rows], [float(r["prefix_hit_rate"]) for r in rows])
        ax.set_ylabel("prefix hit rate"); ax.set_title(os.path.basename(path))
        plt.xticks(rotation=25, ha="right"); plt.tight_layout()
        fig.savefig(path.replace(".csv", ".png"), dpi=150); plt.close(fig)


def breakeven():
    p = f"{R}/breakeven.csv"
    if not os.path.exists(p):
        return
    rows = [r for r in csv.DictReader(open(p)) if r["breakeven_tokens"] != "never"]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for prec in ("fp16", "fp8", "int8", "int4"):
        xs = [r["tier"] for r in rows if r["precision"] == prec]
        ys = [int(r["breakeven_tokens"]) for r in rows if r["precision"] == prec]
        ax.plot(xs, ys, marker="o", label=prec)
    ax.set_yscale("log"); ax.set_ylabel("break-even context (tokens)"); ax.legend()
    plt.tight_layout(); fig.savefig(f"{R}/breakeven.png", dpi=150); plt.close(fig)


def ttft():
    files = sorted(glob.glob(f"{R}/ttft_*.json"))
    if not files:
        return
    tags, p50, p99 = [], [], []
    for f in files:
        d = json.load(open(f)); s = d["summary"]["overall"]
        tags.append(d["tag"]); p50.append(s["p50"]); p99.append(s["p99"])
    fig, ax = plt.subplots(figsize=(7, 3.5))
    x = range(len(tags))
    ax.bar([i - 0.2 for i in x], p50, 0.4, label="P50"); ax.bar([i + 0.2 for i in x], p99, 0.4, label="P99")
    ax.set_xticks(list(x)); ax.set_xticklabels(tags); ax.set_ylabel("TTFT (s)"); ax.legend()
    plt.tight_layout(); fig.savefig(f"{R}/ttft.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    sim(); breakeven(); ttft()
