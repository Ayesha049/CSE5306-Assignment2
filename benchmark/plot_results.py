import csv
import os

import matplotlib.pyplot as plt
import numpy as np


def load_results(filename):
    loads, throughput, avg_lat, p95_lat = [], [], [], []
    with open(filename) as f:
        for row in csv.DictReader(f):
            loads.append(row["Load"])
            throughput.append(float(row["Throughput"]))
            avg_lat.append(float(row["AvgLatency"]) * 1000)   # s → ms
            p95_lat.append(float(row["P95Latency"]) * 1000)   # s → ms
    return loads, throughput, avg_lat, p95_lat


script_dir = os.path.dirname(os.path.abspath(__file__))
loads, mono_thr, mono_avg, mono_p95 = load_results(os.path.join(script_dir, "results_mono.csv"))
_,     micro_thr, micro_avg, micro_p95 = load_results(os.path.join(script_dir, "results_micro.csv"))

x = np.arange(len(loads))
bar_w = 0.35

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Monolithic vs Microservices — Benchmark Comparison", fontsize=14, fontweight="bold")

# ── Throughput ──────────────────────────────────────────────────────────────
ax = axes[0]
b1 = ax.bar(x - bar_w / 2, mono_thr,  bar_w, label="Monolithic",    color="#4C72B0")
b2 = ax.bar(x + bar_w / 2, micro_thr, bar_w, label="Microservices", color="#DD8452")
ax.set_title("Throughput")
ax.set_ylabel("Requests / second")
ax.set_xticks(x)
ax.set_xticklabels(loads)
ax.set_xlabel("Load level")
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.bar_label(b1, fmt="%.1f", padding=3, fontsize=8)
ax.bar_label(b2, fmt="%.1f", padding=3, fontsize=8)

# ── Average Latency ──────────────────────────────────────────────────────────
ax = axes[1]
b1 = ax.bar(x - bar_w / 2, mono_avg,  bar_w, label="Monolithic",    color="#4C72B0")
b2 = ax.bar(x + bar_w / 2, micro_avg, bar_w, label="Microservices", color="#DD8452")
ax.set_title("Average Latency")
ax.set_ylabel("Latency (ms)")
ax.set_xticks(x)
ax.set_xticklabels(loads)
ax.set_xlabel("Load level")
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.bar_label(b1, fmt="%.1f", padding=3, fontsize=8)
ax.bar_label(b2, fmt="%.1f", padding=3, fontsize=8)

# ── P95 Latency ───────────────────────────────────────────────────────────────
ax = axes[2]
b1 = ax.bar(x - bar_w / 2, mono_p95,  bar_w, label="Monolithic",    color="#4C72B0")
b2 = ax.bar(x + bar_w / 2, micro_p95, bar_w, label="Microservices", color="#DD8452")
ax.set_title("P95 Latency")
ax.set_ylabel("Latency (ms)")
ax.set_xticks(x)
ax.set_xticklabels(loads)
ax.set_xlabel("Load level")
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.bar_label(b1, fmt="%.1f", padding=3, fontsize=8)
ax.bar_label(b2, fmt="%.1f", padding=3, fontsize=8)

plt.tight_layout()
out_path = os.path.join(script_dir, "benchmark_comparison.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
