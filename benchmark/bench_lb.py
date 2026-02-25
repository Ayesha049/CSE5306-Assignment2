"""
bench_lb.py — Full-Pipeline Load Balancing Scalability Benchmark
=================================================================
Measures how long it takes to complete a fixed number of RL episodes
under different N2/N3 replica counts.

Why episodes?
  - Each N2 replica runs rollout threads independently.
  - More N2 replicas → less GIL/CPU contention per process → faster rollouts.
  - N6 records one `episode_reward` entry per completed episode, so
    counting new entries gives us a clean, pipeline-spanning completion signal.
  - Training steps (loss) are limited by N4's single lock (~5 steps/s)
    and do NOT scale with N2/N3 count — episodes do.

Measurement:
  start  = when all experiments have been launched
  end    = when N6 confirms `target_episodes` NEW episode_reward entries
  metric = total_time (s) — lower is better with more replicas

LOADS:
  ("Light",  20,   5)  →  5 concurrent experiments, target 20  episodes
  ("Medium", 100,  20) → 20 concurrent experiments, target 100 episodes
  ("Heavy",  300,  50) → 50 concurrent experiments, target 300 episodes

Usage:
    # Round 1 — baseline
    docker compose up --build -d
    python bench_lb.py --n2 1 --n3 1

    # Round 2
    docker compose up --build --scale environment=2 --scale buffer=2 -d
    python bench_lb.py --n2 2 --n3 2

    # Round 3
    docker compose up --build --scale environment=4 --scale buffer=4 -d
    python bench_lb.py --n2 4 --n3 4

    # Plot all rounds
    python bench_lb.py --plot
"""

import sys
import os
import argparse
import time
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../stubs"))

import grpc
import common_pb2
import experiment_pb2_grpc
import analytics_pb2
import analytics_pb2_grpc

N1_ADDR       = "localhost:50053"
N6_ADDR       = "localhost:50052"
RESULTS_FILE  = os.path.join(os.path.dirname(__file__), "results_lb.csv")
POLL_INTERVAL = 0.5   # seconds between N6 polls
TIMEOUT       = 600   # max seconds to wait per load level

LOADS = [
    ("Light",  20,   5),   # (label, target_episodes, num_concurrent_experiments)
    ("Medium", 100,  20),
    ("Heavy",  300,  50),
]

COLUMNS = [
    "n2_replicas", "n3_replicas",
    "load_level", "concurrent_experiments", "target_episodes",
    "actual_episodes", "total_time_s", "eps_per_sec",
]


# ── helpers ────────────────────────────────────────────────────────────────────

def count_episodes(analytics_stub) -> int:
    """Return total number of episode_reward entries in N6 (system-wide)."""
    try:
        resp = analytics_stub.QueryMetrics(
            analytics_pb2.QueryRequest(metric_name="episode_reward"),
            timeout=5,
        )
        return len(resp.metrics)
    except Exception as e:
        print(f"  [warn] N6 query failed: {e}")
        return 0


def make_config(exp_id: str):
    return common_pb2.ExperimentConfig(
        experiment_id=exp_id,
        env_name="CartPole",
        algorithm="DQN",
        seed=42,
        max_steps=500,
    )


# ── core benchmark ─────────────────────────────────────────────────────────────

def bench_full_pipeline(target_episodes: int, num_experiments: int,
                        label: str, n2: int, n3: int):
    """
    Start `num_experiments` concurrent experiments and measure how long it takes
    for N6 to record `target_episodes` NEW episode_reward entries.

    More N2 replicas → experiments distributed across more processes →
    less GIL contention → episodes complete faster → lower total_time.

    Returns (total_time, actual_episodes, episodes_per_sec).
    """
    exp_stub = experiment_pb2_grpc.ExperimentServiceStub(
        grpc.insecure_channel(N1_ADDR)
    )
    analytics_stub = analytics_pb2_grpc.AnalyticsServiceStub(
        grpc.insecure_channel(N6_ADDR)
    )

    tag     = f"lb_n2{n2}_n3{n3}_{label}"
    exp_ids = [f"{tag}_{i}" for i in range(num_experiments)]

    # 1. Record episode count baseline BEFORE starting
    baseline = count_episodes(analytics_stub)
    print(f"  Baseline episodes in N6: {baseline}")

    # 2. Start all experiments
    print(f"  Starting {num_experiments} concurrent experiments...")
    for exp_id in exp_ids:
        try:
            exp_stub.StartExperiment(make_config(exp_id), timeout=15)
        except Exception as e:
            print(f"  [warn] StartExperiment({exp_id}) failed: {e}")

    t_start = time.time()
    print(f"  All experiments running. Waiting for {target_episodes} new episodes...")

    # 3. Poll N6 until target reached
    # Both total_time and actual_eps are captured at the same poll instant
    # so that eps_per_sec = actual_eps / total_time is internally consistent.
    total_time = None
    actual_eps = 0
    last_print = -10

    while True:
        new_episodes = count_episodes(analytics_stub) - baseline
        elapsed      = time.time() - t_start

        if elapsed - last_print >= 5:   # print progress every 5 seconds
            print(f"  [{elapsed:5.1f}s] episodes: {new_episodes}/{target_episodes}")
            last_print = elapsed

        if new_episodes >= target_episodes:
            total_time = elapsed        # lock time at this exact poll
            actual_eps = new_episodes   # lock count at this exact poll
            break
        if elapsed > TIMEOUT:
            print(f"  [warn] Timeout ({TIMEOUT}s) before reaching target.")
            total_time = elapsed
            actual_eps = new_episodes
            break

        time.sleep(POLL_INTERVAL)

    # 4. Stop all experiments
    print(f"  Stopping experiments...")
    for exp_id in exp_ids:
        try:
            exp_stub.StopExperiment(make_config(exp_id), timeout=10)
        except Exception:
            pass

    eps_per_sec = actual_eps / total_time if total_time and total_time > 0 else 0
    return total_time, actual_eps, eps_per_sec


# ── CSV helpers ────────────────────────────────────────────────────────────────

def save_result(row: dict):
    write_header = not os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ── plot ───────────────────────────────────────────────────────────────────────

def plot_results():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("pip install matplotlib")
        return

    rows = []
    with open(RESULTS_FILE) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("No data in", RESULTS_FILE)
        return

    load_levels = [l[0] for l in LOADS]
    configs = sorted(
        {f"N2={r['n2_replicas']} N3={r['n3_replicas']}" for r in rows},
        key=lambda s: int(s.split()[0].split("=")[1])
    )
    x     = range(len(load_levels))
    width = 0.8 / len(configs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Load Balancing Scalability — Time to Complete Fixed Episode Target",
        fontsize=12, fontweight="bold"
    )

    for k, cfg in enumerate(configs):
        offset      = (k - len(configs) / 2 + 0.5) * width
        total_times = []
        eps_per_sec = []

        for load in load_levels:
            match = [r for r in rows
                     if r["load_level"] == load
                     and f"N2={r['n2_replicas']} N3={r['n3_replicas']}" == cfg]
            total_times.append(float(match[-1]["total_time_s"])  if match else 0)
            eps_per_sec.append(float(match[-1]["eps_per_sec"])   if match else 0)

        bars1 = ax1.bar([xi + offset for xi in x], total_times, width, label=cfg)
        bars2 = ax2.bar([xi + offset for xi in x], eps_per_sec, width, label=cfg)

        for bar, val in zip(bars1, total_times):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     f"{val:.1f}s", ha="center", va="bottom", fontsize=7)
        for bar, val in zip(bars2, eps_per_sec):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     f"{val:.1f}", ha="center", va="bottom", fontsize=7)

    ax1.set_title("Total Time to Complete Episode Target\n(lower = better)")
    ax1.set_ylabel("Wall-clock time (seconds)")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels([f"{l}\n({t} eps, {c} exp)"
                         for l, t, c in LOADS])
    ax1.legend()

    ax2.set_title("Episode Throughput\n(higher = better)")
    ax2.set_ylabel("Episodes / second")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels([f"{l}\n({t} eps, {c} exp)"
                         for l, t, c in LOADS])
    ax2.legend()

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "lb_scaling.png")
    plt.savefig(out, dpi=150)
    print(f"Plot saved: {out}")
    plt.show()


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Scalability benchmark: time to complete fixed episode target"
    )
    parser.add_argument("--n2",   type=int, default=1, help="N2 replica count currently running")
    parser.add_argument("--n3",   type=int, default=1, help="N3 replica count currently running")
    parser.add_argument("--plot", action="store_true",  help="Plot all saved results and exit")
    args = parser.parse_args()

    if args.plot:
        plot_results()
        return

    print(f"\n{'='*60}")
    print(f"  Scalability Benchmark  (N2={args.n2}, N3={args.n3})")
    print(f"  Metric: time to reach episode target via full RL pipeline")
    print(f"  N1→N2 rollout → N3 buffer → N4 training → N6 metrics")
    print(f"{'='*60}")

    for load_label, target_episodes, num_experiments in LOADS:
        print(f"\n── Load={load_label}  "
              f"target={target_episodes} episodes  "
              f"experiments={num_experiments} concurrent ──")

        total_time, actual_eps, eps_per_sec = bench_full_pipeline(
            target_episodes = target_episodes,
            num_experiments = num_experiments,
            label           = load_label,
            n2              = args.n2,
            n3              = args.n3,
        )

        print(f"\n  Result:")
        print(f"    Total time     : {total_time:.2f} s")
        print(f"    Episodes done  : {actual_eps}")
        print(f"    Episodes/sec   : {eps_per_sec:.2f}")

        save_result({
            "n2_replicas":           args.n2,
            "n3_replicas":           args.n3,
            "load_level":            load_label,
            "concurrent_experiments": num_experiments,
            "target_episodes":       target_episodes,
            "actual_episodes":       actual_eps,
            "total_time_s":          round(total_time, 3),
            "eps_per_sec":           round(eps_per_sec, 3),
        })

    print(f"\nResults saved to {RESULTS_FILE}")
    print("Run `python bench_lb.py --plot` after all rounds to compare.")


if __name__ == "__main__":
    main()
