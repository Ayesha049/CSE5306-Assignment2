import csv
from bench_core import run_benchmark

print("RUN_ALL STARTED")

MICRO_TARGET = "localhost:50053"
MONO_TARGET = "localhost:50063"

LOADS = [
    ("Light", 20, 5),
    ("Medium", 100, 20),
    ("Heavy", 300, 50),
]

def run_architecture(target, filename):

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Load", "Throughput", "AvgLatency", "P95Latency"])

        for label, requests, concurrency in LOADS:
            print(f"\nRunning {label} load on {target}")

            thr, avg, p95 = run_benchmark(
                target,
                requests,
                concurrency,
                label
            )

            print("Throughput:", thr)
            print("Avg Latency:", avg)
            print("P95:", p95)

            writer.writerow([label, thr, avg, p95])


if __name__ == "__main__":

    print("=== Running Microservice Benchmark ===")
    run_architecture(MICRO_TARGET, "results_micro.csv")

    print("\n=== Running Monolith Benchmark ===")
    run_architecture(MONO_TARGET, "results_mono.csv")