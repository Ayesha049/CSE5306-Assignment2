from bench_core import run_benchmark
import csv

MICRO_TARGET = "localhost:50053"
MICRO_ANALYTICS = "localhost:50052"

MONO_TARGET = "localhost:50063"
MONO_ANALYTICS = "localhost:50063"   # monolith handles analytics inside

WORKLOADS = [
    ("Light", 50, 5),
    ("Medium", 200, 20),
    ("Heavy", 500, 50),
]


def run_architecture(target, analytics_target, output_file):
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Load", "Throughput", "AvgLatency", "P95Latency"])

        for label, requests, concurrency in WORKLOADS:
            print(f"\nRunning {label} load on {target}")

            throughput, avg, p95 = run_benchmark(
                target,
                analytics_target,
                requests,
                concurrency,
                label
            )

            print("Throughput:", throughput)
            print("Avg Latency:", avg)
            print("P95:", p95)

            writer.writerow([label, throughput, avg, p95])


if __name__ == "__main__":
    print("RUN_ALL STARTED")

    print("=== Running Microservice Benchmark ===")
    run_architecture(MICRO_TARGET, MICRO_ANALYTICS, "results_micro.csv")

    print("\n=== Running Monolith Benchmark ===")
    run_architecture(MONO_TARGET, MONO_ANALYTICS, "results_mono.csv")