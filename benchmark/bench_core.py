import grpc
import time
import threading
import statistics
import csv
import sys

sys.path.append("../monolith/stubs")

import experiment_pb2
import experiment_pb2_grpc
import common_pb2


def run_benchmark(target, num_requests, concurrency, label):
    latencies = []
    lock = threading.Lock()

    def worker(i):
        channel = grpc.insecure_channel(target)
        stub = experiment_pb2_grpc.ExperimentServiceStub(channel)

        config = common_pb2.ExperimentConfig(
            experiment_id=f"bench_{label}_{i}",
            env_name="CartPole",
            algorithm="DQN",
            seed=42,
            max_steps=50
        )

        start = time.time()
        try:
            stub.StartExperiment(config)
        except Exception:
            pass
        end = time.time()

        with lock:
            latencies.append(end - start)

    threads = []
    start_total = time.time()

    for i in range(num_requests):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

        if len(threads) >= concurrency:
            for th in threads:
                th.join()
            threads = []

    for th in threads:
        th.join()

    total_time = time.time() - start_total

    throughput = num_requests / total_time
    avg_latency = statistics.mean(latencies)

    latencies_sorted = sorted(latencies)
    p95_index = int(0.95 * len(latencies_sorted)) - 1
    p95_latency = latencies_sorted[max(p95_index, 0)]

    return throughput, avg_latency, p95_latency