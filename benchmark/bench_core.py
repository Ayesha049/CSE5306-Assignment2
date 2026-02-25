import grpc
import time
import threading
import statistics
import sys

sys.path.append("../monolith/stubs")

import experiment_pb2
import experiment_pb2_grpc
import analytics_pb2
import analytics_pb2_grpc
import common_pb2


def run_benchmark(target, analytics_target, num_requests, concurrency, label):
    latencies = []
    lock = threading.Lock()

    def worker(i):
        exp_id = f"bench_{label}_{i}"

        exp_channel = grpc.insecure_channel(target)
        exp_stub = experiment_pb2_grpc.ExperimentServiceStub(exp_channel)

        analytics_channel = grpc.insecure_channel(analytics_target)
        analytics_stub = analytics_pb2_grpc.AnalyticsServiceStub(analytics_channel)

        config = common_pb2.ExperimentConfig(
            experiment_id=exp_id,
            env_name="CartPole",
            algorithm="DQN",
            seed=42,
            max_steps=20
        )

        start = time.time()

        try:
            exp_stub.StartExperiment(config)

            # ⏳ Wait until first episode_reward appears
            timeout = 15
            deadline = time.time() + timeout

            while time.time() < deadline:
                result = analytics_stub.QueryMetrics(
                    analytics_pb2.QueryRequest(
                        experiment_id=exp_id,
                        metric_name="episode_reward"
                    )
                )

                if result.metrics:
                    break

                time.sleep(0.2)

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