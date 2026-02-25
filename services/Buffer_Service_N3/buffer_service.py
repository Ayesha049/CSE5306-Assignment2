"""
N3: Buffer Service (Optimized for Benchmarking)
"""

import sys
sys.path.append("/app/stubs")

import grpc
from concurrent import futures 
import buffer_pb2
import buffer_pb2_grpc
import analytics_pb2
import analytics_pb2_grpc
import learner_pb2
import learner_pb2_grpc
import common_pb2
import threading
import random
import pickle
import os
import time

BUFFER_FILE = "replay_buffer.pkl"
SAVE_INTERVAL = 1000     # reduced disk overhead
BUFFER_CAPACITY = 10000
TRAIN_THRESHOLD = 1000

LEARNER_HOST = "learner:50054"
ANALYTICS_HOST = "analytics:50052"


class BufferService(buffer_pb2_grpc.BufferServiceServicer):

    def __init__(self):
        self.capacity = BUFFER_CAPACITY
        self.lock = threading.Lock()
        self.storage = []
        self.push_count = 0
        self.last_trigger_size = 0

        # ✅ Create stubs once
        self.analytics_channel = grpc.insecure_channel(ANALYTICS_HOST)
        self.analytics_stub = analytics_pb2_grpc.AnalyticsServiceStub(self.analytics_channel)

        self.learner_channel = grpc.insecure_channel(LEARNER_HOST)
        self.learner_stub = learner_pb2_grpc.LearnerServiceStub(self.learner_channel)

        self.load_buffer()

    def load_buffer(self):
        if os.path.exists(BUFFER_FILE):
            with open(BUFFER_FILE, "rb") as f:
                self.storage = pickle.load(f)

    def save_buffer(self):
        with open(BUFFER_FILE, "wb") as f:
            pickle.dump(self.storage, f)

    def send_metric(self, size):
        try:
            metric = analytics_pb2.Metrics(
                experiment_id="bench_exp",
                source_node="N3-Buffer",
                metric_name="buffer_size",
                value=float(size),
                step=int(time.time())
            )
            self.analytics_stub.ReportMetrics(metric)
        except Exception:
            pass

    def trigger_training(self):
        try:
            config = common_pb2.ExperimentConfig(experiment_id="bench_exp")
            self.learner_stub.TrainStep(config)
        except Exception:
            pass

    def PushTransition(self, request, context):
        with self.lock:
            if len(self.storage) >= self.capacity:
                self.storage.pop(0)

            self.storage.append(request.transition)
            self.push_count += 1

            if self.push_count % SAVE_INTERVAL == 0:
                self.save_buffer()

            total = self.push_count
            self.send_metric(len(self.storage))

            # ✅ SAFE threshold logic (no thread explosion)
            if total >= TRAIN_THRESHOLD and total - self.last_trigger_size >= TRAIN_THRESHOLD:
                threading.Thread(
                    target=self.trigger_training,
                    daemon=True
                ).start()
                self.last_trigger_size = total

        return buffer_pb2.Status(ok=True)

    def SampleBatch(self, request, context):
        with self.lock:
            if not self.storage:
                return buffer_pb2.Batch(transitions=[])

            batch = random.sample(
                self.storage,
                min(request.batch_size, len(self.storage))
            )

        return buffer_pb2.Batch(transitions=batch)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=20))
    buffer_pb2_grpc.add_BufferServiceServicer_to_server(BufferService(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()