import sys
sys.path.append("/app/stubs")  # path inside Docker container

import grpc
from concurrent import futures
import buffer_pb2
import buffer_pb2_grpc
import analytics_pb2
import analytics_pb2_grpc
import learner_pb2
import learner_pb2_grpc
import common_pb2  # <-- use this for ExperimentConfig
import threading
import random
import pickle
import os
import time

# Config
BUFFER_FILE = "replay_buffer.pkl"
SAVE_INTERVAL = 10  # save every N pushes
BUFFER_CAPACITY = 10000
TRAIN_THRESHOLD = 1000      # number of transitions to trigger training
TRAIN_BATCH_SIZE = 32       # for reference
LEARNER_HOST = "learner:50054"
ANALYTICS_HOST = "analytics:50052"

# ----------------- Helper functions -----------------

def send_buffer_metric(buffer_size):
    try:
        channel = grpc.insecure_channel(ANALYTICS_HOST)
        stub = analytics_pb2_grpc.AnalyticsServiceStub(channel)

        metric = analytics_pb2.Metrics(
            experiment_id="exp_001",
            source_node="N3-Buffer",
            metric_name="buffer_size",
            value=float(buffer_size),
            step=int(time.time())
        )

        response = stub.ReportMetrics(metric)
        print(f"[Buffer] Metric sent: buffer_size={buffer_size} ok={response.ok}")
    except Exception as e:
        print(f"[Buffer] Metric send failed: {e}")


def trigger_learner_training():

    try:
        print("[Buffer] Inside trigger_learner_training()")
        channel = grpc.insecure_channel("learner:50054")
        stub = learner_pb2_grpc.LearnerServiceStub(channel)

        exp_config = common_pb2.ExperimentConfig(experiment_id="auto_exp")
        response = stub.TrainStep(exp_config)

        print(f"[Buffer] Learner TrainStep called: ok={response.ok}")
    except Exception as e:
        print(f"[Buffer] Failed to trigger learner: {e}")


# ----------------- Buffer Service -----------------

class BufferService(buffer_pb2_grpc.BufferServiceServicer):
    def __init__(self, capacity=BUFFER_CAPACITY):
        self.capacity = capacity
        self.lock = threading.Lock()
        self.storage = []
        self.push_count = 0
        self.last_trigger_size = 0  # track last training trigger
        self.load_buffer()

    # Load existing buffer from disk
    def load_buffer(self):
        if os.path.exists(BUFFER_FILE):
            with open(BUFFER_FILE, "rb") as f:
                self.storage = pickle.load(f)
            print(f"[Buffer] Loaded {len(self.storage)} transitions from disk.")
        else:
            print("[Buffer] No existing buffer file found. Starting fresh.")

    # Save buffer to disk
    def save_buffer(self):
        with open(BUFFER_FILE, "wb") as f:
            pickle.dump(self.storage, f)
        print(f"[Buffer] Saved {len(self.storage)} transitions to disk.")

    # Push a single transition into buffer
    def PushTransition(self, request, context):
        with self.lock:
            transition = request.transition  # protobuf Transition object

            if len(self.storage) >= self.capacity:
                self.storage.pop(0)
            self.storage.append(transition)

            # Save periodically
            self.push_count += 1
            if self.push_count % SAVE_INTERVAL == 0:
                self.save_buffer()

            total = len(self.storage)
            send_buffer_metric(total)
            print(f"[Buffer] Stored transition. Total: {total}")

            # Trigger learner if threshold is reached
            # if total >= TRAIN_THRESHOLD and total - self.last_trigger_size >= TRAIN_THRESHOLD:
            print(f"[Buffer] Threshold reached ({total}). Triggering learner training...")
            threading.Thread(target=trigger_learner_training).start()
            self.last_trigger_size = total

        return buffer_pb2.Status(ok=True)

    # Sample a random batch
    def SampleBatch(self, request, context):
        batch_size = request.batch_size
        with self.lock:
            current_size = len(self.storage)
            if current_size == 0:
                return buffer_pb2.Batch(transitions=[])

            actual_size = min(batch_size, current_size)
            batch = [t for t in random.sample(self.storage, actual_size)]

        print(f"[Buffer] Sampled batch of size {len(batch)}")
        return buffer_pb2.Batch(transitions=batch)


# ----------------- Run gRPC Server -----------------

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    buffer_pb2_grpc.add_BufferServiceServicer_to_server(BufferService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("[Buffer] Buffer Service running on port 50051")
    server.wait_for_termination()


if __name__ == "__main__":
    print("[Buffer] Starting gRPC server...")
    serve()