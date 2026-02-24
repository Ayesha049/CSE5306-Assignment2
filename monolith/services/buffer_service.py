"""
N3: Buffer Service (Monolithic)
──────────────────────────────
• Stores transitions
• Triggers learner (N4)
• Reports metrics to analytics (N6)
• No gRPC — direct function calls
"""

import buffer_pb2
import analytics_pb2
import learner_pb2
import common_pb2

import threading
import random
import pickle
import os
import time


# ── Config ─────────────────────────────────────────────────────────────────────
BUFFER_FILE = "replay_buffer.pkl"
SAVE_INTERVAL = 10
BUFFER_CAPACITY = 10000
TRAIN_THRESHOLD = 1000


# ════════════════════════════════════════════════════════════════════════════════
#  Buffer Service
# ════════════════════════════════════════════════════════════════════════════════

class BufferService:

    def __init__(self, learner_service=None, analytics_service=None, capacity=BUFFER_CAPACITY):
        self.capacity = capacity
        self.lock = threading.Lock()
        self.storage = []
        self.push_count = 0
        self.last_trigger_size = 0

        # injected services
        self.learner_service = learner_service
        self.analytics_service = analytics_service

        self.load_buffer()

    # ───────────────────────────────────────────────────────────────────────────
    # Persistence
    # ───────────────────────────────────────────────────────────────────────────

    def load_buffer(self):
        if os.path.exists(BUFFER_FILE):
            with open(BUFFER_FILE, "rb") as f:
                self.storage = pickle.load(f)
            print(f"[Buffer-Mono] Loaded {len(self.storage)} transitions.")
        else:
            print("[Buffer-Mono] Starting fresh buffer.")

    def save_buffer(self):
        with open(BUFFER_FILE, "wb") as f:
            pickle.dump(self.storage, f)
        print(f"[Buffer-Mono] Saved {len(self.storage)} transitions.")

    # ───────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ───────────────────────────────────────────────────────────────────────────

    def _send_buffer_metric(self, buffer_size):
        if not self.analytics_service:
            return

        try:
            metric = analytics_pb2.Metrics(
                experiment_id="exp_001",
                source_node="N3-Buffer",
                metric_name="buffer_size",
                value=float(buffer_size),
                step=int(time.time())
            )
            self.analytics_service.ReportMetrics(metric)
        except Exception as e:
            print(f"[Buffer-Mono] Metric failed: {e}")

    def _trigger_learner_training(self):
        if not self.learner_service:
            return

        try:
            print("[Buffer-Mono] Triggering learner training...")
            exp_config = common_pb2.ExperimentConfig(experiment_id="auto_exp")

            self.learner_service.TrainStep(exp_config)

        except Exception as e:
            print(f"[Buffer-Mono] Learner trigger failed: {e}")

    # ───────────────────────────────────────────────────────────────────────────
    # Public API (same names as gRPC for compatibility)
    # ───────────────────────────────────────────────────────────────────────────

    def PushTransition(self, request):
        with self.lock:

            transition = request.transition

            if len(self.storage) >= self.capacity:
                self.storage.pop(0)

            self.storage.append(transition)

            # periodic save
            self.push_count += 1
            if self.push_count % SAVE_INTERVAL == 0:
                self.save_buffer()

            total = len(self.storage)
            print(f"[Buffer-Mono] Stored transition. Total: {total}")

            # send metric
            self._send_buffer_metric(total)

            # trigger learner (same behavior as your original)
            print(f"[Buffer-Mono] Trigger learner at size={total}")
            threading.Thread(
                target=self._trigger_learner_training,
                daemon=True
            ).start()

            self.last_trigger_size = total

        return buffer_pb2.Status(ok=True)

    def SampleBatch(self, request):
        batch_size = request.batch_size

        with self.lock:
            current_size = len(self.storage)

            if current_size == 0:
                return buffer_pb2.Batch(transitions=[])

            actual_size = min(batch_size, current_size)
            batch = random.sample(self.storage, actual_size)

        print(f"[Buffer-Mono] Sampled batch size={len(batch)}")

        return buffer_pb2.Batch(transitions=batch)