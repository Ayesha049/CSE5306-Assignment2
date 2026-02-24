"""
N4: Learner Service (Monolithic)
────────────────────────────────
• Pulls batch from BufferService
• Updates PolicyService
• Sends metrics to AnalyticsService
• No gRPC
"""

import time
import random
import threading

import buffer_pb2
import analytics_pb2
import learner_pb2
import policy_pb2
import common_pb2


class LearnerService:

    def __init__(self, buffer_service, policy_service, analytics_service=None):
        self.buffer_service = buffer_service
        self.policy_service = policy_service
        self.analytics_service = analytics_service

        self.current_experiment = None
        self.last_loss = 0.0
        self.lock = threading.Lock()

    # ────────────────────────────────────────────────────────────────
    # Internal helpers
    # ────────────────────────────────────────────────────────────────

    def _sample_batch(self, batch_size=32):
        request = buffer_pb2.SampleRequest(batch_size=batch_size)
        batch_response = self.buffer_service.SampleBatch(request)
        print(f"[Learner-Mono] Sampled batch size={len(batch_response.transitions)}")
        return batch_response.transitions

    def _train_policy(self, batch):
        print(f"[Learner-Mono] Training on batch size={len(batch)}")
        loss = random.random()
        time.sleep(0.2)  # simulate compute
        self.last_loss = loss
        print(f"[Learner-Mono] Training complete. Loss={loss:.4f}")
        return loss

    def _send_metric(self, metric_name, value):
        if not self.analytics_service:
            return

        metric = analytics_pb2.Metrics(
            experiment_id=self.current_experiment or "exp_unknown",
            source_node="N4-Learner",
            metric_name=metric_name,
            value=value,
            step=int(time.time())
        )

        try:
            self.analytics_service.ReportMetrics(metric)
        except Exception as e:
            print(f"[Learner-Mono] Metric failed: {e}")

    def _load_latest_policy(self):
        try:
            request = policy_pb2.PolicyRequest(
                experiment_id=self.current_experiment
            )
            policy = self.policy_service.GetPolicy(request)
            print(f"[Learner-Mono] Loaded policy v{policy.version}")
            return policy
        except Exception as e:
            print(f"[Learner-Mono] Failed to load policy: {e}")
            return None

    def _push_updated_policy(self, new_weights: bytes, version: int):
        try:
            policy_msg = common_pb2.Policy(
                experiment_id=self.current_experiment,
                version=version,
                weights=new_weights
            )
            self.policy_service.UpdatePolicy(policy_msg)
            print(f"[Learner-Mono] Updated policy to v{version}")
        except Exception as e:
            print(f"[Learner-Mono] Failed to push policy: {e}")

    # ────────────────────────────────────────────────────────────────
    # Public API (same names as gRPC for compatibility)
    # ────────────────────────────────────────────────────────────────

    def Initialize(self, request):
        with self.lock:
            self.current_experiment = request.experiment_id
            print(f"[Learner-Mono] Initialized experiment: {self.current_experiment}")

        return learner_pb2.Status(ok=True)

    def TrainStep(self, request):
        with self.lock:

            if request.experiment_id:
                self.current_experiment = request.experiment_id

            print(f"[Learner-Mono] TrainStep for {self.current_experiment}")

            # 1️⃣ Load policy
            latest_policy = self._load_latest_policy()

            # 2️⃣ Sample batch
            batch = self._sample_batch(batch_size=32)

            # 3️⃣ Train
            loss = self._train_policy(batch)

            # 4️⃣ Push updated policy
            if latest_policy:
                new_version = latest_policy.version + 1
                self._push_updated_policy(latest_policy.weights, new_version)

            # 5️⃣ Report metric
            self._send_metric("loss", loss)

        return learner_pb2.Status(ok=True)

    def GetTrainingStatus(self, request):
        with self.lock:
            if request.experiment_id:
                self.current_experiment = request.experiment_id

            return learner_pb2.Metrics(
                experiment_id=self.current_experiment or "exp_unknown",
                source_node="N4-Learner",
                metric_name="loss",
                value=self.last_loss,
                step=int(time.time())
            )