import sys
sys.path.append("/app/stubs")  # path inside Docker container

import grpc
import time
import random
from concurrent import futures
import threading

import buffer_pb2
import buffer_pb2_grpc
import analytics_pb2
import analytics_pb2_grpc
import learner_pb2
import learner_pb2_grpc
import policy_pb2
import policy_pb2_grpc
import common_pb2
import common_pb2_grpc

# Config: gRPC endpoints
BUFFER_HOSTS = ["buffer:50051"]  # can add more buffers
ANALYTICS_HOST = "analytics:50052"
POLICY_HOST = "policy:50056"  # N5 Policy Service gRPC endpoint
PORT = 50054

class LearnerService:
    """Core learner logic."""
    def __init__(self):
        self.current_experiment = None
        self.last_loss = 0.0

        # Connect to analytics
        self.analytics_channel = grpc.insecure_channel(ANALYTICS_HOST)
        self.analytics_stub = analytics_pb2_grpc.AnalyticsServiceStub(self.analytics_channel)

        self.policy_channel = grpc.insecure_channel(POLICY_HOST)
        self.policy_stub = policy_pb2_grpc.PolicyServiceStub(self.policy_channel)

    def sample_batch(self, batch_size=32):
        buffer_host = random.choice(BUFFER_HOSTS)
        channel = grpc.insecure_channel(buffer_host)
        stub = buffer_pb2_grpc.BufferServiceStub(channel)
        request = buffer_pb2.SampleRequest(batch_size=batch_size)
        batch_response = stub.SampleBatch(request)
        print(f"[Learner] Sampled batch of size {len(batch_response.transitions)} from buffer {buffer_host}")
        return batch_response.transitions

    def train_policy(self, batch):
        print(f"[Learner] Starting training on batch of size {len(batch)}...")
        # Dummy training: compute loss as random number
        loss = random.random()
        time.sleep(0.2)  # simulate training time
        print(f"[Learner] Training complete. Loss={loss:.4f}")
        self.last_loss = loss
        return loss

    def send_metric(self, metric_name, value):
        metric = analytics_pb2.Metrics(
            experiment_id=self.current_experiment or "exp_unknown",
            source_node="N4-Learner",
            metric_name=metric_name,
            value=value,
            step=int(time.time())
        )
        try:
            response = self.analytics_stub.ReportMetrics(metric)
            print(f"[Learner] Metric sent: {metric_name}={value} ok={response.ok}")
        except Exception as e:
            print(f"[Learner] Failed to send metric: {e}")

    def load_latest_policy(self):
        """Fetch the latest policy weights from Policy Service before training."""
        try:
            request = policy_pb2.PolicyRequest(experiment_id=self.current_experiment)
            policy = self.policy_stub.GetPolicy(request)
            print(f"[Learner] Loaded policy v{policy.version} from Policy Service")
            return policy
        except Exception as e:
            print(f"[Learner] Failed to load policy: {e}")
            return None

    def push_updated_policy(self, new_weights: bytes, version: int):
        """Push trained policy to Policy Service after training."""
        try:
            policy_msg = common_pb2.Policy(
                experiment_id=self.current_experiment,
                version=version,
                weights=new_weights
            )
            response = self.policy_stub.UpdatePolicy(policy_msg)
            print(f"[Learner] Pushed updated policy v{version} to Policy Service: ok={response.ok}")
        except Exception as e:
            print(f"[Learner] Failed to push policy: {e}")


class LearnerServiceServicer(learner_pb2_grpc.LearnerServiceServicer):
    """gRPC server implementing all LearnerService RPCs."""
    def __init__(self):
        self.core = LearnerService()
        self.lock = threading.Lock()  # protect training if multiple RPCs come

    def Initialize(self, request, context):
        with self.lock:
            self.core.current_experiment = request.experiment_id
            print(f"[Learner] Initialized experiment: {request.experiment_id}")
        return learner_pb2.Status(ok=True)

    def TrainStep(self, request, context):
        with self.lock:
            if request.experiment_id:
                self.core.current_experiment = request.experiment_id

            print(f"[Learner] TrainStep called for experiment: {self.core.current_experiment}")

            # 1️⃣ Load latest policy from Policy Service
            latest_policy = self.core.load_latest_policy()

            # Optionally, use latest_policy.weights in your training logic
            # For example, you could deserialize and initialize network weights here
            # In this dummy example, we just log it:
            if latest_policy:
                print(f"[Learner] Training will use policy v{latest_policy.version}")

            # 2️⃣ Sample batch from buffer
            batch = self.core.sample_batch(batch_size=32)

            # 3️⃣ Train on the batch
            loss = self.core.train_policy(batch)

            # 4️⃣ Update policy weights (dummy: reuse latest weights here)
            if latest_policy:
                new_version = latest_policy.version + 1
                self.core.push_updated_policy(latest_policy.weights, new_version)
            else:
                print("[Learner] No previous policy found; skipping push.")

            # 5️⃣ Report metric
            self.core.send_metric("loss", loss)

            print(f"[Learner] TrainStep complete on batch of size {len(batch)}\n")

        return learner_pb2.Status(ok=True)

    def GetTrainingStatus(self, request, context):
        with self.lock:
            if request.experiment_id:
                self.core.current_experiment = request.experiment_id
            print(f"[Learner] GetTrainingStatus called. Last loss={self.core.last_loss:.4f}")
            return learner_pb2.Metrics(
                experiment_id=self.core.current_experiment or "exp_unknown",
                source_node="N4-Learner",
                metric_name="loss",
                value=self.core.last_loss,
                step=int(time.time())
            )


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=5))
    learner_pb2_grpc.add_LearnerServiceServicer_to_server(
        LearnerServiceServicer(), server
    )

    server.add_insecure_port(f'[::]:{PORT}')
    server.start()
    print(f"[Learner] gRPC server running on port {PORT}")
    server.wait_for_termination()


if __name__ == "__main__":
    print("[Learner] Starting gRPC server...")
    serve()