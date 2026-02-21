import sys
sys.path.append("/app/stubs")  # path inside Docker container

import grpc
import time
import random
from concurrent import futures

import buffer_pb2
import buffer_pb2_grpc
import analytics_pb2
import analytics_pb2_grpc
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Config: gRPC endpoints
BUFFER_HOSTS = ["buffer:50051"]  # can add more for load balancing
ANALYTICS_HOST = "analytics:50052"

class LearnerService:
    def __init__(self):
        # Connect to Analytics service
        self.analytics_channel = grpc.insecure_channel(ANALYTICS_HOST)
        self.analytics_stub = analytics_pb2_grpc.AnalyticsServiceStub(self.analytics_channel)

    def sample_batch(self):
        # Pick a buffer instance (simple random for now)
        buffer_host = random.choice(BUFFER_HOSTS)
        channel = grpc.insecure_channel(buffer_host)
        stub = buffer_pb2_grpc.BufferServiceStub(channel)

        request = buffer_pb2.SampleRequest(batch_size=32)
        batch_response = stub.SampleBatch(request)
        print(f"[Learner] Sampled batch of size {len(batch_response.transitions)}")
        return batch_response.transitions

    def train_policy(self, batch):
        # Dummy training step: compute "loss" as random number
        loss = random.random()
        print(f"[Learner] Training step complete, loss={loss:.4f}")
        return loss

    def send_metric(self, metric_name, value, step):
        metric = analytics_pb2.Metrics(
            experiment_id="exp_001",
            source_node="N4-Learner",
            metric_name=metric_name,
            value=value,
            step=step
        )
        response = self.analytics_stub.ReportMetrics(metric)
        print(f"[Learner] Metric sent: {metric_name}={value} ok={response.ok}")

    def run_training_loop(self, steps=100):
        for step in range(1, steps+1):
            batch = self.sample_batch()
            loss = self.train_policy(batch)
            self.send_metric("loss", loss, step)
            time.sleep(0.1)  # simulate time between steps

    # def run_training_loop(self, steps=100):
    #     for step in range(1, steps+1):

    #         if rank == 0:
    #             # Only master calls buffer
    #             batch = self.sample_batch()
    #         else:
    #             batch = None

    #         # Broadcast batch to all workers
    #         batch = comm.bcast(batch, root=0)

    #         # Each worker computes loss
    #         local_loss = self.train_policy(batch)

    #         # Aggregate losses (average)
    #         total_loss = comm.reduce(local_loss, op=MPI.SUM, root=0)

    #         if rank == 0:
    #             avg_loss = total_loss / size
    #             print(f"[Learner-MPI] Avg loss={avg_loss:.4f}")
    #             self.send_metric("loss", avg_loss, step)


def serve():
    learner = LearnerService()
    print("[Learner] Starting training loop")
    learner.run_training_loop(steps=50)
    print("[Learner] Training loop complete")


if __name__ == "__main__":
    serve()