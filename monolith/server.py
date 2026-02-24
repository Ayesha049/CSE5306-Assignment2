"""
Monolithic Server
─────────────────
• Exposes ONLY ExperimentService (N1) via gRPC
• Internally wires:
    N2 → EnvironmentService
    N3 → BufferService
    N4 → LearnerService
    N5 → PolicyService
    N6 → AnalyticsService
"""

import sys
import os
import grpc
from concurrent import futures

# Make sure stubs are importable
sys.path.append(os.path.join(os.path.dirname(__file__), "stubs"))

import experiment_pb2_grpc

# Import monolithic services
from services.analytics_service import AnalyticsService
from services.policy_service import PolicyService
from services.learner_service import LearnerService
from services.buffer_service import BufferService
from services.environment_service import EnvironmentServicer
from services.experiment_service import ExperimentServicer


PORT = 50053


# ════════════════════════════════════════════════════════════════════════════════
#  Dependency Wiring
# ════════════════════════════════════════════════════════════════════════════════

def build_monolith():
    """
    Build full dependency graph in correct order.
    """

    # N6
    analytics_service = AnalyticsService()

    # N5
    policy_service = PolicyService()

    # N4 (needs buffer injected later)
    learner_service = LearnerService(
        buffer_service=None,
        policy_service=policy_service,
        analytics_service=analytics_service
    )

    # N3
    buffer_service = BufferService(
        learner_service=learner_service,
        analytics_service=analytics_service
    )

    # Fix circular dependency
    learner_service.buffer_service = buffer_service

    # N2
    environment_service = EnvironmentServicer(
        buffer_service=buffer_service,
        policy_service=policy_service,
        analytics_service=analytics_service
    )

    # N1 (external entry point)
    experiment_service = ExperimentServicer(
        env_service=environment_service,
        analytics_service=analytics_service
    )

    return experiment_service


# ════════════════════════════════════════════════════════════════════════════════
#  gRPC Server (Only for Experiment Service)
# ════════════════════════════════════════════════════════════════════════════════

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=20))

    experiment_service = build_monolith()

    experiment_pb2_grpc.add_ExperimentServiceServicer_to_server(
        experiment_service,
        server
    )

    server.add_insecure_port(f"[::]:{PORT}")
    server.start()

    print(f"[Monolith] Experiment Service running on port {PORT}")
    print("[Monolith] All services running inside single process")

    server.wait_for_termination()


if __name__ == "__main__":
    serve()