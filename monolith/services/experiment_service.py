"""
N1: Experiment Service (Monolithic Version)
───────────────────────────────────────────
Changes:
  • No gRPC calls to N2/N6
  • Direct function calls instead
  • Still exposes ONE gRPC interface externally (optional but recommended)
"""

import sys
import threading
import logging
from concurrent import futures
import grpc

sys.path.append("/app/stubs")

import common_pb2
import experiment_pb2_grpc
import analytics_pb2  # still used for request structure


# ── logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[N1-Monolith] %(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("N1-Monolith")

PORT = 50053


# ── experiment state tracker ───────────────────────────────────────────────────

class ExperimentManager:
    def __init__(self):
        self._store = {}
        self._lock = threading.Lock()

    def register(self, config, status="running"):
        with self._lock:
            self._store[config.experiment_id] = {
                "config": config,
                "status": status,
            }

    def update_status(self, experiment_id, status):
        with self._lock:
            if experiment_id in self._store:
                self._store[experiment_id]["status"] = status

    def get_status(self, experiment_id):
        with self._lock:
            entry = self._store.get(experiment_id)
            return entry["status"] if entry else "not_found"

    def is_running(self, experiment_id):
        return self.get_status(experiment_id) == "running"


# ── Monolithic Servicer ────────────────────────────────────────────────────────

class ExperimentServicer(experiment_pb2_grpc.ExperimentServiceServicer):

    def __init__(self, env_service, analytics_service=None):
        """
        env_service        → direct instance of N2 logic
        analytics_service  → direct instance of N6 logic (optional)
        """
        self._manager = ExperimentManager()
        self.env_service = env_service
        self.analytics_service = analytics_service

    # ── user-facing RPCs ───────────────────────────────────────────────────────

    def StartExperiment(self, request, context):
        exp_id = request.experiment_id

        if self._manager.is_running(exp_id):
            return common_pb2.Status(
                ok=False, message=f"Experiment '{exp_id}' is already running"
            )

        logger.info(f"[{exp_id}] Starting (MONOLITH)")

        # ✅ DIRECT CALL instead of gRPC
        resp = self.env_service.Initialize(request)
        if not resp.ok:
            return common_pb2.Status(
                ok=False, message=f"N2 Initialize failed: {resp.message}"
            )

        resp = self.env_service.StartRollout(request)
        if not resp.ok:
            return common_pb2.Status(
                ok=False, message=f"N2 StartRollout failed: {resp.message}"
            )

        self._manager.register(request, status="running")
        return common_pb2.Status(ok=True, message=f"Experiment '{exp_id}' started")

    def StopExperiment(self, request, context):
        exp_id = request.experiment_id

        try:
            # ✅ DIRECT CALL
            self.env_service.StopRollout(request)
            logger.info(f"[{exp_id}] Rollout stopped (MONOLITH)")
        except Exception as e:
            logger.warning(f"[{exp_id}] Stop failed: {str(e)}")

        self._manager.update_status(exp_id, "stopped")
        return common_pb2.Status(ok=True, message=f"Experiment '{exp_id}' stopped")

    def GetExperimentStatus(self, request, context):
        exp_id = request.experiment_id
        status = self._manager.get_status(exp_id)
        detail = ""

        # ✅ DIRECT CALL instead of gRPC
        if self.analytics_service:
            try:
                result = self.analytics_service.QueryMetrics(
                    analytics_pb2.QueryRequest(
                        experiment_id=exp_id,
                        metric_name="episode_reward",
                    )
                )
                if result.metrics:
                    latest = result.metrics[-1]
                    detail = (
                        f" | latest episode_reward={latest.value:.2f}"
                        f" (step {latest.step})"
                    )
            except Exception:
                pass

        return common_pb2.Status(ok=True, message=f"status={status}{detail}")


# ── wiring (IMPORTANT) ─────────────────────────────────────────────────────────

def build_monolith():
    """
    This replaces Docker networking.
    You manually connect all services here.
    """

    # ⬇️ import your actual service implementations
    from services.environment_service import EnvironmentService
    from services.analytics_service import AnalyticsService

    analytics_service = AnalyticsService()
    env_service = EnvironmentService(analytics_service=analytics_service)

    experiment_service = ExperimentServicer(
        env_service=env_service,
        analytics_service=analytics_service
    )

    return experiment_service


# ── server entry point ─────────────────────────────────────────────────────────

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    experiment_service = build_monolith()

    experiment_pb2_grpc.add_ExperimentServiceServicer_to_server(
        experiment_service, server
    )

    server.add_insecure_port(f"[::]:{PORT}")
    server.start()
    logger.info(f"Monolithic service running on port {PORT}")

    server.wait_for_termination()


if __name__ == "__main__":
    serve()