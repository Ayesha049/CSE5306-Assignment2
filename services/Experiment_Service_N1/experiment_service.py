"""
N1: Experiment Service
──────────────────────
System entry point — receives experiment requests from users,
orchestrates N2 (Environment), and exposes status via N6 (Analytics).

Ports / roles:
  • Listens on :50053
  • User calls StartExperiment / StopExperiment / GetExperimentStatus
  • N1 calls N2.Initialize + N2.StartRollout to begin sampling
  • N1 queries N6.QueryMetrics to enrich GetExperimentStatus responses

The N3→N4 training chain runs automatically once N2 starts pushing
transitions; N1 does NOT need to drive the training loop.

gRPC interface (from services/Experiment_Service(N1)/experiment.proto):
  StartExperiment    (ExperimentConfig) → Status
  StopExperiment     (ExperimentConfig) → Status
  GetExperimentStatus(ExperimentConfig) → Status
"""

import sys
import os
import socket
import itertools
import grpc
import threading
import logging
from concurrent import futures

import sys
sys.path.append("/app/stubs")  # path inside Docker container

import common_pb2              # ExperimentConfig, Status
import experiment_pb2_grpc     # ExperimentServiceServicer
import environment_pb2_grpc    # EnvironmentServiceStub  (→ N2)
import analytics_pb2           # QueryRequest
import analytics_pb2_grpc      # AnalyticsServiceStub   (→ N6, optional)

# ── logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[N1-Experiment] %(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("N1-Experiment")

# ── service addresses ──────────────────────────────────────────────────────────
N2_SERVICE = "environment"    # Environment Service (supports multiple replicas via --scale)
N2_PORT    = 50050
N6_ADDR    = "analytics:50052"  # Analytics Service
PORT       = 50053


def discover_n2_hosts() -> list[str]:
    """
    Discover all running N2 (Environment) container IPs via Docker's embedded DNS.

    When 'environment' is scaled to N replicas with:
        docker-compose up --scale environment=N
    Docker DNS resolves 'environment' to all N container IPs, enabling N1 to
    distribute experiments across replicas via round-robin.

    Falls back to a single "environment:50050" entry if DNS resolution fails.
    """
    try:
        results = socket.getaddrinfo(N2_SERVICE, N2_PORT, socket.AF_INET, socket.SOCK_STREAM)
        ips = list({r[4][0] for r in results})   # deduplicate
        if ips:
            hosts = [f"{ip}:{N2_PORT}" for ip in sorted(ips)]
            logger.info(f"[N1] Discovered N2 instances: {hosts}")
            return hosts
    except socket.gaierror:
        pass
    return [f"{N2_SERVICE}:{N2_PORT}"]  # fallback: single instance


# ── experiment state tracker ───────────────────────────────────────────────────

class ExperimentManager:
    """Thread-safe registry of known experiments.
    used to collect all the exps
    """

    def __init__(self):
        self._store: dict[str, dict] = {}   # experiment_id → {config, status}
        self._lock = threading.Lock()
        self._exp_host: dict[str, str] = {}  # experiment_id → assigned N2 host

    def register(self, config, status: str = "running") -> None:
        with self._lock:
            self._store[config.experiment_id] = {
                "config": config,
                "status": status,
            }

    def update_status(self, experiment_id: str, status: str) -> None:
        with self._lock:
            if experiment_id in self._store:
                self._store[experiment_id]["status"] = status

    def get_status(self, experiment_id: str) -> str:
        with self._lock:
            entry = self._store.get(experiment_id)
            return entry["status"] if entry else "not_found"

    def is_running(self, experiment_id: str) -> bool:
        return self.get_status(experiment_id) == "running"

    def set_host(self, experiment_id: str, host: str) -> None:
        """Record which N2 instance owns this experiment."""
        with self._lock:
            self._exp_host[experiment_id] = host

    def get_host(self, experiment_id: str) -> str | None:
        """Return the N2 host assigned to this experiment, or None if unknown."""
        with self._lock:
            return self._exp_host.get(experiment_id)


# ── gRPC servicer ──────────────────────────────────────────────────────────────

class ExperimentServicer(experiment_pb2_grpc.ExperimentServiceServicer):

    def __init__(self):
        self._manager  = ExperimentManager()
        self._rr_lock  = threading.Lock()
        # Discover N2 instances at startup; refreshed on each StartExperiment
        # so that newly scaled-up replicas are picked up automatically.
        self._n2_hosts = discover_n2_hosts()
        self._n2_cycle = itertools.cycle(self._n2_hosts)

    def _next_n2_host(self) -> str:
        """
        Return the next N2 host in round-robin order (thread-safe).
        Re-discovers N2 instances on each call so scaling changes are reflected
        without restarting N1.
        """
        with self._rr_lock:
            fresh = discover_n2_hosts()
            if set(fresh) != set(self._n2_hosts):
                logger.info(f"[N1] N2 instance list changed: {self._n2_hosts} → {fresh}")
                self._n2_hosts = fresh
                self._n2_cycle = itertools.cycle(fresh)
            return next(self._n2_cycle)

    # ── user-facing RPCs ───────────────────────────────────────────────────────

    def StartExperiment(self, request, context):
        """
        1. Call N2.Initialize(config)    — create the RL environment
        2. Call N2.StartRollout(config)  — begin collecting (s,a,r,s') tuples
        The downstream N3→N4→N5 chain then runs automatically.
        """
        exp_id = request.experiment_id
        print(f"[{exp_id}] StartExperiment called with env='{request.env_name}' algo='{request.algorithm}' seed={request.seed}")

        if self._manager.is_running(exp_id):
            return common_pb2.Status(
                ok=False, message=f"Experiment '{exp_id}' is already running"
            )

        logger.info(
            f"[{exp_id}] Starting — env={request.env_name!r} "
            f"algo={request.algorithm!r} seed={request.seed}"
        )

        # ── select N2 via round-robin and contact it ───────────────────────────
        n2_host     = self._next_n2_host()
        logger.info(f"[{exp_id}] Routing to N2 host: {n2_host}")
        env_channel = grpc.insecure_channel(n2_host)
        env_stub    = environment_pb2_grpc.EnvironmentServiceStub(env_channel)

        try:
            print(f"[{exp_id}] Calling N2.Initialize on {n2_host} ...")
            resp = env_stub.Initialize(request, timeout=10)
            if not resp.ok:
                return common_pb2.Status(
                    ok=False, message=f"N2 Initialize failed: {resp.message}"
                )
            logger.info(f"[{exp_id}] N2 initialized")
        except grpc.RpcError as e:
            msg = f"Cannot reach N2 at {n2_host}: {e.details()}"
            logger.error(msg)
            return common_pb2.Status(ok=False, message=msg)

        try:
            resp = env_stub.StartRollout(request, timeout=10)
            if not resp.ok:
                return common_pb2.Status(
                    ok=False, message=f"N2 StartRollout failed: {resp.message}"
                )
            logger.info(f"[{exp_id}] N2 rollout started")
        except grpc.RpcError as e:
            msg = f"N2 StartRollout error: {e.details()}"
            logger.error(msg)
            return common_pb2.Status(ok=False, message=msg)

        # Record which N2 owns this experiment (needed for StopRollout routing)
        self._manager.set_host(exp_id, n2_host)
        self._manager.register(request, status="running")
        return common_pb2.Status(ok=True, message=f"Experiment '{exp_id}' started")

    def StopExperiment(self, request, context):
        """Signal N2 to halt the rollout, then mark experiment as stopped."""
        exp_id = request.experiment_id

        # Route StopRollout to the specific N2 that owns this experiment.
        # Falls back to the first known N2 if the mapping was lost (e.g. N1 restart).
        n2_host = self._manager.get_host(exp_id)
        if n2_host is None:
            n2_host = self._n2_hosts[0] if self._n2_hosts else f"{N2_SERVICE}:{N2_PORT}"
            logger.warning(f"[{exp_id}] No N2 host mapping found; falling back to {n2_host}")
        env_channel = grpc.insecure_channel(n2_host)
        env_stub    = environment_pb2_grpc.EnvironmentServiceStub(env_channel)

        try:
            env_stub.StopRollout(request, timeout=10)
            logger.info(f"[{exp_id}] N2 rollout stopped on {n2_host}")
        except grpc.RpcError as e:
            logger.warning(f"[{exp_id}] Could not stop N2 rollout: {e.details()}")

        self._manager.update_status(exp_id, "stopped")
        return common_pb2.Status(ok=True, message=f"Experiment '{exp_id}' stopped")

    def GetExperimentStatus(self, request, context):
        """
        Return current status string.
        Also queries N6 for the latest episode_reward metric if available.
        """
        exp_id = request.experiment_id
        status = self._manager.get_status(exp_id)
        detail = ""

        # Optional: enrich with latest metric from N6
        try:
            analytics_channel = grpc.insecure_channel(N6_ADDR)
            analytics_stub = analytics_pb2_grpc.AnalyticsServiceStub(analytics_channel)
            result = analytics_stub.QueryMetrics(
                analytics_pb2.QueryRequest(
                    experiment_id=exp_id,
                    metric_name="episode_reward",
                ),
                timeout=3,
            )
            if result.metrics:
                latest = result.metrics[-1]
                detail = (
                    f" | latest episode_reward={latest.value:.2f}"
                    f" (step {latest.step})"
                )
        except Exception:
            pass  # N6 is optional for status queries

        return common_pb2.Status(ok=True, message=f"status={status}{detail}")


# ── server entry point ─────────────────────────────────────────────────────────
def auto_start_experiment():
    import time
    time.sleep(5)  # wait for other containers (N2, N6) to be ready

    logger.info("[AUTO] Triggering StartExperiment...")

    channel = grpc.insecure_channel(f"localhost:{PORT}")
    stub = experiment_pb2_grpc.ExperimentServiceStub(channel)

    config = common_pb2.ExperimentConfig(
        experiment_id="auto_exp",
        env_name="CartPole",
        algorithm="DQN",
        seed=42,
        max_steps=10
    )

    try:
        response = stub.StartExperiment(config)
        logger.info(f"[AUTO] Response: {response.message}")
    except grpc.RpcError as e:
        logger.error(f"[AUTO] Failed to call StartExperiment: {e.details()}")

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    experiment_pb2_grpc.add_ExperimentServiceServicer_to_server(
        ExperimentServicer(), server
    )

    server.add_insecure_port(f"[::]:{PORT}")
    server.start()
    logger.info(f"N1 Experiment Service started on port {PORT}")

    # 🔥 Start auto-client thread
    threading.Thread(target=auto_start_experiment, daemon=True).start()

    server.wait_for_termination()


if __name__ == "__main__":
    serve()
