"""
N5: Policy Management Service
──────────────────────────────
Stores and serves RL policy weights.

Ports / callers:
  • Listens on :50056
  • N4 (Learner) calls UpdatePolicy()  → push trained weights
  • N2 (Environment) calls GetPolicy() → fetch latest weights for action selection

gRPC interface (from services/Policy_Service(N5)/policy.proto):
  GetPolicy(PolicyRequest)        → Policy   # latest weights for experiment
  UpdatePolicy(Policy)            → Status   # store new weights (N4)
  GetPolicyVersion(PolicyRequest) → Policy   # same as GetPolicy (version in message)
"""

import sys
import os
import grpc
import pickle
import numpy as np
import threading
import logging
from concurrent import futures

# ── stubs ──────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../stubs"))

import common_pb2         # Policy, Status  (from common.proto)
import policy_pb2         # PolicyRequest   (from policy.proto)
import policy_pb2_grpc    # PolicyServiceServicer

# ── logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[N5-Policy] %(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("N5-Policy")

# ── default network dimensions (CartPole fallback) ─────────────────────────────
DEFAULT_OBS_DIM  = 4
DEFAULT_HIDDEN   = 16
DEFAULT_N_ACTIONS = 2
PORT = 50056


def _random_weights(obs_dim: int = DEFAULT_OBS_DIM, # TODO discuss with Ayesha about the weights
                    hidden: int = DEFAULT_HIDDEN,
                    n_actions: int = DEFAULT_N_ACTIONS,
                    seed: int = 42) -> bytes:
    """Return pickle-serialised random MLP weights for the initial policy."""
    rng = np.random.RandomState(seed)
    weights = {
        "W1": rng.randn(obs_dim, hidden).astype(np.float32) * 0.1,
        "b1": np.zeros(hidden, dtype=np.float32),
        "W2": rng.randn(hidden, n_actions).astype(np.float32) * 0.1,
        "b2": np.zeros(n_actions, dtype=np.float32),
    }
    return pickle.dumps(weights)


# ── policy store ───────────────────────────────────────────────────────────────

class PolicyStore:
    """
    Thread-safe in-memory store.
    Layout:  experiment_id  →  list[Policy]   (appended in version order)
    """

    def __init__(self):
        self._store: dict[str, list] = {}   # experiment_id → [Policy v0, v1, …]
        self._lock = threading.RLock()

    def update(self, policy) -> None:
        """Append a newly trained policy (called by N4 via UpdatePolicy)."""
        with self._lock:
            exp_id = policy.experiment_id
            if exp_id not in self._store:
                self._store[exp_id] = []
            self._store[exp_id].append(policy)
            logger.info(
                f"[{exp_id}] Stored policy v{policy.version} "
                f"(total versions: {len(self._store[exp_id])})"
            )

    def get_latest(self, experiment_id: str, seed: int = 42):
        """
        Return the latest policy for an experiment.
        If no policy exists yet, auto-generate a random v0 so N2 can start
        collecting experience before N4 has trained anything.
        """
        with self._lock:
            policies = self._store.get(experiment_id, [])
            if policies:
                return policies[-1]

            # Auto-seed random initial policy (version 0)
            logger.info(
                f"[{experiment_id}] No policy found — auto-seeding random v0"
            )
            initial = common_pb2.Policy(
                experiment_id=experiment_id,
                version=0,
                weights=_random_weights(seed=seed),
            )
            self._store[experiment_id] = [initial]
            return initial

    def get_by_version(self, experiment_id: str, version: int):
        """Return the policy at a specific version number, or None."""
        with self._lock:
            for p in self._store.get(experiment_id, []):
                if p.version == version:
                    return p
            return None


# ── gRPC servicer ──────────────────────────────────────────────────────────────

class PolicyServicer(policy_pb2_grpc.PolicyServiceServicer):

    def __init__(self):
        self._store = PolicyStore()

    # ── N2 calls this ──────────────────────────────────────────────────────────
    def GetPolicy(self, request, context):
        """Return the latest policy weights for action selection."""
        return self._store.get_latest(request.experiment_id)

    # ── N4 calls this ──────────────────────────────────────────────────────────
    def UpdatePolicy(self, request, context):
        """Store newly trained policy weights pushed by the Learner."""
        self._store.update(request)
        return common_pb2.Status(
            ok=True,
            message=f"Policy v{request.version} stored for {request.experiment_id}",
        )

    # ── utility ────────────────────────────────────────────────────────────────
    def GetPolicyVersion(self, request, context):
        """Return the current latest policy (version info is inside the message)."""
        return self._store.get_latest(request.experiment_id)


# ── server entry point ─────────────────────────────────────────────────────────

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    policy_pb2_grpc.add_PolicyServiceServicer_to_server(PolicyServicer(), server)
    server.add_insecure_port(f"[::]:{PORT}")
    server.start()
    logger.info(f"N5 Policy Service started on port {PORT}")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
