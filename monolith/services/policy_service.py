"""
N5: Policy Service (Monolithic)
──────────────────────────────
• Stores and serves RL policy weights
• Called directly by:
    - LearnerService (UpdatePolicy)
    - EnvironmentService (GetPolicy)
• No gRPC
"""

import pickle
import numpy as np
import threading
import logging

import common_pb2
import policy_pb2


# ── logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[N5-Policy-Mono] %(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("N5-Policy-Mono")


# ── default dimensions (CartPole fallback) ─────────────────────────────────────
DEFAULT_OBS_DIM   = 4
DEFAULT_HIDDEN    = 16
DEFAULT_N_ACTIONS = 2


def _random_weights(obs_dim=DEFAULT_OBS_DIM,
                    hidden=DEFAULT_HIDDEN,
                    n_actions=DEFAULT_N_ACTIONS,
                    seed=42) -> bytes:
    """Return pickle-serialised random MLP weights for initial policy."""
    rng = np.random.RandomState(seed)

    weights = {
        "W1": rng.randn(obs_dim, hidden).astype(np.float32) * 0.1,
        "b1": np.zeros(hidden, dtype=np.float32),
        "W2": rng.randn(hidden, n_actions).astype(np.float32) * 0.1,
        "b2": np.zeros(n_actions, dtype=np.float32),
    }

    return pickle.dumps(weights)


# ════════════════════════════════════════════════════════════════════════════════
#  Policy Store
# ════════════════════════════════════════════════════════════════════════════════

class PolicyStore:
    """
    Thread-safe in-memory policy storage.

    Layout:
        experiment_id → [Policy v0, v1, v2, ...]
    """

    def __init__(self):
        self._store = {}
        self._lock = threading.RLock()

    def update(self, policy: common_pb2.Policy):
        """Append new policy version."""
        with self._lock:
            exp_id = policy.experiment_id

            if exp_id not in self._store:
                self._store[exp_id] = []

            self._store[exp_id].append(policy)

            logger.info(
                f"[{exp_id}] Stored policy v{policy.version} "
                f"(total versions={len(self._store[exp_id])})"
            )

    def get_latest(self, experiment_id: str, seed: int = 42):
        """
        Return latest policy.
        Auto-generate v0 if none exists.
        """
        with self._lock:
            policies = self._store.get(experiment_id, [])

            if policies:
                return policies[-1]

            # auto seed initial random policy
            logger.info(f"[{experiment_id}] No policy found — auto-seeding v0")

            initial = common_pb2.Policy(
                experiment_id=experiment_id,
                version=0,
                weights=_random_weights(seed=seed),
            )

            self._store[experiment_id] = [initial]
            return initial

    def get_by_version(self, experiment_id: str, version: int):
        with self._lock:
            for p in self._store.get(experiment_id, []):
                if p.version == version:
                    return p
            return None


# ════════════════════════════════════════════════════════════════════════════════
#  Policy Service (Monolithic API)
# ════════════════════════════════════════════════════════════════════════════════

class PolicyService:

    def __init__(self):
        self._store = PolicyStore()

    # same name as gRPC for compatibility
    def GetPolicy(self, request: policy_pb2.PolicyRequest):
        return self._store.get_latest(request.experiment_id)

    def UpdatePolicy(self, request: common_pb2.Policy):
        self._store.update(request)

        return common_pb2.Status(
            ok=True,
            message=f"Policy v{request.version} stored for {request.experiment_id}"
        )

    def GetPolicyVersion(self, request: policy_pb2.PolicyRequest):
        return self._store.get_latest(request.experiment_id)