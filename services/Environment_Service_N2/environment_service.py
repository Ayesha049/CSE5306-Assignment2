"""
N2: Environment Service
───────────────────────
Central sampling hub — runs RL environments, queries N5 for the latest
policy, generates (s, a, r, s') transitions, and pushes them to N3.

Ports / callers:
  • Listens on :50054
  • N1 calls Initialize + StartRollout / StopRollout
  • N2 calls N5.GetPolicy      (:50056)  — fetch latest weights
  • N2 calls N3.PushTransition (:50051)  — push experience
  • N2 calls N6.ReportMetrics  (:50052)  — report episode reward (optional)

Environment support:
  • Built-in CartPole (pure numpy, no external deps)
  • Any gymnasium environment (e.g. "LunarLander-v3", "MountainCar-v0")
    — falls back to CartPole if gymnasium is not installed or env unknown.

Transition serialisation to N3 (buffer_pb2.Transition):
  state / next_state  →  JSON string  (json.dumps(obs.tolist()))
  action              →  string       (str(int(action)))
  reward              →  float
  done                →  bool

gRPC interface (from services/Environment_Service(N2)/environment.proto):
  Initialize  (ExperimentConfig) → Status
  StartRollout(ExperimentConfig) → Status
  StopRollout (ExperimentConfig) → Status
"""
import sys
sys.path.append("/app/stubs")  # path inside Docker container

import sys
import os
import json
import grpc
import pickle
import numpy as np
import threading
import time
import logging
from concurrent import futures

# # ── stubs ──────────────────────────────────────────────────────────────────────
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../stubs"))

import common_pb2             # ExperimentConfig, Status, Policy
import environment_pb2_grpc   # EnvironmentServiceServicer
import policy_pb2             # PolicyRequest
import policy_pb2_grpc        # PolicyServiceStub  (→ N5)
import buffer_pb2             # Transition, PushRequest, SampleRequest
import buffer_pb2_grpc        # BufferServiceStub  (→ N3)
import analytics_pb2          # Metrics
import analytics_pb2_grpc     # AnalyticsServiceStub (→ N6, optional)

# ── logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[N2-Environment] %(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("N2-Environment")

# ── service addresses ──────────────────────────────────────────────────────────
N3_ADDR = "buffer:50051"   # Buffer Service
N5_ADDR = "policy:50056"   # Policy Service
N6_ADDR = "analytics:50052"   # Analytics Service
PORT    = 50050

# ── rollout hyper-parameters ───────────────────────────────────────────────────
POLICY_REFRESH_STEPS = 50     # re-fetch policy from N5 every N environment steps
EPSILON_START        = 1.0    # initial exploration rate
EPSILON_END          = 0.05   # minimum exploration rate
EPSILON_DECAY        = 0.995  # per-episode multiplicative decay
STEP_DELAY           = 0.005  # seconds between steps (throttle data collection)


# ════════════════════════════════════════════════════════════════════════════════
#  Built-in CartPole (no gymnasium required)
# ════════════════════════════════════════════════════════════════════════════════

class CartPoleEnv:
    """
    Minimal CartPole-v1 simulation using only numpy.
    Physics matches the OpenAI / Gymnasium CartPole implementation.
    """

    obs_dim   = 4
    n_actions = 2

    # physics constants
    GRAVITY       = 9.8
    MASSCART      = 1.0
    MASSPOLE      = 0.1
    POLE_LEN      = 0.5        # half-pole length
    FORCE_MAG     = 10.0
    TAU           = 0.02       # seconds per step
    THETA_THRESH  = 12 * np.pi / 180   # ±12°
    X_THRESH      = 2.4

    def __init__(self, seed: int = 42, max_steps: int = 500):
        self.max_steps  = max_steps
        self.rng        = np.random.RandomState(seed)
        self.state      = None
        self._step_cnt  = 0
        self._total_mass    = self.MASSCART + self.MASSPOLE
        self._pole_mass_len = self.MASSPOLE * self.POLE_LEN

    def reset(self) -> np.ndarray:
        self.state    = self.rng.uniform(-0.05, 0.05, 4).astype(np.float32)
        self._step_cnt = 0
        return self.state.copy()

    def step(self, action: int):
        x, x_dot, theta, theta_dot = self.state
        force     = self.FORCE_MAG if action == 1 else -self.FORCE_MAG
        costheta  = np.cos(theta)
        sintheta  = np.sin(theta)
        temp      = (force + self._pole_mass_len * theta_dot ** 2 * sintheta) / self._total_mass
        theta_acc = (
            self.GRAVITY * sintheta - costheta * temp
        ) / (self.POLE_LEN * (4 / 3 - self.MASSPOLE * costheta ** 2 / self._total_mass))
        x_acc = temp - self._pole_mass_len * theta_acc * costheta / self._total_mass
        x         += self.TAU * x_dot
        x_dot     += self.TAU * x_acc
        theta     += self.TAU * theta_dot
        theta_dot += self.TAU * theta_acc
        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        self._step_cnt += 1
        done = bool(
            x         < -self.X_THRESH or x         > self.X_THRESH
            or theta  < -self.THETA_THRESH or theta  > self.THETA_THRESH
            or self._step_cnt >= self.max_steps
        )
        return self.state.copy(), 1.0, done


# ════════════════════════════════════════════════════════════════════════════════
#  Gymnasium wrapper (discrete action spaces)
# ════════════════════════════════════════════════════════════════════════════════

class GymnasiumWrapper:
    """Thin adapter around a gymnasium environment (discrete actions only)."""

    def __init__(self, env, seed: int):
        self._env     = env
        self._seed    = seed
        self.obs_dim  = env.observation_space.shape[0]
        self.n_actions = env.action_space.n

    def reset(self) -> np.ndarray:
        obs, _ = self._env.reset(seed=self._seed)
        return np.array(obs, dtype=np.float32)

    def step(self, action: int):
        obs, reward, terminated, truncated, _ = self._env.step(action)
        done = terminated or truncated
        return np.array(obs, dtype=np.float32), float(reward), bool(done)


# ════════════════════════════════════════════════════════════════════════════════
#  Environment factory
# ════════════════════════════════════════════════════════════════════════════════

_CARTPOLE_NAMES = {"CartPole", "cartpole", "CartPole-v0", "CartPole-v1"}


def make_env(env_name: str, seed: int, max_steps: int):
    """
    Create an environment by name.
    Priority:
      1. If env_name is a CartPole variant (or empty) → built-in CartPoleEnv
      2. Otherwise try gymnasium.make(env_name)
      3. Fall back to CartPoleEnv on any error
    """
    max_steps = max_steps or 500

    if not env_name or env_name in _CARTPOLE_NAMES:
        logger.info(f"Using built-in CartPole (seed={seed}, max_steps={max_steps})")
        return CartPoleEnv(seed=seed, max_steps=max_steps)

    try:
        import gymnasium as gym
        env = gym.make(env_name, max_episode_steps=max_steps)
        wrapper = GymnasiumWrapper(env, seed)
        logger.info(
            f"Using gymnasium env '{env_name}' "
            f"(obs_dim={wrapper.obs_dim}, n_actions={wrapper.n_actions})"
        )
        return wrapper
    except Exception as exc:
        logger.warning(
            f"Cannot load gymnasium env '{env_name}': {exc}. "
            "Falling back to built-in CartPole."
        )
        return CartPoleEnv(seed=seed, max_steps=max_steps)


# ════════════════════════════════════════════════════════════════════════════════
#  Policy utilities
# ════════════════════════════════════════════════════════════════════════════════

def _select_action(state: np.ndarray, weights: dict,
                   epsilon: float, rng, n_actions: int) -> int:
    """
    Epsilon-greedy action using a 2-layer MLP:
      h       = ReLU(state @ W1 + b1)
      q_values = h @ W2 + b2
      action   = argmax(q_values)   (with probability 1−ε)
               = random             (with probability ε)
    Falls back to random if weight shapes are incompatible.
    """
    if rng.random() < epsilon:
        return int(rng.randint(0, n_actions))
    try:
        h = np.maximum(0.0, state @ weights["W1"] + weights["b1"])
        q = h @ weights["W2"] + weights["b2"]
        return int(np.argmax(q))
    except Exception:
        return int(rng.randint(0, n_actions))


# ════════════════════════════════════════════════════════════════════════════════
#  Rollout runner  (one background thread per experiment)
# ════════════════════════════════════════════════════════════════════════════════

class RolloutRunner:

    def __init__(self, config):
        self.config     = config
        self.exp_id     = config.experiment_id
        self.env        = make_env(config.env_name, config.seed, config.max_steps)
        self.stop_event = threading.Event()
        self.thread     = None
        self.rng        = np.random.RandomState(config.seed)
        self.epsilon    = EPSILON_START

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(
            target=self._run, daemon=True, name=f"rollout-{self.exp_id}"
        )
        self.thread.start()

    def stop(self):
        self.stop_event.set()

    # ── main loop ─────────────────────────────────────────────────────────────
    def _run(self):
        # create gRPC stubs once per thread
        policy_stub   = policy_pb2_grpc.PolicyServiceStub(
            grpc.insecure_channel(N5_ADDR)
        )
        buffer_stub   = buffer_pb2_grpc.BufferServiceStub(
            grpc.insecure_channel(N3_ADDR)
        )
        analytics_stub = analytics_pb2_grpc.AnalyticsServiceStub(
            grpc.insecure_channel(N6_ADDR)
        )

        policy_req = policy_pb2.PolicyRequest(experiment_id=self.exp_id)

        weights         = None
        current_version = 0
        state           = self.env.reset()
        episode_reward  = 0.0
        episode         = 0
        global_step     = 0

        logger.info(
            f"[{self.exp_id}] Rollout started "
            f"(env={self.config.env_name or 'CartPole'}, "
            f"n_actions={self.env.n_actions})"
        )

        while not self.stop_event.is_set():

            # ── refresh policy from N5 every POLICY_REFRESH_STEPS steps ───────
            if global_step % POLICY_REFRESH_STEPS == 0:
                try:
                    policy = policy_stub.GetPolicy(policy_req, timeout=3)
                    weights = pickle.loads(policy.weights)
                    current_version = policy.version
                except Exception as exc:
                    logger.warning(
                        f"[{self.exp_id}] Policy fetch failed: {exc} — using random action"
                    )

            # ── choose action ─────────────────────────────────────────────────
            if weights is not None:
                action = _select_action(
                    state, weights, self.epsilon, self.rng, self.env.n_actions
                )
            else:
                action = int(self.rng.randint(0, self.env.n_actions))

            # ── step environment ──────────────────────────────────────────────
            next_state, reward, done = self.env.step(action)
            episode_reward += reward

            # ── push transition to N3 ─────────────────────────────────────────
            # N3 buffer_pb2.Transition uses STRING fields, not bytes.
            # State arrays are JSON-serialised so they are human-readable and
            # can be easily parsed by N4 with json.loads() + np.array().
            try:
                transition = buffer_pb2.Transition(
                    state      = json.dumps(state.tolist()),
                    action     = str(int(action)),
                    reward     = float(reward),
                    next_state = json.dumps(next_state.tolist()),
                    done       = bool(done),
                )
                buffer_stub.PushTransition(
                    buffer_pb2.PushRequest(transition=transition),
                    timeout=3,
                )
            except Exception as exc:
                logger.warning(f"[{self.exp_id}] Buffer push failed: {exc}")

            global_step += 1

            # ── episode bookkeeping ───────────────────────────────────────────
            if done:
                logger.info(
                    f"[{self.exp_id}] Episode {episode:4d} "
                    f"reward={episode_reward:7.1f}  "
                    f"ε={self.epsilon:.3f}  "
                    f"policy_v={current_version}  "
                    f"step={global_step}"
                )

                # report episode reward to N6
                try:
                    analytics_stub.ReportMetrics(
                        analytics_pb2.Metrics(
                            experiment_id = self.exp_id,
                            source_node   = "N2",
                            metric_name   = "episode_reward",
                            value         = float(episode_reward),
                            step          = global_step,
                        ),
                        timeout=2,
                    )
                except Exception:
                    pass   # analytics is non-critical

                # decay exploration rate
                self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
                state        = self.env.reset()
                episode_reward = 0.0
                episode       += 1
            else:
                state = next_state

            time.sleep(STEP_DELAY)

        logger.info(
            f"[{self.exp_id}] Rollout stopped "
            f"after {global_step} steps / {episode} episodes"
        )


# ════════════════════════════════════════════════════════════════════════════════
#  gRPC servicer
# ════════════════════════════════════════════════════════════════════════════════

class EnvironmentServicer(environment_pb2_grpc.EnvironmentServiceServicer):

    def __init__(self):
        self._runners: dict[str, RolloutRunner] = {}
        self._lock = threading.Lock()

    def Initialize(self, request, context):
        """Create a RolloutRunner (and the underlying environment) for this experiment."""
        print(f"[{request.experiment_id}] Initialize called with env='{request.env_name}'")
        exp_id = request.experiment_id
        with self._lock:
            if exp_id not in self._runners:
                self._runners[exp_id] = RolloutRunner(request)
                logger.info(f"[{exp_id}] Environment initialised")
            else:
                logger.info(f"[{exp_id}] Already initialised — skipping")
        return common_pb2.Status(ok=True, message="Initialised")

    def StartRollout(self, request, context):
        """Start (or resume) the background sampling loop for this experiment."""
        print(f"[{request.experiment_id}] StartRollout called")
        exp_id = request.experiment_id
        with self._lock:
            if exp_id not in self._runners:
                # allow StartRollout without a prior Initialize call
                self._runners[exp_id] = RolloutRunner(request)
            runner = self._runners[exp_id]
        runner.start()
        logger.info(f"[{exp_id}] Rollout started")
        return common_pb2.Status(ok=True, message="Rollout started")

    def StopRollout(self, request, context):
        print(f"[{request.experiment_id}] StopRollout called")
        """Signal the sampling loop to stop (non-blocking)."""
        exp_id = request.experiment_id
        with self._lock:
            runner = self._runners.get(exp_id)
        if runner:
            runner.stop()
            logger.info(f"[{exp_id}] Rollout stop signalled")
        return common_pb2.Status(ok=True, message="Rollout stop signalled")


# ── server entry point ─────────────────────────────────────────────────────────

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    environment_pb2_grpc.add_EnvironmentServiceServicer_to_server(
        EnvironmentServicer(), server
    )
    server.add_insecure_port(f"[::]:{PORT}")
    server.start()
    logger.info(f"N2 Environment Service started on port {PORT}")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
