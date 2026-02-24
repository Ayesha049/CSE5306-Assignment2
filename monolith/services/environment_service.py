"""
N2: Environment Service (Monolithic)
───────────────────────────────────
• No gRPC calls
• Direct calls to:
    - buffer_service (N3)
    - policy_service (N5)
    - analytics_service (N6)
• Used internally by exp_service (N1)
"""

import json
import pickle
import numpy as np
import threading
import time
import logging

import common_pb2
import policy_pb2
import buffer_pb2
import analytics_pb2

# ── logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[N2-Monolith] %(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("N2-Monolith")

# ── rollout hyper-parameters ───────────────────────────────────────────────────
POLICY_REFRESH_STEPS = 50
EPSILON_START        = 1.0
EPSILON_END          = 0.05
EPSILON_DECAY        = 0.995
STEP_DELAY           = 0.005


# ════════════════════════════════════════════════════════════════════════════════
#  Built-in CartPole
# ════════════════════════════════════════════════════════════════════════════════

class CartPoleEnv:

    obs_dim   = 4
    n_actions = 2

    GRAVITY       = 9.8
    MASSCART      = 1.0
    MASSPOLE      = 0.1
    POLE_LEN      = 0.5
    FORCE_MAG     = 10.0
    TAU           = 0.02
    THETA_THRESH  = 12 * np.pi / 180
    X_THRESH      = 2.4

    def __init__(self, seed=42, max_steps=500):
        self.max_steps  = max_steps
        self.rng        = np.random.RandomState(seed)
        self.state      = None
        self._step_cnt  = 0
        self._total_mass    = self.MASSCART + self.MASSPOLE
        self._pole_mass_len = self.MASSPOLE * self.POLE_LEN

    def reset(self):
        self.state = self.rng.uniform(-0.05, 0.05, 4).astype(np.float32)
        self._step_cnt = 0
        return self.state.copy()

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state

        force = self.FORCE_MAG if action == 1 else -self.FORCE_MAG
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (force + self._pole_mass_len * theta_dot**2 * sintheta) / self._total_mass

        theta_acc = (
            self.GRAVITY * sintheta - costheta * temp
        ) / (self.POLE_LEN * (4/3 - self.MASSPOLE * costheta**2 / self._total_mass))

        x_acc = temp - self._pole_mass_len * theta_acc * costheta / self._total_mass

        x         += self.TAU * x_dot
        x_dot     += self.TAU * x_acc
        theta     += self.TAU * theta_dot
        theta_dot += self.TAU * theta_acc

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        self._step_cnt += 1

        done = bool(
            x < -self.X_THRESH or x > self.X_THRESH
            or theta < -self.THETA_THRESH or theta > self.THETA_THRESH
            or self._step_cnt >= self.max_steps
        )

        return self.state.copy(), 1.0, done


# ════════════════════════════════════════════════════════════════════════════════

def make_env(env_name, seed, max_steps):
    return CartPoleEnv(seed=seed, max_steps=max_steps or 500)


# ════════════════════════════════════════════════════════════════════════════════

def _select_action(state, weights, epsilon, rng, n_actions):
    if rng.random() < epsilon:
        return int(rng.randint(0, n_actions))

    try:
        h = np.maximum(0.0, state @ weights["W1"] + weights["b1"])
        q = h @ weights["W2"] + weights["b2"]
        return int(np.argmax(q))
    except Exception:
        return int(rng.randint(0, n_actions))


# ════════════════════════════════════════════════════════════════════════════════
#  Rollout Runner
# ════════════════════════════════════════════════════════════════════════════════

class RolloutRunner:

    def __init__(self, config, buffer_service, policy_service, analytics_service=None):
        self.config     = config
        self.exp_id     = config.experiment_id
        self.env        = make_env(config.env_name, config.seed, config.max_steps)

        self.buffer_service    = buffer_service
        self.policy_service    = policy_service
        self.analytics_service = analytics_service

        self.stop_event = threading.Event()
        self.thread     = None
        self.rng        = np.random.RandomState(config.seed)
        self.epsilon    = EPSILON_START

    def start(self):
        if self.thread and self.thread.is_alive():
            return

        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()

    def _run(self):

        policy_req = policy_pb2.PolicyRequest(experiment_id=self.exp_id)

        weights = None
        current_version = 0
        state = self.env.reset()
        episode_reward = 0.0
        episode = 0
        global_step = 0

        logger.info(f"[{self.exp_id}] Rollout started")

        while not self.stop_event.is_set():

            # ── fetch policy ───────────────────────────────────────────────
            if global_step % POLICY_REFRESH_STEPS == 0:
                try:
                    policy = self.policy_service.GetPolicy(policy_req)
                    weights = pickle.loads(policy.weights)
                    current_version = policy.version
                except Exception as e:
                    logger.warning(f"[{self.exp_id}] Policy fetch failed: {e}")

            # ── action selection ──────────────────────────────────────────
            if weights is not None:
                action = _select_action(
                    state, weights, self.epsilon, self.rng, self.env.n_actions
                )
            else:
                action = int(self.rng.randint(0, self.env.n_actions))

            # ── step env ──────────────────────────────────────────────────
            next_state, reward, done = self.env.step(action)
            episode_reward += reward

            # ── push to buffer ────────────────────────────────────────────
            try:
                transition = buffer_pb2.Transition(
                    state=json.dumps(state.tolist()),
                    action=str(int(action)),
                    reward=float(reward),
                    next_state=json.dumps(next_state.tolist()),
                    done=bool(done),
                )

                self.buffer_service.PushTransition(
                    buffer_pb2.PushRequest(transition=transition)
                )
            except Exception as e:
                logger.warning(f"[{self.exp_id}] Buffer push failed: {e}")

            global_step += 1

            # ── episode end ───────────────────────────────────────────────
            if done:

                if self.analytics_service:
                    try:
                        self.analytics_service.ReportMetrics(
                            analytics_pb2.Metrics(
                                experiment_id=self.exp_id,
                                source_node="N2",
                                metric_name="episode_reward",
                                value=float(episode_reward),
                                step=global_step,
                            )
                        )
                    except Exception:
                        pass

                self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

                state = self.env.reset()
                episode_reward = 0.0
                episode += 1
            else:
                state = next_state

            time.sleep(STEP_DELAY)


# ════════════════════════════════════════════════════════════════════════════════
#  Environment Service (Monolithic)
# ════════════════════════════════════════════════════════════════════════════════

class EnvironmentServicer:

    def __init__(self, buffer_service, policy_service, analytics_service=None):
        self._runners = {}
        self._lock = threading.Lock()

        self.buffer_service = buffer_service
        self.policy_service = policy_service
        self.analytics_service = analytics_service

    def Initialize(self, request):
        exp_id = request.experiment_id

        with self._lock:
            if exp_id not in self._runners:
                self._runners[exp_id] = RolloutRunner(
                    request,
                    buffer_service=self.buffer_service,
                    policy_service=self.policy_service,
                    analytics_service=self.analytics_service,
                )

        return common_pb2.Status(ok=True, message="Initialized")

    def StartRollout(self, request):
        exp_id = request.experiment_id

        with self._lock:
            if exp_id not in self._runners:
                self._runners[exp_id] = RolloutRunner(
                    request,
                    buffer_service=self.buffer_service,
                    policy_service=self.policy_service,
                    analytics_service=self.analytics_service,
                )

            runner = self._runners[exp_id]

        runner.start()
        return common_pb2.Status(ok=True, message="Rollout started")

    def StopRollout(self, request):
        exp_id = request.experiment_id

        with self._lock:
            runner = self._runners.get(exp_id)

        if runner:
            runner.stop()

        return common_pb2.Status(ok=True, message="Rollout stopped")