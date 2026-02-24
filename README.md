# CSE5306-Assignment2 （DS for RL training）
(for compiling, running, and testing your system)
A microservice-based distributed reinforcement learning (RL) system where six independent services communicate over **gRPC** to collectively run RL experiments. The system supports training agents on CartPole (built-in) and any discrete-action Gymnasium environment.

## Architecture Overview

```
User / Client
     │
     ▼
┌─────────────┐   round-robin   ┌──────────────────┐
│  N1         │────────────────▶│  N2  ×M replicas │
│  Experiment │  (per-exp route)│  Environment Svc │
│  Service    │                 │  :50050          │
│  :50053     │                 └────────┬─────────┘
└─────────────┘                          │ push transitions (round-robin)
                                         │              fetch policy
                     ┌───────────────────▼───┐  ┌─────────────────────┐
                     │  N3  ×N replicas      │  │  N5                 │
                     │  Buffer Service       │  │  Policy Service     │
                     │  :50051              │  │  :50056             │
                     └───────────┬───────────┘  └──────────▲──────────┘
                                 │ sample from all shards   │ push weights
                     ┌───────────▼───────────────────────────┴──────────┐
                     │  N4  Learner Service : 50054                      │
                     └───────────┬───────────────────────────────────────┘
                                 │ report metrics
                     ┌───────────▼────────┐
                     │  N6                │
                     │  Analytics Svc     │◀── N2, N3, N4 also report
                     │  :50052 (MySQL)    │
                     └────────────────────┘
```

## Services

| # | Service | Port | Scalable | Description |
|---|---------|------|----------|-------------|
| N1 | Experiment Service | 50053 | — | User-facing entry point; manages experiment lifecycle and load-balances across N2 replicas |
| N2 | Environment Service | 50050 | **Yes** | Runs RL rollouts and distributes transitions across N3 replicas via round-robin |
| N3 | Buffer Service | 50051 | **Yes** | Shared replay buffer — stores transitions and serves sampled batches to N4 |
| N4 | Learner Service | 50054 | — | Trains the policy by sampling from all N3 shards and pushing weights to N5 |
| N5 | Policy Service | 50056 | — | Stores and serves versioned MLP policy weights |
| N6 | Analytics Service | 50052 | — | Centralised metrics collection backed by MySQL |

### N1 — Experiment Service
- Provides the API for starting, stopping, and monitoring experiments
- Accepts experiment configurations (or uses defaults)
- Delegates environment initialisation to N2 and fetches results from N6
- Uses `threading.Lock` to run multiple experiments concurrently without interference
- `auto_start_experiment` can launch a default-config experiment to smoke-test the whole system
- **Load balancing**: discovers all running N2 replicas via Docker DNS at each `StartExperiment` call and distributes experiments using round-robin; tracks the assigned N2 per experiment so `StopExperiment` routes to the correct instance

### N2 — Environment Service
- Central sampling node; runs background rollout loops
- Supports CartPole (built-in, zero dependencies) and any discrete-action Gymnasium environment via `GymnasiumWrapper`; falls back to CartPole if Gymnasium is unavailable
- Action selection: epsilon-greedy with a 2-layer MLP
- Fetches fresh policy weights from N5 every `POLICY_REFRESH_STEPS` steps
- Pushes `(s, a, r, s', done)` transitions to N3 and per-episode rewards to N6
- Thread-isolated per experiment
- **Load balancing**: at rollout start, discovers all running N3 replicas via Docker DNS and pushes transitions using round-robin, distributing write load evenly across all buffer shards

**Key defaults**

| Parameter | Value |
|-----------|-------|
| Policy refresh steps | 50 |
| Epsilon start / end / decay | 1.0 / 0.05 / 0.995 |
| Step delay | 0.005 s |
| Max steps per episode | 500 |

### N3 — Buffer Service
- Fixed-capacity FIFO replay buffer (10 000 transitions per replica)
- Thread-safe concurrent access via locks
- Persists buffer to `replay_buffer.pkl` every 10 pushes; auto-restores on startup
- Reports buffer-size metrics to N6
- Serves random-without-replacement batches to N4
- **Scalable**: multiple replicas act as independent shards of the global buffer; scale with `--scale buffer=N`

### N4 — Learner Service
- Requests batches (size 32) from N3, runs a training step, and pushes updated weights to N5
- Reports loss metrics to N6
- Training currently uses a simulated loss computation (random values)
- **Load balancing**: discovers all running N3 replicas via Docker DNS and splits each sample request evenly across all shards, then merges results — ensures every shard contributes to each training step, preventing biased sampling when N2 distributes transitions via round-robin

### N5 — Policy Management Service
- Central policy registry; stores versioned MLP weights
- Auto-seeds a random v0 policy (CartPole dimensions) when no weights exist yet
- N2 calls `GetPolicy` at rollout start and every `POLICY_REFRESH_STEPS` steps
- N4 calls `UpdatePolicy` after each training iteration with incremented version number

### N6 — Analytics Service
- Receives metrics via gRPC `ReportMetrics` from N2, N3, N4
- Stores data in a normalised MySQL schema (`rl_analytics` database)
- Supports `QueryMetrics` for experiment monitoring
- Tables are auto-created on first run

**Database schema**

```
nodes            metric_types       metrics
─────────        ────────────       ───────
node_id (PK)     metric_type_id     metric_id (PK)
node_name        name (unique)      node_id (FK)
node_type        description        metric_type_id (FK)
created_at                          value (FLOAT)
                                    experiment_id
                                    step (BIGINT)
                                    timestamp
```

## Repository Layout

```
.
├── docker-compose.yml          # One-command deployment
├── compile_proto.sh            # Regenerate gRPC stubs from protos/
├── protos/                     # Protocol Buffer definitions
│   ├── common.proto            # Shared messages (ExperimentConfig, Policy, …)
│   ├── analytics.proto
│   ├── buffer.proto
│   ├── environment.proto
│   ├── experiment.proto
│   ├── learner.proto
│   └── policy.proto
├── stubs/                      # Generated *_pb2.py / *_pb2_grpc.py files
└── services/
    ├── Experiment_Service_N1/
    ├── Environment_Service_N2/
    ├── Buffer_Service_N3/
    ├── Learner_Service_N4/
    ├── Policy_Service_N5/
    └── Analytics_Service_N6/
```

## Getting Started

### Prerequisites

- Docker and Docker Compose

### Run with Docker Compose (recommended)

**Single instance (default)**

```bash
docker compose up --build
```

**Scale N2 and N3 for load balancing**

```bash
# Example: 2 Environment workers + 3 Buffer shards
docker compose up --build --scale environment=2 --scale buffer=3

# Any replica count is supported — no code or config changes needed
docker compose up --build --scale environment=M --scale buffer=N
```

> N1 and N4 automatically discover all running N2 / N3 replicas via Docker's embedded DNS at runtime and adapt their routing accordingly.

Services start in dependency order:

1. **MySQL** (N6 dependency)
2. **N6** Analytics Service
3. **N3** Buffer Service × N + **N5** Policy Service (in parallel)
4. **N2** Environment Service × M
5. **N4** Learner Service
6. **N1** Experiment Service

### Run Individual Containers

First create a shared network:

```bash
docker network create rl-net
```

Then start each service:

```bash
# N5 – Policy Service
docker run -d --name policy --network rl-net \
  -v ./stubs:/app/stubs \
  -v ./services/Policy_Service_N5:/app/Policy_Service_N5 \
  -p 50056:50056 policy-image

# N6 – Analytics Service
docker run -d --name analytics --network rl-net \
  -v ./stubs:/app/stubs \
  -v ./services/Analytics_Service_N6:/app/Analytics_Service_N6 \
  -p 50052:50052 analytics-image

# N3 – Buffer Service
docker run -d --name buffer --network rl-net \
  -v ./stubs:/app/stubs \
  -v ./services/Buffer_Service_N3:/app/Buffer_Service_N3 \
  -p 50051:50051 buffer-image

# N2 – Environment Service
docker run -d --name environment --network rl-net \
  -v ./stubs:/app/stubs \
  -v ./services/Environment_Service_N2:/app/Environment_Service_N2 \
  -p 50050:50050 environment-image

# N1 – Experiment Service
docker run -d --name experiment --network rl-net \
  -v ./stubs:/app/stubs \
  -v ./services/Experiment_Service_N1:/app/Experiment_Service_N1 \
  -p 50053:50053 experiment-image
```

### Regenerate gRPC Stubs

If you modify any `.proto` file, regenerate the Python stubs:

```bash
bash compile_proto.sh
```

Requires `grpcio-tools`:

```bash
pip install grpcio-tools
```

## Service Communication Summary

| Caller | Callee | Call | Load Balancing |
|--------|--------|------|----------------|
| User | N1 | Start / Stop / Status experiment | — |
| N1 | N2 ×M | Initialize, StartRollout, StopRollout | Round-robin per experiment; StopRollout routed to owning replica |
| N1 | N6 | QueryMetrics (fetch results) | — |
| N2 | N3 ×N | PushTransition | Round-robin per step across all N3 shards |
| N2 | N5 | GetPolicy | — |
| N2 | N6 | ReportMetrics (episode reward) | — |
| N3 | N4 | TrainStep (trigger) | — |
| N3 | N6 | ReportMetrics (buffer size) | — |
| N4 | N3 ×N | SampleBatch | Split batch evenly across all N3 shards, then merge |
| N4 | N5 | UpdatePolicy | — |
| N4 | N6 | ReportMetrics (loss) | — |

## Scalability Design

### Dynamic Service Discovery

N2 and N3 replicas are discovered at runtime using **Docker's embedded DNS**: when a service is scaled to N containers, all N container IPs are returned by a single `socket.getaddrinfo(service_name, port)` call. No configuration changes are needed to add or remove replicas.

```
socket.getaddrinfo("buffer", 50051)
  → [172.18.0.3, 172.18.0.4, 172.18.0.5]   # 3 N3 replicas discovered automatically
```

### Load Balancing Strategies

| Link | Strategy | Rationale |
|------|-----------|-----------|
| N1 → N2 | Round-robin | Distributes experiment startup load; `experiment_id → N2 IP` mapping ensures `StopRollout` reaches the correct instance |
| N2 → N3 | Round-robin (per step) | Evenly spreads write load across all buffer shards so no single N3 becomes a bottleneck |
| N4 → N3 | Split-merge | Requests a proportional sub-batch from **each** N3 shard and merges results; necessary because N2's round-robin means each shard holds only 1/N of total experience — sampling from one shard alone would produce a biased training batch |

### Backward Compatibility

All discovery functions fall back to the original single-instance address if DNS resolution fails (e.g. when running outside Docker), so the system works identically with or without `--scale`.