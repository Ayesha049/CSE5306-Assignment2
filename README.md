# CSE5306-Assignment2 （DS for RL training）
(for compiling, running, and testing your system)
A microservice-based distributed reinforcement learning (RL) system where six independent services communicate over **gRPC** to collectively run RL experiments. The system supports training agents on CartPole (built-in) and any discrete-action Gymnasium environment.

## Architecture Overview

```
User / Client
     │
     ▼
┌─────────────┐        ┌──────────────────┐
│  N1         │──────▶│  N2              │
│  Experiment │        │  Environment     │
│  Service    │        │  Service         │
│  :50053     │        │  :50050          │
└─────────────┘        └────────┬─────────┘
                                │  push transitions      fetch policy
                     ┌──────────▼────────┐    ┌─────────────────────┐
                     │  N3               │    │  N5                 │
                     │  Buffer Service   │    │  Policy Service     │
                     │  :50051           │    │  :50056             │
                     └──────────┬────────┘    └──────────▲──────────┘
                                │ sample batches          │ push weights
                     ┌──────────▼────────────────────────┴──────────┐
                     │  N4  Learner Service                          │
                     └──────────┬────────────────────────────────────┘
                                │ report metrics
                     ┌──────────▼────────┐
                     │  N6               │
                     │  Analytics Svc    │◀── N2, N3, N4 also report
                     │  :50052 (MySQL)   │
                     └───────────────────┘
```

## Services

| # | Service | Port | Description |
|---|---------|------|-------------|
| N1 | Experiment Service | 50053 | User-facing entry point; manages experiment lifecycle |
| N2 | Environment Service | 50050 | Runs RL rollouts and generates experience transitions |
| N3 | Buffer Service | 50051 | Replay buffer — stores and samples experience transitions |
| N4 | Learner Service | — | Trains the policy on sampled batches |
| N5 | Policy Service | 50056 | Stores and serves versioned MLP policy weights |
| N6 | Analytics Service | 50052 | Centralised metrics collection backed by MySQL |

### N1 — Experiment Service
- Provides the API for starting, stopping, and monitoring experiments
- Accepts experiment configurations (or uses defaults)
- Delegates environment initialisation to N2 and fetches results from N6
- Uses `threading.Lock` to run multiple experiments concurrently without interference
- `auto_start_experiment` can launch a default-config experiment to smoke-test the whole system

### N2 — Environment Service
- Central sampling node; runs background rollout loops
- Supports CartPole (built-in, zero dependencies) and any discrete-action Gymnasium environment via `GymnasiumWrapper`; falls back to CartPole if Gymnasium is unavailable
- Action selection: epsilon-greedy with a 2-layer MLP
- Fetches fresh policy weights from N5 every `POLICY_REFRESH_STEPS` steps
- Pushes `(s, a, r, s', done)` transitions to N3 and per-episode rewards to N6
- Thread-isolated per experiment

**Key defaults**

| Parameter | Value |
|-----------|-------|
| Policy refresh steps | 50 |
| Epsilon start / end / decay | 1.0 / 0.05 / 0.995 |
| Step delay | 0.005 s |
| Max steps per episode | 500 |

### N3 — Buffer Service
- Fixed-capacity FIFO replay buffer (10 000 transitions)
- Thread-safe concurrent access via locks
- Persists buffer to `replay_buffer.pkl` every 10 pushes; auto-restores on startup
- Reports buffer-size metrics to N6
- Serves random-without-replacement batches to N4

### N4 — Learner Service
- Requests batches (size 32) from N3, runs a training step, and pushes updated weights to N5
- Reports loss metrics to N6
- Supports simple load balancing across multiple buffer nodes
- Training currently uses a simulated loss computation (random values)

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

```bash
docker compose up --build
```

Services start in dependency order:

1. **MySQL** (N6 dependency)
2. **N6** Analytics Service
3. **N3** Buffer Service + **N5** Policy Service (in parallel)
4. **N2** Environment Service
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

| Caller | Callee | Call |
|--------|--------|------|
| User | N1 | Start / Stop / Status experiment |
| N1 | N2 | Initialize, StartRollout, StopRollout |
| N1 | N6 | QueryMetrics (fetch results) |
| N2 | N3 | PushTransition |
| N2 | N5 | GetPolicy |
| N2 | N6 | ReportMetrics (episode reward) |
| N3 | N6 | ReportMetrics (buffer size) |
| N4 | N3 | SampleBatch |
| N4 | N5 | UpdatePolicy |
| N4 | N6 | ReportMetrics (loss) |