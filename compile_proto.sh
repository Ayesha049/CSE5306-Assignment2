#!/bin/bash
# Compile all proto files to stubs/ directory.
# Run from the project root: bash compile_proto.sh

set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
STUBS="$ROOT/stubs"

mkdir -p "$STUBS"

echo "=== Compiling proto files ==="
echo "Root: $ROOT"
echo "Output: $STUBS"
echo ""

# ── 1. common.proto ───────────────────────────────────────────────────────────
# Shared messages: ExperimentConfig, Policy, Status, Transition, Batch, Metrics
echo "[1/4] common.proto → common_pb2.py"
python3 -m grpc_tools.protoc \
  -I"$ROOT" \
  --python_out="$STUBS" \
  --grpc_python_out="$STUBS" \
  "$ROOT/common.proto"

# ── 2. policy.proto (N5) ──────────────────────────────────────────────────────
# PolicyService: GetPolicy, UpdatePolicy, GetPolicyVersion
echo "[2/4] policy.proto → policy_pb2.py / policy_pb2_grpc.py"
(cd "$ROOT/services/Policy_Service(N5)" && \
  python3 -m grpc_tools.protoc \
    -I. \
    -I"$ROOT" \
    --python_out="$STUBS" \
    --grpc_python_out="$STUBS" \
    policy.proto)

# ── 3. environment.proto (N2) ─────────────────────────────────────────────────
# EnvironmentService: Initialize, StartRollout, StopRollout
echo "[3/4] environment.proto → environment_pb2.py / environment_pb2_grpc.py"
(cd "$ROOT/services/Environment_Service(N2)" && \
  python3 -m grpc_tools.protoc \
    -I. \
    -I"$ROOT" \
    --python_out="$STUBS" \
    --grpc_python_out="$STUBS" \
    environment.proto)

# ── 4. experiment.proto (N1) ──────────────────────────────────────────────────
# ExperimentService: StartExperiment, StopExperiment, GetExperimentStatus
echo "[4/4] experiment.proto → experiment_pb2.py / experiment_pb2_grpc.py"
(cd "$ROOT/services/Experiment_Service(N1)" && \
  python3 -m grpc_tools.protoc \
    -I. \
    -I"$ROOT" \
    --python_out="$STUBS" \
    --grpc_python_out="$STUBS" \
    experiment.proto)

echo ""
echo "=== Done. Generated stubs: ==="
ls "$STUBS"/*.py 2>/dev/null | xargs -I{} basename {}
