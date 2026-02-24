import grpc
import time

import sys
sys.path.append("monolith/stubs")

import experiment_pb2
import experiment_pb2_grpc
import common_pb2


channel = grpc.insecure_channel("localhost:50063")
stub = experiment_pb2_grpc.ExperimentServiceStub(channel)

config = common_pb2.ExperimentConfig(
    experiment_id="test_exp",
    env_name="CartPole",
    algorithm="DQN",
    seed=42,
    max_steps=200
)

print("Starting experiment...")
response = stub.StartExperiment(config)
print(response)

time.sleep(5)

print("Querying status...")
status = stub.GetExperimentStatus(config)
print(status)

print("Stopping experiment...")
stop = stub.StopExperiment(config)
print(stop)