Policy

docker run -d --name policy --network rl-net -v ./stubs:/app/stubs -v ./services/Policy_Service_N5:/app/Policy_Service_N5 -p 50056:50056 policy-image

Analytics

docker run -d --name analytics --network rl-net -v ./stubs:/app/stubs -v ./services/Analytics_Service_N6:/app/Analytics_Service_N6 -p 50052:50052 analytics-image

Buffer

docker run -d --name buffer --network rl-net -v ./stubs:/app/stubs -v ./services/Buffer_Service_N3:/app/Buffer_Service_N3 -p 50051:50051 buffer-image

Environment

docker run -d --name environment --network rl-net -v ./stubs:/app/stubs -v ./services/Environment_Service_N2:/app/Environment_Service_N2 -p 50050:50050 environment-image

Experiment

docker run -d --name experiment --network rl-net -v ./stubs:/app/stubs -v ./services/Experiment_Service_N1:/app/Experiment_Service_N1 -p 50053:50053 experiment-image

