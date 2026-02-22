FROM python:3.10-slim

RUN pip install grpcio grpcio-tools

WORKDIR /app

# Copy proto stubs
COPY stubs ./stubs

# Copy experiment service code
COPY services/Experiment_Service_N1 ./Experiment_Service_N1

WORKDIR /app/Experiment_Service_N1

EXPOSE 50053

CMD ["python", "-u", "experiment_service.py"]