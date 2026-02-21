FROM python:3.10-slim

RUN pip install grpcio grpcio-tools

WORKDIR /app

# Copy proto stubs from root context
COPY stubs ./stubs

# Copy buffer service code
COPY services/Buffer_Service_N3 ./Buffer_Service_N3

WORKDIR /app/Buffer_Service_N3

EXPOSE 50051

CMD ["python", "-u", "buffer_service.py"]