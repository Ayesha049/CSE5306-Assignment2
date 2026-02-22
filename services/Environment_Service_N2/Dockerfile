FROM python:3.10-slim

RUN pip install grpcio grpcio-tools numpy

WORKDIR /app

COPY stubs ./stubs
COPY services/Environment_Service_N2 ./Environment_Service_N2

WORKDIR /app/Environment_Service_N2

EXPOSE 50050

CMD ["python", "-u", "environment_service.py"]