FROM python:3.10-slim

RUN pip install grpcio grpcio-tools numpy

WORKDIR /app

COPY stubs ./stubs
COPY services/Policy_Service_N5 ./Policy_Service_N5

WORKDIR /app/Policy_Service_N5

EXPOSE 50056

CMD ["python", "-u", "policy_service.py"]