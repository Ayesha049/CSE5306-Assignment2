FROM python:3.10-slim

RUN pip install grpcio grpcio-tools mysql-connector-python

WORKDIR /app

COPY stubs ./stubs
COPY services/Analytics_Service_N6 ./Analytics_Service_N6

WORKDIR /app/Analytics_Service_N6

EXPOSE 50052

CMD ["python", "-u", "analytics_service.py"]