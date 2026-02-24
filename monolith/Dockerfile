FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir \
    grpcio \
    grpcio-tools \
    mysql-connector-python \
    numpy

EXPOSE 50053

CMD ["python", "-u", "server.py"]