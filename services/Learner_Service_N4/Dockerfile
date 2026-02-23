FROM python:3.10-slim

# Install system dependencies for MPI
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenmpi-dev \
    openmpi-bin \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install grpcio grpcio-tools mpi4py

# Set working directory
WORKDIR /app

# Copy stubs and service code
COPY stubs ./stubs
COPY services/Learner_Service_N4 ./Learner

WORKDIR /app/Learner

EXPOSE 50054

# Set default command
CMD ["python", "-u", "learner_service.py"]