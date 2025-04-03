# Use the official Python image from the Docker Hub
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install git, build-essential, gfortran, and pkg-config
RUN apt-get update && apt-get install -y git build-essential gfortran pkg-config libopenblas-dev && rm -rf /var/lib/apt/lists/*
# Install LaTeX
RUN apt-get update && apt-get install -y --no-install-recommends texlive-latex-base texlive-fonts-recommended texlive-fonts-extra cm-super dvipng && rm -rf /var/lib/apt/lists/*
# Copy the requirements file into the container
COPY requirements.txt /app/requirements.txt
COPY requirements-dev.txt /app/requirements-dev.txt

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt  -r requirements-dev.txt
RUN pip install notebook jupyterlab

# Expose Jupyter's default port
EXPOSE 8888

# Copy the rest of the application code into the container
COPY . /app

# Define a volume
VOLUME [".:/app"]
