# filepath: /Users/dove/Work/taipy/Dockerfile
# Use the official Python 3.10 slim image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install necessary system dependencies
RUN apt-get update && \
    apt-get install -y \
    git \
    libmagic1 \
    file \
    && rm -rf /var/lib/apt/lists/*

# Copy the Taipy repository and install dependencies
COPY Taipy /app

# Install pip dependencies
RUN pip install --upgrade pip
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt && \
    rm -rf /root/.cache/pip

# Expose the port that the Flask app will run on
EXPOSE 60675

# Copy the rest of your application
COPY . /app

# Command to run the application
CMD ["taipy", "run", "main.py", "--use-reloader"]
