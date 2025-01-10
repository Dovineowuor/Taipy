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

# Clone the Taipy repository and install dependencies
COPY Taipy /app

# Install pip dependencies
RUN pip install --upgrade pip
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt -i https://pypi.org/simple

# Expose the port that the Flask app will run on
EXPOSE 60675

# Copy the rest of your application
COPY . /app

# Command to run the application
CMD ["python", "main.py"]
