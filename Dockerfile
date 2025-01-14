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

# Clone the Taipy repository and set up the application
RUN git clone https://github.com/dovineowuor/Taipy.git /app

# Install pip dependencies
RUN pip install --upgrade pip

# Copy requirements.txt for dependency installation
COPY requirements.txt /app/requirements.txt

# Download and install dependencies with retries
RUN mkdir -p ./packages && \
    pip download nvidia_cusparse_cu12==12.3.1.170 -d ./packages || \
    (echo "Retrying download..." && sleep 5 && \
    pip download nvidia_cusparse_cu12==12.3.1.170 -d ./packages) && \
    pip install --no-cache-dir -r /app/requirements.txt --find-links=./packages

# Expose the port that the application will run on
EXPOSE 5000

# Copy the rest of your application
COPY . /app

# Command to run the application
CMD ["taipy", "run", "main.py", "--use-reloader", "--host", "0.0.0.0", "--port", "5000"]
# "taipy", "run", "main.py", "--use-reloader", "--host", "0.0.0.0", "--port", "5000"
# taipy run main.py --use-reloader --host 0.0.0.0 --port 5000