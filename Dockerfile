# Use the official Python 3.10 slim image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install necessary system dependencies
RUN apt-get update && \
    apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone the Taipy repository and install dependencies
RUN git clone https://github.com/Dovineowuor/Taipy

# Install pip dependencies
RUN pip install --upgrade pip
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt -i https://pypi.org/simple

# Set environment variables for the ngrok token (you can update this later in your script)
# Copy the .env file to the working directory
COPY .env /app/.env

# Use the contents of the .env file to set environment variables
RUN export $(cat /app/.env | xargs)

# Expose the port that the Flask app will run on
EXPOSE 60675

# Copy the rest of your application
COPY . /app

# Run the app (you should have a main.py script in the current directory)
CMD ["python", "main.py"]
