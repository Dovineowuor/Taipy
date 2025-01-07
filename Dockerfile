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
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Set environment variables for the ngrok token (you can update this later in your script)
ENV NGROK_AUTH_TOKEN='your_ngrok_token_here'
ENV HF_TOKEN='your_hf_token_here'
ENV GOOGLE_AI_API_KEY='your_google_ai_api_key_here'

# Expose the port that the Flask app will run on
EXPOSE 60675

# Copy the rest of your application
COPY . /app

# Run the app (you should have a main.py script in the current directory)
CMD ["python", "main.py"]
