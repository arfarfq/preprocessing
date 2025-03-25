# Use Python official image from DockerHub
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the Python script and requirements.txt into the container
COPY processing_lcs.py ./
COPY requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run the Python script when the container starts
CMD ["python", "processing_lcs.py"]
