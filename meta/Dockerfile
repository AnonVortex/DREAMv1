FROM python:3.9-slim

WORKDIR /app

# Install system dependencies if needed
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire module
COPY . .

# Expose the service port (e.g., 8300)
EXPOSE 8300

# Run the FastAPI application
CMD ["uvicorn", "meta_main:app", "--host", "0.0.0.0", "--port", "8300", "--reload"]
