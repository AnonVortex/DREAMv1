FROM python:3.9-slim

WORKDIR /app

# Install system dependencies if needed
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all module files
COPY . .

# Expose the service port (default: 8900)
EXPOSE 8900

# Run the FastAPI application
CMD ["uvicorn", "comm_optimization:app", "--host", "0.0.0.0", "--port", "8900", "--reload"]
