FROM python:3.9-slim

WORKDIR /app

# Install system dependencies if needed
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all module files
COPY . .

# Expose the service port (example: 8700)
EXPOSE 8700

# Run the FastAPI application
CMD ["uvicorn", "monitoring_main:app", "--host", "0.0.0.0", "--port", "8700", "--reload"]
