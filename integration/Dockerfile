FROM python:3.9-slim

WORKDIR /app

# Install system dependencies if needed (e.g., for advanced numeric processing)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire module code
COPY . .

# Expose the FastAPI port (example: 8200)
EXPOSE 8200

# Default command (with auto-reload for dev)
CMD ["uvicorn", "integration_main:app", "--host", "0.0.0.0", "--port", "8200", "--reload"]
