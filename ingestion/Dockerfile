FROM python:3.9-slim

WORKDIR /app

# System dependencies (optional)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source code
COPY . .

# Expose the FastAPI port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "ingestion_main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
