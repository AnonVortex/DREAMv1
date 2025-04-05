FROM python:3.9-slim

WORKDIR /app

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all module files
COPY . .

# Expose the Memory module port (example: 8401)
EXPOSE 8401

# Run the application
CMD ["uvicorn", "memory_main:app", "--host", "0.0.0.0", "--port", "8401", "--reload"]
