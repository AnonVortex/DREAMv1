FROM python:3.9-slim

WORKDIR /app

# Install any system dependencies (if needed)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all module files
COPY . .

# Expose the Aggregation module port (default: 8500)
EXPOSE 8500

# Run the FastAPI application
CMD ["uvicorn", "aggregation_main:app", "--host", "0.0.0.0", "--port", "8500", "--reload"]
