FROM python:3.9-slim

WORKDIR /app

# Install system dependencies if needed
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire aggregator code
COPY . .

# Expose the aggregator port (default: 9000)
EXPOSE 9000

# Run the aggregator application
CMD ["uvicorn", "main_pipeline:app", "--host", "0.0.0.0", "--port", "9000", "--reload"]
