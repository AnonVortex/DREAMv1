# Use Python 3.11 slim as base image
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install build dependencies and Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get remove -y gcc python3-dev && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Final stage
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/

# Copy application code
COPY . .

# Create logs directory
RUN mkdir -p logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MODULE_NAME=pipeline_service
ENV PORT=8500

# Expose port
EXPOSE 8500

# Run the application
CMD ["uvicorn", "pipeline_service:app", "--host", "0.0.0.0", "--port", "8500"]
