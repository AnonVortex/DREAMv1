FROM python:3.9-slim

WORKDIR /app

# Install any system dependencies if needed (for example, for image processing libraries)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all module files
COPY . .

# Expose the service port (default 8300)
EXPOSE 8300

# Run the FastAPI app with auto-reload (for development)
CMD ["uvicorn", "routing_main:app", "--host", "0.0.0.0", "--port", "8300", "--reload"]
