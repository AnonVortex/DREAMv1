FROM python:3.9-slim

WORKDIR /app

# Install any system dependencies if needed (e.g., for GPU drivers or image processing libraries)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all module files
COPY . .

# Expose the service port (e.g., 8400)
EXPOSE 8400

# Run the application
CMD ["uvicorn", "specialized_main:app", "--host", "0.0.0.0", "--port", "8400", "--reload"]
