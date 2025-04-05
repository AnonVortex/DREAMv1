FROM python:3.9-slim

WORKDIR /app

# (Optional) For installing system libs needed for image/audio processing
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8100

CMD ["uvicorn", "perception_main:app", "--host", "0.0.0.0", "--port", "8100", "--reload"]
