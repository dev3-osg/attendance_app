# One Step Greener – Face recognition attendance (Hugging Face Spaces)
# Spaces expect the app to listen on port 7860.

FROM python:3.10-slim

# OpenCV and other libs need these
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

COPY . .

# Hugging Face Spaces use port 7860
ENV PORT=7860
EXPOSE 7860

# Single worker (ML models in memory); multiple threads for concurrent requests
CMD gunicorn --bind 0.0.0.0:7860 --workers 1 --threads 4 --timeout 120 app:app
