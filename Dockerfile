# ------------------------------------------------------------------
# PaddleOCR API - Portable Docker image (works on Windows, macOS, Linux)
# ------------------------------------------------------------------
FROM python:3.11-slim

# Set non-interactive apt + locale to avoid warnings on some platforms
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# System libraries required by OpenCV + PaddleOCR + pyzbar
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libzbar0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY flask_api.py .

# Default port (can be overridden at run time)
ENV PORT=5100
EXPOSE 5100

# Run the Flask app
CMD ["python", "flask_api.py"]
