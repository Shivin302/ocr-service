FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ocr.py .

# Expose the port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "ocr:app", "--host", "0.0.0.0", "--port", "8000"] 