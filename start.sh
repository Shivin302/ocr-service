#!/bin/bash

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Install requirements if needed
if [ "$1" == "--install" ]; then
    echo "Installing requirements..."
    pip install -r requirements.txt
fi

# Start the OCR API service
echo "Starting OCR API service..."
uvicorn ocr:app --host 0.0.0.0 --port 8000 --reload

# To run with installation: ./start.sh --install 