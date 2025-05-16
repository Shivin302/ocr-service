import io
from typing import List
import logging
import sys
import time
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image, UnidentifiedImageError
import doctr
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from collections import deque
import threading
import asyncio

ocr_models = []
if torch.cuda.is_available():
    num_devices = torch.cuda.device_count()
    for i in range(num_devices):
        device_i = torch.device(f"cuda:{i}")
        predictor = ocr_predictor('db_resnet50', 'crnn_vgg16_bn', pretrained=True, det_bs=20, reco_bs=1024).to(device_i)
        ocr_models.append(predictor)
else:
    raise RuntimeError("No GPU available")

app = FastAPI(title="DocTR OCR API", description="API for OCR using docTR")


class APIMetrics:
    def __init__(self):
        self.total_requests = 0
        self.active_requests = 0
        self.max_concurrent_requests = 0
        self.request_times = deque(maxlen=1000)  # Store last 1000 request timestamps
        self.lock = threading.Lock()
    
    def start_request(self):
        with self.lock:
            self.total_requests += 1
            self.active_requests += 1
            self.max_concurrent_requests = max(self.max_concurrent_requests, self.active_requests)
            self.request_times.append(time.time())
            return self.active_requests  # Return current count for logging
    
    def end_request(self):
        with self.lock:
            self.active_requests -= 1
            return self.active_requests  # Return current count for logging
    
    def get_request_interval(self):
        """Calculate average interval between requests in seconds"""
        if len(self.request_times) < 2:
            return 0
        intervals = [self.request_times[i] - self.request_times[i-1] 
                     for i in range(1, len(self.request_times))]
        return sum(intervals) / len(intervals) if intervals else 0
    
    def get_metrics(self):
        with self.lock:
            metrics = {
                "total_requests": self.total_requests,
                "active_requests": self.active_requests,
                "max_concurrent_requests": self.max_concurrent_requests,
                "avg_request_interval": self.get_request_interval()
            }
            return metrics
        
    def reset_metrics(self):
        with self.lock:
            self.total_requests = 0
            self.active_requests = 0
            self.max_concurrent_requests = 0
            self.request_times.clear()

metrics = APIMetrics()

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    # Skip metrics tracking for metrics endpoint itself
    if request.url.path == "/metrics":
        return await call_next(request)
    
    # Start timing and tracking
    active = metrics.start_request()
    print(f"Request started - Active: {active} - Path: {request.url.path}")
    
    try:
        # Process the request
        response = await call_next(request)
        return response
    finally:
        # Always end request tracking even if there was an exception
        active = metrics.end_request()
        print(f"Request completed - Active: {active} - Path: {request.url.path}")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {
        "message": "DocTR OCR API is running",
        "endpoints": [
            {"path": "/", "method": "GET", "description": "This documentation"},
            {"path": "/ocr", "method": "POST", "description": "Perform OCR on an uploaded image"},
            {"path": "/metrics", "method": "GET", "description": "Get API usage metrics"}
        ]
    }

@app.get("/metrics")
def get_metrics(reset: bool = False):
    result = metrics.get_metrics()
    if reset:
        metrics.reset_metrics()
    return result

# @app.post("/ocr", response_model=OCRResponse)
@app.post("/ocr")
async def perform_ocr(file: UploadFile = File(...)):
    try:
        # Start timing the processing
        # Note: The metrics.start_request() is now handled by the middleware
        start_time = time.time()

        contents = await file.read()

        # Create BytesIO object
        image_bytes = io.BytesIO(contents)
        
        doc = DocumentFile.from_images(contents)

        random_gpu_id = np.random.randint(0, num_devices)
        predictor = ocr_models[random_gpu_id]

        result = predictor(doc)

        end_time = time.time()
        request_time = round(end_time - start_time, 3)
        
        return {"pred": result.export(), "request_time": request_time}

    except Exception as img_error:
        raise HTTPException(status_code=422, detail=f"Invalid image format: {str(img_error)}")
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"OCR processing error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
