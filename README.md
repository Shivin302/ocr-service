# OCR API with docTR

A simple OCR (Optical Character Recognition) service built with [docTR](https://github.com/mindee/doctr) and FastAPI.

## Features

- OCR processing of uploaded images
- Returns both full text and word-level information with confidence scores
- Simple REST API with JSON responses
- CORS-enabled for cross-origin requests

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Start the server:

```bash
python ocr.py
```

or

```bash
uvicorn ocr:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at http://localhost:8000.

## API Endpoints

### GET /

Returns basic information about the API.

### POST /ocr

Processes an image and returns OCR results.

**Request:**
- Content-Type: multipart/form-data
- Body: file (image file)

**Response:**
```json
{
  "text": "Extracted text from the image",
  "words": [
    {
      "text": "word",
      "confidence": 0.95,
      "bbox": [0.1, 0.2, 0.3, 0.4]
    },
    ...
  ]
}
```

## Example Usage

Using curl:

```bash
curl -X POST -F "file=@path/to/your/image.jpg" http://localhost:8000/ocr
```

Using Python requests:

```python
import requests

url = "http://localhost:8000/ocr"
files = {"file": open("path/to/your/image.jpg", "rb")}
response = requests.post(url, files=files)
result = response.json()
print(result)
```

## API Documentation

Interactive API documentation is available at http://localhost:8000/docs when the server is running. 