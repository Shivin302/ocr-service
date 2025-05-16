#!/bin/bash

# Test script using curl to send an image to the OCR API

echo "Testing OCR API with curl..."
echo "Sending test_image.jpg to http://localhost:8000/ocr"

curl -v -F "file=@test_image.jpg" http://localhost:8000/ocr

echo
echo "Done!" 