#!/usr/bin/env python
import requests

def test_ocr():
    image_path = "test_image.jpg"
    url = "http://localhost:8000/ocr"
    
    print(f"Sending {image_path} to {url}")
    
    with open(image_path, 'rb') as img:
        files = {'file': img}
        response = requests.post(url, files=files)
    
    print(f"Status code: {response.status_code}")
    print(response.text[:100])

if __name__ == "__main__":
    test_ocr() 
