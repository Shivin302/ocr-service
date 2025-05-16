#!/usr/bin/env python
import requests
import os
import time
IMAGES_FOLDER = "/home/ubuntu/ocr/all_images"
URL = "http://localhost:8000/"
API_URL = URL + "/ocr"
METRICS_URL = URL + "/metrics"

def traffic_sim():
    start_time = time.time()
    all_image_paths = [os.path.join(IMAGES_FOLDER, f) for f in os.listdir(IMAGES_FOLDER)]
    
    for image_path in all_image_paths[:3]:
        print("Sending image:", image_path)
        with open(image_path, 'rb') as img:
            files = {'file': img}
            response = requests.post(API_URL, files=files)
        result = response.json()
        ocr_output, request_time = result["pred"], result["request_time"]
        
        if response.status_code != 200:
            print(f"ERROR! Status code: {response.status_code}")
        else:
            print("Output:", ocr_output.keys(), "Request time:", request_time)

    end_time = time.time()
    print(f"Time taken to do OCR: {end_time - start_time:.3f} seconds")

    response = requests.get(METRICS_URL)
    print(response.text)

if __name__ == "__main__":
    traffic_sim() 
