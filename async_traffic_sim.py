#!/usr/bin/env python
import os
import time
import asyncio
import aiohttp
import json
from datetime import datetime

IMAGES_FOLDER = "/home/ubuntu/ocr/all_images"
URL = "http://localhost:8000/"
API_URL = URL + "ocr"
METRICS_URL = URL + "metrics"

async def process_image(session, image_path, semaphore):
    """Process a single image with OCR API using a semaphore to control concurrency"""
    async with semaphore:
        start_time = time.time()
        image_name = os.path.basename(image_path)
        
        try:
            with open(image_path, 'rb') as img:
                data = aiohttp.FormData()
                data.add_field('file', img.read(), filename=image_name)
                
                async with session.post(API_URL, data=data) as response:
                    result = await response.json()
                    
                    end_time = time.time()
                    elapsed = round(end_time - start_time, 3)
                    
                    if response.status != 200:
                        print(f"❌ Error processing {image_name}: Status {response.status}")
                        return None
                    else:
                        ocr_output, request_time = result["pred"], result["request_time"]
                        print(f"✅ {image_name} - API: {request_time}s - Total: {elapsed}s")
                        return {
                            "image": image_name,
                            "status": response.status,
                            "api_time": request_time,
                            "total_time": elapsed
                        }
        except Exception as e:
            print(f"❌ Error processing {image_name}: {str(e)}")
            return None


async def traffic_sim(max_concurrent=5, num_images=20):
    """
    Run traffic simulation with controlled concurrency
    
    Args:
        max_concurrent: Maximum number of concurrent requests
        num_images: Number of images to process (None for all)
    """
    print(f"Starting async traffic simulation at {datetime.now().strftime('%H:%M:%S')}")
    print(f"Max concurrent requests: {max_concurrent}")
    
    start_time = time.time()
    all_image_paths = [os.path.join(IMAGES_FOLDER, f) for f in os.listdir(IMAGES_FOLDER) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]
    
    images_to_process = all_image_paths[:num_images]
    
    print(f"Processing {len(images_to_process)} images")
    
    # Get initial metrics
    async with aiohttp.ClientSession() as session:        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        # Create tasks for all images
        tasks = [process_image(session, img_path, semaphore) for img_path in images_to_process]
        
        # Execute all tasks with controlled concurrency
        results = await asyncio.gather(*tasks)
        results = [r for r in results if r is not None]  # Filter out failed requests
        # Get final metrics
        async with session.get(METRICS_URL) as response:
            if response.status == 200:
                metrics = await response.json()
                final_metrics = metrics
            else:
                print(f"❌ Error getting metrics: Status {response.status}")
                final_metrics = {}

    # Calculate statistics
    successful_requests = len(results)
    total_time = time.time() - start_time
    avg_time = sum(r["total_time"] for r in results) / successful_requests if successful_requests > 0 else 0
    
    # Print summary
    print("\n" + "="*60)
    print(f"Traffic Simulation Complete - {datetime.now().strftime('%H:%M:%S')}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Successful requests: {successful_requests}/{len(images_to_process)}")
    print(f"Average processing time per semaphore: {avg_time:.2f} seconds")
    print("\nAPI Metrics:")
    
    print("Final metrics:")
    print(json.dumps(final_metrics, indent=2))
    print("="*60)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Async OCR Traffic Simulator')
    parser.add_argument('--concurrent', type=int, default=16, help='Maximum concurrent requests')
    parser.add_argument('--images', type=int, default=20, help='Number of images to process')
    
    args = parser.parse_args()
    
    asyncio.run(traffic_sim(max_concurrent=args.concurrent, num_images=args.images)) 