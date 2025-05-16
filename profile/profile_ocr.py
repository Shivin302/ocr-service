import io
from typing import List
import logging
import sys

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, UnidentifiedImageError
import doctr
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import time
import os

import torch
from torch import nn

from doctr.io.elements import Document
from doctr.models._utils import get_language
from doctr.models.detection.predictor import DetectionPredictor
from doctr.models.recognition.predictor import RecognitionPredictor
from doctr.utils.geometry import detach_scores

IMAGES_FOLDER = "/home/ubuntu/ocr/all_images"

@profile
def all_forward(predictor, pages, **kwargs):
    # Dimension check
    if any(page.ndim != 3 for page in pages):
        raise ValueError("incorrect input shape: all pages are expected to be multi-channel 2D images.")

    origin_page_shapes = [page.shape[:2] if isinstance(page, np.ndarray) else page.shape[-2:] for page in pages]

    # Localize text elements
    loc_preds, out_maps = predictor.det_predictor(pages, return_maps=True, **kwargs)

    # Detect document rotation and rotate pages
    # seg_maps = [
    #     np.where(out_map > getattr(predictor.det_predictor.model.postprocessor, "bin_thresh"), 255, 0).astype(np.uint8)
    #     for out_map in out_maps
    # ]

    out_maps_array = np.array(out_maps)
    bin_thresh = getattr(predictor.det_predictor.model.postprocessor, "bin_thresh")
    seg_maps_array = np.where(out_maps_array > bin_thresh, 255, 0).astype(np.uint8)
    seg_maps = list(seg_maps_array)


    if predictor.detect_orientation:
        general_pages_orientations, origin_pages_orientations = predictor._get_orientations(pages, seg_maps)
        orientations = [
            {"value": orientation_page, "confidence": None} for orientation_page in origin_pages_orientations
        ]
    else:
        orientations = None
        general_pages_orientations = None
        origin_pages_orientations = None
    if predictor.straighten_pages:
        pages = predictor._straighten_pages(pages, seg_maps, general_pages_orientations, origin_pages_orientations)
        # update page shapes after straightening
        origin_page_shapes = [page.shape[:2] for page in pages]

        # Forward again to get predictions on straight pages
        loc_preds = predictor.det_predictor(pages, **kwargs)

    assert all(len(loc_pred) == 1 for loc_pred in loc_preds), (
        "Detection Model in ocr_predictor should output only one class"
    )

    loc_preds = [list(loc_pred.values())[0] for loc_pred in loc_preds]
    # Detach objectness scores from loc_preds
    loc_preds, objectness_scores = detach_scores(loc_preds)
    # Check whether crop mode should be switched to channels first
    channels_last = len(pages) == 0 or isinstance(pages[0], np.ndarray)

    # Apply hooks to loc_preds if any
    for hook in predictor.hooks:
        loc_preds = hook(loc_preds)

    # Crop images
    crops, loc_preds = predictor._prepare_crops(
        pages,
        loc_preds,
        channels_last=channels_last,
        assume_straight_pages=predictor.assume_straight_pages,
        assume_horizontal=predictor._page_orientation_disabled,
    )
    # Rectify crop orientation and get crop orientation predictions
    crop_orientations = []
    if not predictor.assume_straight_pages:
        crops, loc_preds, _crop_orientations = predictor._rectify_crops(crops, loc_preds)
        crop_orientations = [
            {"value": orientation[0], "confidence": orientation[1]} for orientation in _crop_orientations
        ]

    # Identify character sequences
    word_preds = predictor.reco_predictor([crop for page_crops in crops for crop in page_crops], **kwargs)
    if not crop_orientations:
        crop_orientations = [{"value": 0, "confidence": None} for _ in word_preds]

    boxes, text_preds, crop_orientations = predictor._process_predictions(loc_preds, word_preds, crop_orientations)

    if predictor.detect_language:
        languages = [get_language(" ".join([item[0] for item in text_pred])) for text_pred in text_preds]
        languages_dict = [{"value": lang[0], "confidence": lang[1]} for lang in languages]
    else:
        languages_dict = None

    out = predictor.doc_builder(
        pages,
        boxes,
        objectness_scores,
        text_preds,
        origin_page_shapes,
        crop_orientations,
        orientations,
        languages_dict,
    )
    return out

def main():
    device = "cuda:0"
    # predictor = ocr_predictor('db_resnet50', 'crnn_vgg16_bn', pretrained=True, det_bs=20, reco_bs=1024).to(device)
    predictor = ocr_predictor('db_resnet50', 'vitstr_base', pretrained=True, det_bs=20, reco_bs=1024).to(device)
    
    start_time = time.time()
    all_image_paths = [os.path.join(IMAGES_FOLDER, f) for f in os.listdir(IMAGES_FOLDER)]
    pages = DocumentFile.from_images(all_image_paths[:100])
    result = all_forward(predictor, pages)

    # for image_path in all_image_paths[:5]:
    #     print("Running OCR on image:", image_path)
    #     page = DocumentFile.from_images(image_path)
    #     print(page[0].shape)
    #     # page = torch.tensor(page, device=device)
    #     # result = predictor(page)
    #     result = all_forward(predictor, page)
   
    end_time = time.time()
    print(f"Time taken to do OCR: {end_time - start_time:.3f} seconds")


    # pages_1584x1224 = []
    # for p in pages:
    #     if p.shape == (1584, 1224, 3):
    #         pages_1584x1224.append(p)
    # print(f"Found {len(pages_1584x1224)} pages of size 1584x1224")
    # pages_tensor = torch.tensor(np.array(pages_1584x1224), device=device)
    # print(pages_tensor.shape)

if __name__ == "__main__":
    main()



