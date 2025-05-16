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

from doctr.models.detection._utils import _remove_padding
from doctr.models.preprocessor import PreProcessor
from doctr.models.utils import set_device_and_dtype


IMAGES_FOLDER = "/home/ubuntu/ocr/all_images"

# @profile
def all_forward(predictor, pages, return_maps=False, **kwargs):
    # Extract parameters from the preprocessor
    preserve_aspect_ratio = predictor.pre_processor.resize.preserve_aspect_ratio
    symmetric_pad = predictor.pre_processor.resize.symmetric_pad
    assume_straight_pages = predictor.model.assume_straight_pages

    # Dimension check
    if any(page.ndim != 3 for page in pages):
        raise ValueError("incorrect input shape: all pages are expected to be multi-channel 2D images.")

    processed_batches = predictor.pre_processor(pages)
    _params = next(predictor.model.parameters())
    predictor.model, processed_batches = set_device_and_dtype(
        predictor.model, processed_batches, _params.device, _params.dtype
    )
    predicted_batches = [
        predictor.model(batch, return_preds=True, return_model_output=True, **kwargs) for batch in processed_batches
    ]
    # Remove padding from loc predictions
    preds = _remove_padding(
        pages,
        [pred for batch in predicted_batches for pred in batch["preds"]],
        preserve_aspect_ratio=preserve_aspect_ratio,
        symmetric_pad=symmetric_pad,
        assume_straight_pages=assume_straight_pages,  # type: ignore[arg-type]
    )

    if return_maps:
        seg_maps = [
            pred.permute(1, 2, 0).detach().cpu().numpy() for batch in predicted_batches for pred in batch["out_map"]
        ]
        return preds, seg_maps
    return preds


def main():
    start_time = time.time()
    device = "cuda:0"
    predictor = ocr_predictor('db_resnet50', 'vitstr_base', pretrained=True, det_bs=20, reco_bs=1024).to(device)
    det_predictor = predictor.det_predictor
    all_image_paths = [os.path.join(IMAGES_FOLDER, f) for f in os.listdir(IMAGES_FOLDER)]


    # pages = DocumentFile.from_images(all_image_paths[0])
    # result = det_predictor(pages, return_maps=True)
    compile_time = time.time()
    print(f"Time taken to compile: {compile_time - start_time:.3f} seconds")


    pages = DocumentFile.from_images(all_image_paths[:100])
    # result = det_predictor(pages, return_maps=True)
    result = all_forward(det_predictor, pages, return_maps=True)

   
    end_time = time.time()
    print(f"Time taken to do OCR: {end_time - compile_time:.3f} seconds")

if __name__ == "__main__":
    main()



