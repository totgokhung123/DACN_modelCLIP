from flask import Flask, request, render_template, send_from_directory, Response,send_file,url_for, jsonify
import os
from io import BytesIO
import numpy as np
import clip
import torch
from PIL import Image
from tqdm import tqdm
import requests
import base64
import re
import json
from word_processing import VietnameseTextProcessor
from pathlib import Path
FRAMES_DIR = "E:\\THIHE\\testfitty one\\SegmentVideo\\seg1\\SegmentVideo"
FRAMES_JSON = "output_samples.json" 
all = sorted([f.name for f in Path(FRAMES_DIR).iterdir()], key=lambda x: int(Path(x).stem))
print(all)

def load_frames_from_json(json_path):
    """Load danh sách tên file từ file JSON."""
    with open(json_path, 'r', encoding='utf-8') as file:
        samples = json.load(file)
    # Trả về danh sách tên file (không bao gồm đường dẫn đầy đủ)
    return [os.path.basename(sample["filepath"]) for sample in samples if "filepath" in sample]

a  = sorted(load_frames_from_json(FRAMES_JSON), key=lambda x: int(Path(x).stem))
print("frame trong JSON")
print(a)
k =len(load_frames_from_json(FRAMES_JSON))
print("so luong frame ")
print(k)