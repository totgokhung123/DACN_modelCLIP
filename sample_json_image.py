import os
import easyocr
from PIL import Image, UnidentifiedImageError
import json

def detect_text_in_images(json_path, output_json_path):
    """Nhận diện văn bản cho từng ảnh từ JSON metadata và cập nhật thông tin detections."""
    reader = easyocr.Reader(['vi'], gpu=True)  # Sử dụng GPU nếu có

    # Đọc file JSON
    with open(json_path, 'r', encoding='utf-8') as file:
        samples = json.load(file)

    for sample in samples:
        file_path = sample.get("filepath")
        metadata = sample.get("metadata")
        if not file_path or not metadata:
            continue

        try:
            results = reader.readtext(file_path, detail=1)
            detections = [
                {
                    "label": text,
                    "bounding_box": [
                        bbox[0][0] / metadata["width"],
                        bbox[0][1] / metadata["height"],
                        (bbox[2][0] - bbox[0][0]) / metadata["width"],
                        (bbox[2][1] - bbox[0][1]) / metadata["height"],
                    ],
                    "confidence": prob,
                }
                for bbox, text, prob in results
            ]
            sample["detections"]["detections"] = detections

        except Exception as e:
            print(f"Lỗi khi xử lý nhận diện cho file {file_path}: {e}")

    # Lưu lại file JSON sau khi nhận diện
    with open(output_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(samples, json_file, ensure_ascii=False, indent=4)
    print(f"Detections đã được cập nhật và lưu vào {output_json_path}")

# Sử dụng hàm nhận diện khi cần
json_metadata_path = 'output_metadata.json'
output_with_detections = 'output_with_detections.json'
detect_text_in_images(json_metadata_path, output_with_detections)
