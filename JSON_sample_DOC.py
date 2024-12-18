import os
import easyocr
from PIL import Image, UnidentifiedImageError
import json
import uuid  # Để tạo id độc nhất
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


def get_image_metadata(image_path):
    """Lấy metadata cơ bản của ảnh."""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            return {
                "size_bytes": os.path.getsize(image_path),
                "mime_type": Image.MIME.get(img.format, "unknown"),
                "width": width,
                "height": height,
                "num_channels": len(img.getbands()),
            }
    except (UnidentifiedImageError, OSError) as e:
        print(f"Không thể đọc metadata từ file {image_path}: {e}")
        return None


def process_image(file_path, reader):
    """Xử lý một ảnh duy nhất và trả về sample JSON."""
    metadata = get_image_metadata(file_path)
    if not metadata:
        return None

    try:
        # Nhận diện văn bản bằng EasyOCR
        results = reader.readtext(file_path, detail=1)
        detections = [
            {
                "label": text,
                "bounding_box": [
                    bbox[0][0] / metadata["width"],  # x0
                    bbox[0][1] / metadata["height"],  # y0
                    (bbox[2][0] - bbox[0][0]) / metadata["width"],  # width
                    (bbox[2][1] - bbox[0][1]) / metadata["height"],  # height
                ],
                "confidence": prob,
            }
            for bbox, text, prob in results
        ]

        return {
            "id": str(uuid.uuid4()), 
            "media_type": "image",
            "filepath": file_path,
            "tags": ["MainData"],
            "metadata": metadata,
            "video": "seg1",
            "frameid": os.path.basename(file_path),
            "detections": {"detections": detections},
            "frameidx": int(os.path.splitext(os.path.basename(file_path))[0]) 
                        if os.path.splitext(os.path.basename(file_path))[0].isdigit() else None,
        }
    except Exception as e:
        print(f"Lỗi khi xử lý file {file_path}: {e}")
        return None


def process_images_in_folder(folder_path, output_json_path, max_workers=4):
    reader = easyocr.Reader(['vi'], gpu=True)
    if os.path.exists(output_json_path):
        with open(output_json_path, 'r', encoding='utf-8') as json_file:
            existing_data = json.load(json_file)
    else:
        existing_data = []
    files = [
        os.path.join(folder_path, file_name)
        for file_name in os.listdir(folder_path)
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(lambda f: process_image(f, reader), files), 
                            total=len(files), desc="Processing Frames"))
    new_data = [res for res in results if res]
    existing_data.extend(new_data)
    with open(output_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(existing_data, json_file, ensure_ascii=False, indent=4)
    print(f"Updated JSON has been saved to {output_json_path}")