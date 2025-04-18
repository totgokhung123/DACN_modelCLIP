import os
import easyocr
from PIL import Image, UnidentifiedImageError
import json
from concurrent.futures import ThreadPoolExecutor

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

    frame_id = os.path.splitext(os.path.basename(file_path))[0]

    try:
        # Nhận diện văn bản bằng EasyOCR
        results = reader.readtext(file_path, detail=1)
        detections = [
            {
                "label": text,
                "bounding_box": [
                    bbox[0][0] / metadata["width"],  # x0 (top-left)
                    bbox[0][1] / metadata["height"], # y0 (top-left)
                    (bbox[2][0] - bbox[0][0]) / metadata["width"],  # width
                    (bbox[2][1] - bbox[0][1]) / metadata["height"], # height
                ],
                "confidence": prob,
            }
            for bbox, text, prob in results
        ]

        return {
            "id": frame_id,  
            "media_type": "image",
            "filepath": file_path,
            "tags": ["MainData"],
            "metadata": metadata,
            "video": "seg1",  
            "frameid": frame_id,
            "detections": {"detections": detections},
            "frameidx": int(frame_id) if frame_id.isdigit() else None,
        }
    except Exception as e:
        print(f"Lỗi khi xử lý file {file_path}: {e}")
        return None

def process_images_in_folder(folder_path, output_json_path, max_workers=4):
    """Xử lý tất cả các ảnh trong thư mục và lưu thông tin vào JSON."""
    reader = easyocr.Reader(['vi'], gpu=True)  

    all_samples = []
    files = [
        os.path.join(folder_path, file_name)
        for file_name in os.listdir(folder_path)
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(lambda file: process_image(file, reader), files)
        all_samples.extend(filter(None, results))

    # Lưu tất cả thông tin vào file JSON
    with open(output_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(all_samples, json_file, ensure_ascii=False, indent=4)

    print(f"Thông tin đã được lưu vào {output_json_path}")

# Sử dụng hàm
folder_path = 'E:\\THIHE\\testfitty one\\SegmentVideo\\seg1\\SegmentVideo'
output_json_path = 'output_samples.json'
process_images_in_folder(folder_path, output_json_path, max_workers=8)
