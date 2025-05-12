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

def create_sample(file_path, folder_name):
    """Tạo sample JSON cơ bản từ file ảnh."""
    metadata = get_image_metadata(file_path)
    if not metadata:
        return None

    frame_id = os.path.splitext(os.path.basename(file_path))[0]

    return {
        "id": frame_id,
        "media_type": "image",
        "filepath": file_path,
        "tags": ["MainData"],
        "metadata": metadata,
        "video": folder_name,  # Tên folder chứa ảnh
        "frameid": frame_id,
        "detections": {"detections": []},  # Chưa nhận diện
        "frameidx": int(frame_id) if frame_id.isdigit() else None,
    }

def process_images_in_folder(folder_path, output_json_path, max_workers=4):
    """Lưu thông tin metadata cơ bản của tất cả các ảnh trong thư mục vào JSON."""
    folder_name = os.path.basename(folder_path)
    all_samples = []
    files = [
        os.path.join(folder_path, file_name)
        for file_name in os.listdir(folder_path)
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(lambda file: create_sample(file, folder_name), files)
        all_samples.extend(filter(None, results))

    # Lưu tất cả thông tin vào file JSON
    with open(output_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(all_samples, json_file, ensure_ascii=False, indent=4)

    print(f"Metadata đã được lưu vào {output_json_path}")

# Sử dụng hàm lưu metadata
folder_path = 'E:\\THIHE\\testfitty one\\SegmentVideo\\seg1\\SegmentVideo'
output_json_path = 'output_metadata.json'
process_images_in_folder(folder_path, output_json_path, max_workers=8)