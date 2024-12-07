import easyocr
import json
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
# Tạo đối tượng reader của EasyOCR
reader = easyocr.Reader(['vi'])

# Hàm để lấy kích thước ảnh
def get_image_metadata(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")
    height, width, channels = img.shape
    size_bytes = os.path.getsize(image_path)
    return {"width": width, "height": height, "num_channels": channels, "size_bytes": size_bytes}

# Hàm để tạo đối tượng detection
def create_detection(bbox, text, prob, metadata):
    return {
        "label": text,
        "bounding_box": [
            bbox[0][0] / metadata["width"],  # x0 (top-left)
            bbox[0][1] / metadata["height"],  # y0 (top-left)
            (bbox[2][0] - bbox[0][0]) / metadata["width"],  # width
            (bbox[2][1] - bbox[0][1]) / metadata["height"],  # height
        ],
        "confidence": prob,
    }

# Hàm xử lý nhận diện ký tự trong ảnh
def process_image(image_path, video_id, frame_id, tags=None):
    # Metadata ảnh
    metadata = get_image_metadata(image_path)

    # Nhận diện ký tự
    results = reader.readtext(image_path)

    # Tạo các detection
    detections = []
    for bbox, text, prob in results:
        detection = create_detection(bbox, text, prob, metadata)
        detections.append(detection)

    # Tạo đối tượng sample
    sample = {
        "media_type": "image",
        "filepath": image_path,
        "tags": tags or [],
        "metadata": metadata,
        "video": video_id,
        "frameid": frame_id,
        "detections": detections,
    }
    return sample

def display_image_with_boxes(image_path, detections, metadata):
    # Đọc ảnh bằng OpenCV
    img = cv2.imread(image_path)
    
    # Chuyển đổi ảnh OpenCV sang định dạng Pillow
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # Tải font chữ hỗ trợ tiếng Việt
    font_path = "arial.ttf"  # Thay bằng đường dẫn tới font hỗ trợ Unicode
    try:
        font = ImageFont.truetype(font_path, 20)  # Chọn kích thước font
    except:
        print(f"Không tìm thấy font tại {font_path}. Hãy đảm bảo font hỗ trợ Unicode tồn tại.")
        return

    # Duyệt qua các bounding box và vẽ chúng
    for detection in detections:
        bbox = detection["bounding_box"]
        label = detection["label"]
        
        # Chuyển tọa độ từ tỷ lệ thành pixel
        x0 = int(bbox[0] * metadata["width"])
        y0 = int(bbox[1] * metadata["height"])
        w = int(bbox[2] * metadata["width"])
        h = int(bbox[3] * metadata["height"])
        
        # Vẽ bounding box
        draw.rectangle([x0, y0, x0 + w, y0 + h], outline="green", width=2)
        
        # Vẽ nhãn (label)
        draw.text((x0, y0 - 25), label, fill="green", font=font)
    
    # Chuyển đổi ảnh từ Pillow về OpenCV
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    # Lưu ảnh hoặc hiển thị
    output_image_path = "output_image_with_boxes.jpg"
    cv2.imwrite(output_image_path, img)
    print(f"Ảnh đã lưu tại: {output_image_path}")


# Đường dẫn ảnh và thông tin liên quan
image_path = 'E:\\THIHE\\testfitty one\\SegmentVideo\\seg1\\SegmentVideo\\621.jpg'
video_id = "SegmentVideo"
frame_id = "532"
tags = ["MainData"]

# Xử lý ảnh và tạo JSON
sample = process_image(image_path, video_id, frame_id, tags)

# Lưu JSON
output_json_path = "output_sample.json"
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(sample, f, ensure_ascii=False, indent=4)

# Hiển thị bounding box trên ảnh
display_image_with_boxes(image_path, sample["detections"], sample["metadata"])

print(f"Đã lưu JSON tại: {output_json_path}")