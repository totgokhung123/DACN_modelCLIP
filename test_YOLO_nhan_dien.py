from ultralytics import YOLO

# Tải mô hình YOLO được huấn luyện sẵn
model = YOLO('yolov8x.pt')  # Chọn mô hình phù hợp, ví dụ: yolov8n, yolov8s

# Dự đoán trên ảnh
results = model('D:\\code\\projects\\git\\Data\\khung_hinh_1_1\\27449.jpg')

# Hiển thị kết quả
results[0].show()
