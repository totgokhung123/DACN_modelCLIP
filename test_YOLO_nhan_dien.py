from ultralytics import YOLO

# Tải mô hình YOLO được huấn luyện sẵn
model = YOLO('yolov8x.pt')  # Chọn mô hình phù hợp, ví dụ: yolov8n, yolov8s

# Train the model
train_results = model.train(
    data="coco8.yaml",  # path to dataset YAML
    epochs=100,  # number of training epochs
    imgsz=640,  # training image size
    device="cpu",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)

# Evaluate model performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model('E:\\THIHE\\testfitty one\\SegmentVideo\\seg1\\SegmentVideo\\10343.jpg')
results[0].show()

# Export the model to ONNX format
path = model.export(format="onnx") 
# # Dự đoán trên ảnh
# results = model('E:\\THIHE\\testfitty one\\SegmentVideo\\seg1\\SegmentVideo\\10343.jpg')

# # Hiển thị kết quả
# results[0].show()
print(results)