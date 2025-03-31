
# DACN_modelCLIP

DACN_modelCLIP là một ứng dụng truy vấn sự kiện trong video dựa trên mô tả văn bản hoặc hình ảnh. Dự án sử dụng mô hình CLIP để tìm kiếm khung hình liên quan đến truy vấn và hiển thị kết quả trên giao diện web.

## 🚀 Tính năng chính
- **Tìm kiếm theo văn bản**: Người dùng nhập mô tả văn bản để tìm các khung hình phù hợp nhất trong video.
- **Tìm kiếm theo hình ảnh**: Người dùng có thể tải lên hoặc nhập URL hình ảnh để tìm các khung hình tương tự.
- **Phát lại video từ khung hình được tìm thấy**: Hiển thị video từ khung hình liên quan đến truy vấn.

## 🛠 Công nghệ sử dụng
- **Python & Flask**: Xây dựng backend và API tìm kiếm.
- **CLIP (ViT-B/32)**: Mô hình nhận diện hình ảnh và văn bản của OpenAI.
- **HTML, CSS, JavaScript**: Giao diện người dùng đơn giản nhưng hiệu quả.
- **OpenCV & NumPy**: Xử lý video và khung hình.
- **easyocr**: thư viện cung cấp detection ký tự.
- **YOLOv8**: model nhận diện Object.
- **Scenedetect**: thư viện phân tách khung hình frame theo ngưỡng.

## 🔧 Cài đặt và chạy project
### 1️⃣ Cài đặt môi trường ảo (tuỳ chọn)
```bash
python -m venv venv
source venv/bin/activate  # Trên macOS/Linux
venv\Scripts\activate     # Trên Windows
```

### 2️⃣ Cài đặt thư viện cần thiết
```bash
pip install -r requirements.txt
```

### 3️⃣ Chạy ứng dụng
```bash
python app.py
```

### 4️⃣ Mở trình duyệt và truy cập
```
http://127.0.0.1:5000/
```

## 📌 Sử dụng
1. **Tìm kiếm bằng văn bản**: Nhập mô tả sự kiện trong video và nhấn "Search".
2. **Tìm kiếm bằng hình ảnh**: Nhập URL hoặc tải lên hình ảnh để tìm các khung hình tương tự.
3. **Xem kết quả**: Nhấp vào ảnh kết quả để phát video từ khung hình tương ứng.

## 🤝 Đóng góp
Nếu bạn muốn cải thiện dự án, hãy fork repo và gửi pull request! Mọi đóng góp đều được hoan nghênh.

---
🎯 **DACN_modelCLIP** giúp bạn truy vấn video nhanh chóng và hiệu quả bằng mô tả ngôn ngữ tự nhiên hoặc hình ảnh! 🚀

#Preview demo:
- Truy vấn mô tả bằng tiếng anh
![image](https://github.com/user-attachments/assets/d639f4a9-add1-4d27-ac61-7334a753aedf)
- Truy vấn mô tả bằng tiếng việt
![image](https://github.com/user-attachments/assets/68029a4e-6615-4723-9a1e-fed005728dd8)
- Truy vấn mô tả bằng hình ảnh tương tự (Local)
![image](https://github.com/user-attachments/assets/24ee7734-12d2-4a1f-ae5c-0b0ef10178eb)
- Truy vấn mô tả bằng hình ảnh tương tự (URL)
![image](https://github.com/user-attachments/assets/63a9b512-cc19-48c3-9658-22fb1be94ab8)
- Detect Object
![image](https://github.com/user-attachments/assets/688e5b39-6d22-4fa4-9f20-d8d1c445ea21)
- Detect Character
![image](https://github.com/user-attachments/assets/7b0de2cb-76fe-460b-b670-133c18e76dd4)
- Segmentation video to frame
![image](https://github.com/user-attachments/assets/b3e72453-c762-4b54-9e19-e9b4c23c8390)

