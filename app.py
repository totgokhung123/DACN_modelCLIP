from flask import Flask, request, render_template, send_from_directory, Response,send_file,url_for, jsonify,abort
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
from pymongo import MongoClient  
from flask_caching import Cache  
from bson import ObjectId 
import urllib.parse  
import cv2
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

FRAMES_JSON = "D:\\code\\projects\\git\\DACN_modelCLIP-3\\output_samples.json" 
EMBEDDINGS_FILE = "D:\\code\\projects\\git\\DACN_modelCLIP-3\\embedding\\image_embeddings.npy"
text_processor = VietnameseTextProcessor()

embeddings = np.load(EMBEDDINGS_FILE)

def load_frames_from_json(json_path):
    """Load danh sách tên file từ file JSON."""
    with open(json_path, 'r', encoding='utf-8') as file:
        samples = json.load(file)

    return [os.path.basename(sample["filepath"]) for sample in samples if "filepath" in sample]
def load_frames_mapping_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        return {Path(sample["filepath"]).name: sample["filepath"] for sample in data}

FRAMES_MAPPING = load_frames_mapping_from_json(FRAMES_JSON)
app = Flask(__name__)

def load_tags_from_file():  
    """Hàm lấy danh sách tag duy nhất từ MongoDB."""    
    tags = frames_collection.distinct("tags")   
    return [{"tag": tag} for tag in tags] 

cache = Cache(app, config={'CACHE_TYPE': 'SimpleCache'})  

MONGODB_URI = 'mongodb://localhost:27017/'  
DATABASE_NAME = 'testmongoDACN'  
COLLECTION_NAME = 'frames'   

client = MongoClient(MONGODB_URI)  
db = client[DATABASE_NAME]  
frames_collection = db[COLLECTION_NAME]  

@cache.cached(timeout=60)  
def load_frames_from_mongodb():  
    """Lấy danh sách các frame từ MongoDB."""  
    frames = list(frames_collection.find({}, {"frameid": 1, "filepath": 1}))  
    print("Frames found:", frames) 
    return [{"frameid": str(frame['frameid']), "filepath": frame['filepath']} for frame in frames if "filepath" in frame]  
@app.route('/frames/<path:filename>')  
def serve_frame(filename):  
    """Phục vụ file hình ảnh từ MongoDB."""  
    frame_data = frames_collection.find_one({"filepath": filename}, {"_id": 0})  
    if frame_data:  
        full_path = frame_data['filepath']  
        if os.path.exists(full_path):  
            return send_file(full_path, mimetype='image/jpeg')  
        else:  
            abort(404, description=f"File {filename} not found.")  
    else:  
        abort(404, description=f"File {filename} not found in database.")  

@app.route("/", methods=["GET", "POST"])  
def index():  
    """Trang chính để hiển thị danh sách frame và xử lý tìm kiếm."""  
    all_frames = load_frames_from_mongodb()  
    all_frames = sorted(all_frames, key=lambda x: int(Path(x['filepath']).stem))  
    tags = load_tags_from_file() 
    if request.method == "POST":  
        query = request.form.get("query")  
        top_k = parse_top_k(request.form.get("top_k"))  

        processed_text = text_processor.preprocess_and_translate(query)  
        print("Câu truy vấn đã xử lý:", processed_text)  
        top_frames = search_top_frames(processed_text, top_k)  
        return render_template("index.html", query=query, top_frames=top_frames, tags=tags)  
    
    tag_name = request.args.get("tag_name")  
    if tag_name:  
        filtered_frames = list(frames_collection.find({"tags": tag_name}, {"frameid": 1, "filepath": 1}))  
        filtered_frames = [{"frameid": str(frame['frameid']), "filepath": frame['filepath']} for frame in filtered_frames]  
        return render_template("index.html", query=None, top_frames=filtered_frames, current_tag=tag_name, tags=tags)  
     
    return render_template("index.html", query=None, top_frames=all_frames, tags=tags)    

def parse_top_k(top_k_value):  
    """giới hạn giá trị top_k."""  
    try:  
        top_k = int(float(top_k_value))  
    except ValueError:  
        top_k = 200  
    return top_k  


@app.route("/save-tags", methods=["POST"])  
def save_tags():  
    """Route để lưu tag cho các frame được chọn."""  
    selected_frame_ids = request.form.getlist("frame_ids")  
    tags_to_add = request.form.get("tags")  
    print(selected_frame_ids)
    print(tags_to_add)
    if not tags_to_add:  
        return jsonify({"message": "Tags cannot be empty"}), 400  
    if not selected_frame_ids:  
        return jsonify({"message": "No frames selected."}), 400  
    tags_list = [tag.strip() for tag in tags_to_add.split(",")]  
    for frame_id in selected_frame_ids:  
        print(f"Updating frame with ID: {frame_id} and adding tags: {tags_list}")  
        result = frames_collection.update_one(  
            {"frameid": frame_id}, 
            {"$addToSet": {"tags": {"$each": tags_list}}} 
        )  
        if result.modified_count == 0:  
            print(f"No updates made for frame ID: {frame_id}. Maybe it doesn't exist or tags already present.")  
        else:  
            print(f"Tags successfully added to frame ID: {frame_id}")  
    return jsonify({"message": "Tags saved successfully"}), 200  
@app.route("/tags")  
def get_tags():  
    """Route để lấy danh sách tag duy nhất từ frame."""  
    tags = frames_collection.distinct("tags")  
    return jsonify(tags), 200  

@app.route("/frames")  
def get_frames_by_tag():  
    """Route để lấy frame theo tag."""  
    tag_name = request.args.get("tag")  
    if tag_name:  
        frames = list(frames_collection.find({"tags": tag_name}, {"_id": 0, "filepath": 1}))  
        return jsonify([frame['filepath'] for frame in frames]), 200  
    return jsonify([]), 200  

@app.route("/get_frame_info/<frame_id>")  
def get_frame_info(frame_id):  
    """Lấy thông tin chi tiết của một frame từ MongoDB"""  
    try:    
        frame = frames_collection.find_one({"_id": ObjectId(frame_id)})  
        
        if not frame:  
            return jsonify({"error": "Frame not found"}), 404  

        frame_info = {  
            "id": str(frame['_id']),  
            "filepath": frame['filepath'],  
            "metadata": frame.get('metadata', {}),  
            "tags": frame.get('tags', []),  
            "detections": {  
                "text_detections": frame.get('text_detections', {"detections": []}),  
                "object_detections": frame.get('object_detections', {"detections": []})  
            },  
            "video": frame.get('video', ''),  
            "frameid": frame.get('frameid', '')  
        }  
        
        return jsonify(frame_info)  
    
    except Exception as e:  
        print(f"Error fetching frame info: {e}")  
        return jsonify({"error": "Internal server error"}), 500
# frames_data = load_frame_data()

# @app.route('/get_frame_info/<frameidx>')
# def get_frame_info(frameidx):
#     # Tìm frame tương ứng theo frameidx
#     frame_info = next((frame for frame in frames_data if str(frame['frameidx']) == frameidx), None)

#     if frame_info:
#         return jsonify(frame_info)
#     else:
#         return jsonify({"error": "Frame not found"}), 404

# @app.route('/frames/<path:filename>')
# def serve_frame(filename):
#     if filename in FRAMES_MAPPING:
#         full_path = FRAMES_MAPPING[filename]
#         return send_file(full_path, mimetype="image/jpeg")  
#     else:
#         abort(404, description=f"File {filename} not found.")

@app.route('/reset')
def reset():
    all = sorted(load_frames_from_json(FRAMES_JSON), key=lambda x: int(Path(x).stem))
    tags = load_tags_from_file()
    return render_template("index.html", query=None, top_frames=all, tags=[t['tag'] for t in tags])

def load_tags_from_file():
    with open('tags.json', 'r',  encoding='utf-8') as file:
        return json.load(file)
    

# @app.route("/", methods=["GET", "POST"])
# def index():
#     all = sorted(load_frames_from_json(FRAMES_JSON), key=lambda x: int(Path(x).stem))
#     tags = load_tags_from_file()

#     if request.method == "POST":
#         query = request.form.get("query")
#         top_k = request.form.get("top_k", 50)

#         try:
#             top_k = int(float(top_k))
#         except ValueError:
#             top_k = 50

#         total_frames = len(all)    
#         top_k = min(max(1, top_k), total_frames)
#         processed_text = text_processor.preprocess_and_translate(query)
#         print("Câu truy vấn đã xử lý:", processed_text)
#         top_frames = search_top_frames(processed_text, top_k)
#         return render_template("index.html", query=query, top_frames=top_frames,tags=[t['tag'] for t in tags])
    
#     # Xử lý khi có GET request (lấy `tag_name` từ URL)
#     tag_name = request.args.get("tag_name")
#     if tag_name:
#         # Lọc ảnh theo tag
#         tag = next((t for t in tags if t["tag"] == tag_name), None)
#         if tag:
#             filtered_frames = tag["frames"]
#         else:
#             filtered_frames = []  
#         return render_template("index.html", query=None, top_frames=filtered_frames, current_tag=tag_name, tags=[t['tag'] for t in tags])
#     # Nếu không có tag_name, hiển thị tất cả các frame
#     return render_template("index.html", query=None, top_frames=all, tags=[t['tag'] for t in tags])
@app.route("/search_image", methods=["POST"])  
def search_image():  
    tags = load_tags_from_file()  
    top_k = int(request.form.get("top_k", 50))  
    try:  
        top_k = int(float(top_k))  
    except ValueError:  
        top_k = 50  

    total_frames = frames_collection.count_documents({})  
    top_k = min(max(1, top_k), total_frames)  

    image_files = request.files.getlist("image_files")   
    image_url = request.form.get("image_url")  
    image_features_list = []  

    if image_files and image_files[0].filename:  
        for image_file in image_files:  
            image_features = process_image_query(image_file, None)  
            if image_features is not None:  
                image_features_list.append(image_features)  

    if image_url:  
        image_features = process_image_query(None, image_url)  
        if image_features is not None:  
            image_features_list.append(image_features)  
 
    if image_features_list:  

        image_features_combined = np.mean(np.vstack(image_features_list), axis=0, keepdims=True)  
        top_frames = search_top_frames_by_image(image_features_combined, top_k)  
    else:  
        top_frames = []  

    return render_template("index.html", query=None, top_frames=top_frames, tags=[t['tag'] for t in tags])  

def process_image_query(image_file, image_url):  
    """Xử lý và trích xuất features từ ảnh"""  
    try:  
        if image_file:  
            image = Image.open(image_file).convert("RGB")  
        elif image_url:  
            if re.match(r"^data:image\/[a-zA-Z]+;base64,", image_url):  
                base64_str = image_url.split(",", 1)[1]  
                image_data = base64.b64decode(base64_str)  
                image = Image.open(BytesIO(image_data)).convert("RGB")  
            else:  
                response = requests.get(image_url)  
                response.raise_for_status()  
                image = Image.open(BytesIO(response.content)).convert("RGB")  
        else:  
            return None  

        image_input = preprocess(image).unsqueeze(0).to(device)  
        with torch.no_grad():  
            image_features = model.encode_image(image_input).cpu().numpy()  

        image_features = image_features / np.linalg.norm(image_features, axis=-1, keepdims=True)  
        return image_features  
    
    except Exception as e:  
        print(f"Error processing image: {e}")  
        return None  

def search_top_frames_by_image(image_features, top_k):  
 
    frame_cursor = frames_collection.find({}, {"_id": 0, "filepath": 1, "embedding": 1})    
    frames_with_embeddings = list(frame_cursor)  
     
    embeddings = np.array([frame.get('embedding', []) for frame in frames_with_embeddings])  
    filepaths = [frame['filepath'] for frame in frames_with_embeddings]  
    embeddings = embeddings / np.linalg.norm(embeddings, axis=-1, keepdims=True)  
    similarities = np.dot(embeddings, image_features.T).flatten()  
 
    top_indices = np.argsort(similarities)[-top_k:][::-1]  
 
    return [{"id": str(frames_with_embeddings[i].get('_id', '')),   
             "filepath": filepaths[i]} for i in top_indices]  

def search_top_frames(query, top_k):  

    text_input = clip.tokenize([query]).to(device)  
    with torch.no_grad():  
        text_features = model.encode_text(text_input).cpu().numpy()  
    text_features = text_features / np.linalg.norm(text_features, axis=-1, keepdims=True)  
    frame_cursor = frames_collection.find({}, {"_id": 0, "filepath": 1, "embedding": 1})   
    frames_with_embeddings = list(frame_cursor)  
    
    embeddings = np.array([frame.get('embedding', []) for frame in frames_with_embeddings])  
    filepaths = [frame['filepath'] for frame in frames_with_embeddings]  
    
    embeddings = embeddings / np.linalg.norm(embeddings, axis=-1, keepdims=True)  
    
    similarities = np.dot(embeddings, text_features.T).flatten()  
    top_indices = np.argsort(similarities)[-top_k:][::-1]  
    return [{"id": str(frames_with_embeddings[i].get('_id', '')),   
             "filepath": filepaths[i]} for i in top_indices]

@app.route('/video_popup')
def video_popup():
    frameidx = request.args.get('frameidx', default='0')
    return render_template('video_popup.html', frameidx=frameidx)

# @app.route('/serve_video')
# def serve_video():
#     return send_file("D:\\code\\projects\\git\\Data\\video\\L01_V001.mp4", mimetype='video/mp4')

@app.route('/serve_video')  
def serve_video():  
    print("Received serve_video request")  
    
    try:  
        # Lấy frame index từ query  
        frame_idx = request.args.get('frame_idx')  
        
        # Log chi tiết  
        print(f"Received Frame Index: {frame_idx}")  
        
        # Validate input  
        if not frame_idx:  
            print("Không có thông tin frame")  
            return "Không có thông tin frame", 400  

        # Chuyển đổi sang số nguyên  
        try:  
            frame_number = int(frame_idx)  
        except ValueError:  
            print(f"Frame index không hợp lệ: {frame_idx}")  
            return "Frame index không hợp lệ", 400  

        # Danh sách video có thể  
        video_paths = [  
            r"D:\code\projects\git\Data\video\L01_V001.mp4",  
            "/path/to/video/L01_V001.mp4",  
            "./videos/L01_V001.mp4"  
        ]  

        # Tìm video tồn tại  
        video_path = next((path for path in video_paths if os.path.exists(path)), None)  

        if not video_path:  
            print("Không tìm thấy video")  
            return "Không tìm thấy video", 404  

        print(f"Selected Video Path: {video_path}")  

        # Mở video  
        video = cv2.VideoCapture(video_path)  
        
        # Kiểm tra video có mở được không  
        if not video.isOpened():  
            print("Không thể mở video")  
            return "Không thể mở video", 404  
        
        # Lấy thông số video  
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  
        fps = video.get(cv2.CAP_PROP_FPS)  
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))  
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))  
        
        print(f"Video Details:")  
        print(f"Total Frames: {total_frames}")  
        print(f"FPS: {fps}")  
        print(f"Frame Dimensions: {frame_width}x{frame_height}")  
        
        # Điều chỉnh frame number  
        frame_number = min(max(0, frame_number - 1), total_frames - 1)  
        print(f"Adjusted Frame Number: {frame_number}")  

        # Đặt vị trí frame   
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)  
        
        # Tạo thư mục tạm  
        temp_dir = './temp_frames'  
        os.makedirs(temp_dir, exist_ok=True)  
        
        # Đường dẫn video tạm  
        temp_video_path = os.path.join(temp_dir, f'video_from_frame_{frame_number}.mp4')  
        
        # Khởi tạo video writer  
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
        out = cv2.VideoWriter(  
            temp_video_path,   
            fourcc,   
            fps,   
            (frame_width, frame_height)  
        )  
        
        # Đọc và ghi video từ frame đó  
        frame_count = 0  
        max_frames = 300  # Giới hạn tối đa 10 giây video  
        
        while True:  
            ret, frame = video.read()  
            
            # Dừng nếu hết video hoặc đủ frame  
            if not ret or frame_count >= max_frames:  
                break  
            
            out.write(frame)  
            frame_count += 1  
        
        # Đóng video  
        video.release()  
        out.release()  

        print(f"Exported Video: {temp_video_path}")  
        print(f"Exported Frames: {frame_count}")  

        # Trả file video  
        response = send_file(  
            temp_video_path,   
            mimetype='video/mp4',  
            as_attachment=False  
        )  
        
        # Thêm header để disable cache  
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'  
        response.headers['Pragma'] = 'no-cache'  
        response.headers['Expires'] = '0'  
        
        return response  
    
    except Exception as e:  
        print(f"Lỗi phát video: {e}")  
        import traceback  
        traceback.print_exc()  
        return f"Lỗi phát video: {e}", 500

def extract_frame_number(frame_path):  
    """  
    Trích xuất số thứ tự frame từ đường dẫn  
    Hỗ trợ nhiều định dạng đường dẫn, bao gồm URL encode  
    """  
    import re  
    import urllib.parse  
    import os  

    print(f"Original frame path: {frame_path}")  
    
    # Giải mã URL nếu là URL encode  
    try:  
        decoded_path = urllib.parse.unquote(frame_path)  
        print(f"Decoded path: {decoded_path}")  
    except Exception as e:  
        print(f"Lỗi giải mã URL: {e}")  
        decoded_path = frame_path  

    if not decoded_path:  
        print("Không có frame path")  
        return None  
    
    # Chuẩn hóa đường dẫn  
    decoded_path = decoded_path.replace('\\', '/')  
    
    # Các pattern để trích xuất số  
    patterns = [  
        r'/(\d+)\.jpg$',          # Matching number before .jpg ở cuối đường dẫn  
        r'khung_hinh_1_1[/\\](\d+)\.jpg$',  # Matching trong thư mục khung_hinh_1_1  
        r'[/\\](\d+)\.jpg$',       # Matching number trước .jpg  
        r'(\d+)\.jpg$',            # Matching toàn bộ số trước .jpg  
        r'[/\\](\d+)[^/]*$',       # Matching number trước phần mở rộng  
        r'(\d+)$'                  # Matching pure number  
    ]  
    
    for pattern in patterns:  
        match = re.search(pattern, decoded_path)  
        if match:  
            try:  
                frame_num = int(match.group(1))  
                print(f"Matched pattern {pattern}, Frame Number: {frame_num}")  
                return frame_num  
            except (ValueError, TypeError):  
                continue  
    
    # Nếu không match, thử lấy tên file không có phần mở rộng  
    try:  
        filename = os.path.basename(decoded_path)  
        filename_without_ext = os.path.splitext(filename)[0]  
        frame_num = int(filename_without_ext)  
        print(f"Extracted from filename: {frame_num}")  
        return frame_num  
    except (ValueError, TypeError):  
        pass  
    
    print("Không tìm thấy frame number phù hợp")  
    return None  
# @app.route("/save_tags", methods=["POST"])
# def save_tags():
#     data = request.json
#     tag_name = data.get("tag", "").strip()
#     frames = data.get("frames", [])

#     if not tag_name:
#         return jsonify({"error": "Tag name is required"}), 400
#     if not frames:
#         return jsonify({"error": "No frames provided"}), 400

#     try:
#         # Tải tags hiện có
#         existing_tags = load_tags_from_file()

#         # Kiểm tra nếu tag đã tồn tại
#         for tag in existing_tags:
#             if tag["tag"] == tag_name:
#                 tag["frames"] = frames  
#                 break
#         else:
#             # Thêm tag mới
#             existing_tags.append({"tag": tag_name, "frames": frames})

#         # Lưu lại vào file
#         save_tags_to_file(existing_tags)

#         # Trả về URL chứa tag vừa lưu
#         return jsonify({"message": "Tags saved successfully", "redirect_url": url_for('index', tag_name=tag_name)}), 200
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
    
# @app.route('/load-images', methods=['GET'])
# def load_images():
#     tag_name = request.args.get('tag_name')
#     if tag_name:
#         tags = load_tags_from_file()
#         tag = next((t for t in tags if t['tag'] == tag_name), None)
#         if tag:
#             # Trả về danh sách tên ảnh theo tag
#             return jsonify({'images': tag['frames']})
#     return jsonify({'images': []})

# TAGS_FILE = "D:\\code\\projects\\git\\DACN_modelCLIP-3\\tags.json"
# # Lưu trữ tags toàn cục để sử dụng trong các route
# TAGS = load_tags_from_file()
# @app.route('/get-tags', methods=['GET'])
# def get_tags():
#     with open('tags.json', 'r',  encoding='utf-8') as file:
#         data = json.load(file)
#     return jsonify(data)

# def save_tags_to_file(tags):
#     with open(TAGS_FILE, "w") as f:
#         json.dump(tags, f)

if __name__ == "__main__":
    app.run(debug=True)