from flask import Flask, request, render_template,stream_with_context, send_from_directory, Response,send_file,url_for, jsonify,abort, redirect
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
from segment_video import extract_frames_from_video
from JSON_sample_DOC import process_images_in_folder
from embedding import extract_and_save_embeddings_from_folder
import tempfile
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
import time  
from unidecode import unidecode
import faiss 
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import cv2

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

FRAMES_JSON = "E:\\Đồ án chuyên ngành\\19_12_2024\\output_samples.json" 
EMBEDDINGS_FILE = "E:\\Đồ án chuyên ngành\\19_12_2024\\embedding\\image_embeddings.npy"
text_processor = VietnameseTextProcessor()
# embeddings = np.load(EMBEDDINGS_FILE)

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
# @app.route('/frames/<path:filename>')
# def serve_frame(filename):
#     a = send_from_directory(FRAMES_DIR, filename)
#     print(a)
#     return send_from_directory(FRAMES_DIR, filename)
def load_frame_data():
    with open('output_samples.json', 'r',encoding="utf-8") as file:
        return json.load(file)
frames_data = load_frame_data()

@app.route('/get_frame_info/<frameidx>')
def get_frame_info(frameidx):
    try:
        frame_number = int(frameidx)
        # Mở và đọc file output_samples.json
        with open("output_samples.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        
        frame_info = None
        for frame_data in data:
            if frame_data.get('frameidx') == frame_number:
                frame_info = frame_data
                break
        
        if frame_info:
            return jsonify(frame_info)
        else:
            return jsonify({"error": "Frame not found"}), 404
    except Exception as e:
        print(f"Error getting frame info: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/frames/<path:filename>')
def serve_frame(filename):
    try:
        if filename in FRAMES_MAPPING:
            full_path = FRAMES_MAPPING[filename]
            return send_file(full_path, mimetype="image/jpeg")
        else:
            # Kiểm tra xem filename có phải là đường dẫn đầy đủ không
            if os.path.exists(filename):
                return send_file(filename, mimetype="image/jpeg")
            
            # Thử tìm trong output_samples.json
            with open("output_samples.json", "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for frame_data in data:
                filepath = frame_data.get('filepath')
                if filepath and os.path.basename(filepath) == filename:
                    return send_file(filepath, mimetype="image/jpeg")
            
            abort(404, description=f"File {filename} not found.")
    except Exception as e:
        print(f"Error serving frame: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/reset')
def reset():
    all = sorted(load_frames_from_json(FRAMES_JSON), key=lambda x: int(Path(x).stem))
    tags = load_tags_from_file()
    return render_template("index.html", query=None, top_frames=all, tags=[t['tag'] for t in tags])

def load_tags_from_file():
    with open('tags.json', 'r',encoding="utf-8") as file:
        return json.load(file)
def search_frames_by_keyword(keyword, top_k):
    matching_frames = []
    print("keyword trong ham:", keyword)
    keyword_without_accents = unidecode(keyword.lower())
    print("keyword ko dau:",keyword_without_accents)
    for frame_data in frames_data:
        detections = frame_data.get("text_detections", {}).get("detections", [])
        for detection in detections:
            detection_label = detection.get("label", "")
            if not detection_label:
                continue 
            detection_label = detection_label.lower()
            label_without_accents = unidecode(detection_label)
            if keyword_without_accents in label_without_accents:
                matching_frames.append({
                    "frameid": frame_data.get("frameid", ""),
                    "confidence": detection.get("confidence", 0) 
                })
                break 
    matching_frames.sort(key=lambda x: x["confidence"], reverse=True)
    return [frame["frameid"] for frame in matching_frames[:top_k]]
def extract_query_confidence(frame_path, query):
    text_input = clip.tokenize([query]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_input).cpu().numpy()
    text_features = text_features / np.linalg.norm(text_features, axis=-1, keepdims=True)
    embeddings = np.load("E:\\Đồ án chuyên ngành\\19_12_2024\\embedding\\image_embeddings.npy")
    embeddings = embeddings / np.linalg.norm(embeddings, axis=-1, keepdims=True)
    all_files = load_frames_from_json(FRAMES_JSON)
    index = all_files.index(frame_path)
    similarity = np.dot(embeddings[index], text_features.T).flatten()[0]
    return similarity  
def load_frame_data(frame_id):
    with open(FRAMES_JSON, 'r',encoding="utf-8") as file:
        frames_data = json.load(file)
    for frame in frames_data:
        if frame.get('frameid') == frame_id:
            return frame 
    return {} 
def filter_frame_by_keyword_and_confidence(frame, keyword, keyword_frames, min_confidence):
    if frame not in keyword_frames:
        return False
    keyword_without_accents = unidecode(keyword.lower())
    frame_data = load_frame_data(frame)
    detections = frame_data.get("text_detections", {}).get("detections", [])
    for detection in detections:
        detection_label = detection.get("label", "").lower()
        # Kiểm tra nếu label chứa keyword và có confidence >= min_confidence
        if keyword_without_accents in unidecode(detection_label) and detection.get("confidence", 0) >= min_confidence:
            return True
    return False
@app.route('/')
def home():
    # Chuyển hướng người dùng đến giao diện mới
    return redirect('/video_search')

@app.route("/frame_search", methods=["GET", "POST"])
def frame_search():
    all = sorted(load_frames_from_json(FRAMES_JSON), key=lambda x: int(Path(x).stem))
    tags = load_tags_from_file()
    if request.method == "POST":
        query = request.form.get("query")
        top_k = request.form.get("top_k", 50)
        keyword = request.form.get("keyword") 
        min_confidence = float(request.form.get("min_confidence", 0.0))
        # min_confidence_keyword = float(request.form.get("min_confidence_keyword", 0.0))
        print("keyword:", keyword)
        print("query:", query)
        print("min_confidence:",min_confidence)
        try:
            top_k = int(float(top_k))
        except ValueError:
            top_k = 50
        total_frames = len(all)
        top_k = min(max(1, top_k), total_frames)
        top_frames_query = []
        top_frames_keyword = []
        if keyword:
            keyword_frames = search_frames_by_keyword(keyword, total_frames)
            print("danh sách keyword_frames:", keyword_frames)
            top_frames_keyword = [
                frame
                for frame in all
                if filter_frame_by_keyword_and_confidence(frame, keyword, keyword_frames, min_confidence)
            ][:top_k]
            print("Kết quả từ keyword:", top_frames_keyword)
        if query:
            processor = VietnameseTextProcessor()
            processed_text = processor.preprocess_and_translate(query)
            print("Câu truy vấn đã xử lý:", processed_text)
            query_frames = search_top_frames(processed_text, total_frames)
            top_frames_query = [
                frame
                for frame in query_frames
                #if extract_query_confidence(frame, processed_text) >= float(min_confidence/3)
            ][:top_k]
            print("Kết quả từ mô tả:", top_frames_query)
            # for frame in query_frames:
            #     print(extract_query_confidence(frame, processed_text))
        if query and keyword:
            combined_frames = list(set(top_frames_query + top_frames_keyword))[:top_k]
        elif query:
            combined_frames = top_frames_query
        elif keyword:
            combined_frames = top_frames_keyword
        else:
            combined_frames = all
        print("Kết hợp kết quả:", combined_frames)
        return render_template(
            "index.html",
            query=query,
            keyword=keyword,
            top_frames=combined_frames,
            tags=[t["tag"] for t in tags]
        )
    
    # Xử lý khi có GET request (lấy `tag_name` từ URL)
    tag_name = request.args.get("tag_name")
    if tag_name:
        # Lọc ảnh theo tag
        tag = next((t for t in tags if t["tag"] == tag_name), None)
        if tag:
            filtered_frames = tag["frames"]
        else:
            filtered_frames = []  
        return render_template("index.html", query=None,keyword=None, top_frames=filtered_frames, current_tag=tag_name, tags=[t['tag'] for t in tags])
    # Nếu không có tag_name, hiển thị tất cả các frame
    return render_template("index.html", query=None,keyword=None, top_frames=all, tags=[t['tag'] for t in tags])
@app.route("/search_image", methods=["POST"])
def search_image():
    tags = load_tags_from_file()
    top_k = int(request.form.get("top_k", 50))
    try:
        top_k = int(float(top_k))  
    except ValueError:
        top_k = 50  
    total_frames = len(load_frames_from_json(FRAMES_JSON))
    top_k = min(max(1, top_k), total_frames)  
    image_files = request.files.getlist("image_files") 
    image_url = request.form.get("image_url")
    image_features_list = []
    print(image_files)
    # # Process uploaded image files
    if (image_files != ""):
        for image_file in image_files:
            image = Image.open(image_file).convert("RGB")
            image_input = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image_input).cpu().numpy()
            image_features = image_features / np.linalg.norm(image_features, axis=-1, keepdims=True)
            image_features_list.append(image_features)

    if image_url:
        try:
            
            if re.match(r"^data:image\/[a-zA-Z]+;base64,", image_url):
                base64_str = image_url.split(",", 1)[1]  
                image_data = base64.b64decode(base64_str)
                image = Image.open(BytesIO(image_data)).convert("RGB")
            else:
               
                response = requests.get(image_url)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert("RGB")

            image_input = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image_input).cpu().numpy()
            image_features = image_features / np.linalg.norm(image_features, axis=-1, keepdims=True)
            image_features_list.append(image_features)
        except Exception as e:
            print(f"Error processing image from URL: {e}")

    
    if image_features_list:
        image_features_combined = np.mean(np.vstack(image_features_list), axis=0, keepdims=True)
        top_frames = search_top_frames_by_image(image_features_combined, top_k)
    else:
        top_frames = []

    return render_template("index.html", query=None, top_frames=top_frames,tags=[t['tag'] for t in tags])

def search_top_frames(query, top_k):
    
    
    text_input = clip.tokenize([query]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_input).cpu().numpy()

    text_features = text_features / np.linalg.norm(text_features, axis=-1, keepdims=True)

    embeddings = np.load("E:\\Đồ án chuyên ngành\\19_12_2024\\embedding\\image_embeddings.npy")
    embeddings = embeddings / np.linalg.norm(embeddings, axis=-1, keepdims=True)

    # similarities = np.dot(embeddings, text_features.T).flatten()
    # top_indices = np.argsort(similarities)[-top_k:][::-1] 
    # FAISS expects float32
    embeddings = embeddings.astype('float32')
    text_features = text_features.astype('float32')

    # Dùng chỉ mục FAISS cho cosine similarity (Inner Product)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    D, I = index.search(text_features, top_k)  # text_features shape: (1, dim)

    all_files = load_frames_from_json(FRAMES_JSON)
    top_indices = I[0]
    return [all_files[i] for i in top_indices]

def process_image_query(image_file, image_url):
    if image_file:
        image = Image.open(image_file).convert("RGB")
    elif image_url:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        return None

    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input).cpu().numpy()

    image_features = image_features / np.linalg.norm(image_features, axis=-1, keepdims=True)
    return image_features

def search_top_frames_by_image(image_features, top_k):
    embeddings = np.load(EMBEDDINGS_FILE)
    similarities = np.dot(embeddings, image_features.T).flatten()
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    all_files = load_frames_from_json(FRAMES_JSON)
    return [all_files[i] for i in top_indices]
# @app.route('/serve_video')
# def serve_video():
#     return send_file("E:\\Đồ án chuyên ngành\\Data\\video\\L01_V001.mp4", mimetype='video/mp4')
@app.route('/get_video_path/<frame>')
def get_video_path(frame):
    try:
        print("Type of frame:", type(frame))
        frame_number = int(frame)
        # Mở và đọc file out_samples.json
        with open("output_samples.json", "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"Looking for video for frame: {frame_number}")
        
        video_path = None
        for frame_data in data:
            if frame_data.get('frameidx') == frame_number: 
                video_path = frame_data.get('video')
                print("đường dẫn video: ", video_path)
                break
        
        if not video_path or not os.path.exists(video_path):
            return jsonify({"error": "Video not found"}), 404
        
        # Tạo đường dẫn web cho video
        video_filename = os.path.basename(video_path)
        web_video_path = url_for('serve_video', video_path=video_filename, _external=True)
        
        return jsonify({"video_path": web_video_path})

    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({"error": str(e)}), 500
@app.route("/save_tags", methods=["POST"])
def save_tags():
    data = request.json
    tag_name = data.get("tag", "").strip()
    frames = data.get("frames", [])

    if not tag_name:
        return jsonify({"error": "Tag name is required"}), 400
    if not frames:
        return jsonify({"error": "No frames provided"}), 400

    try:
        # Tải tags hiện có
        existing_tags = load_tags_from_file()

        # Kiểm tra nếu tag đã tồn tại
        for tag in existing_tags:
            if tag["tag"] == tag_name:
                tag["frames"] = frames  
                break
        else:
            # Thêm tag mới
            existing_tags.append({"tag": tag_name, "frames": frames})

        # Lưu lại vào file
        save_tags_to_file(existing_tags)

        # Trả về URL chứa tag vừa lưu
        return jsonify({"message": "Tags saved successfully", "redirect_url": url_for('frame_search', tag_name=tag_name)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/load-images', methods=['GET'])
def load_images():
    tag_name = request.args.get('tag_name')
    if tag_name:
        tags = load_tags_from_file()
        tag = next((t for t in tags if t['tag'] == tag_name), None)
        if tag:

            return jsonify({'images': tag['frames']})
    return jsonify({'images': []})

TAGS_FILE = "E:\\Đồ án chuyên ngành\\19_12_2024\\tags.json"
TAGS = load_tags_from_file()
@app.route('/get-tags', methods=['GET'])
def get_tags():
    with open('tags.json', 'r', encoding="utf-8") as file:
        data = json.load(file)
    return jsonify(data)

def save_tags_to_file(tags):
    with open(TAGS_FILE, "w") as f:
        json.dump(tags, f)

BASE_DIR = "E:\\Đồ án chuyên ngành\\19_12_2024\\static\\processed_frames"

@app.route("/upload-video", methods=["POST"])
def upload_video():
    """Endpoint upload video -> trích xuất frames -> xử lý JSON metadata."""
    video_file = request.files.get("video")
    print("video file: ",video_file)
    if not video_file:
        return jsonify({"error": "No video uploaded"}), 400
    video_name = os.path.splitext(video_file.filename)[0]
    print("video name:",video_name)
    frame_dir = os.path.join(BASE_DIR, video_name)
    frame_dir_process = r"{}".format(frame_dir)
    print("frame dir:",frame_dir)
    os.makedirs(frame_dir, exist_ok=True)
    BaseVideo_dir ="E:\\Đồ án chuyên ngành\\19_12_2024\\static\\video_frame"
    dir_save_video = os.path.join(BaseVideo_dir, video_name) 
    os.makedirs(dir_save_video, exist_ok=True)
    path_save_video = os.path.join(dir_save_video, video_file.filename)
    print("video path: ",path_save_video)
    video_file.save(path_save_video)

    def run_progress():
        total_steps = 3  
        step = 0  
        yield f"data:{{'step': 'extracting_frames', 'progress': {int((step / total_steps) * 100)}}}\n\n"
        time.sleep(0.5)
        
        try:
            extract_frames_from_video(path_save_video, frame_dir, threshold=30.0)
        except Exception as e:
            yield f"data:{{'error': 'Error extracting frames: {e}'}}\n\n"
            return
        step += 1
        yield f"data:{{'step': 'extracting_frames', 'progress': {int((step / total_steps) * 100)}}}\n\n"
        yield f"data:{{'step': 'extracting_embeddings', 'progress': {int((step / total_steps) * 100)}}}\n\n"
        time.sleep(0.5)
        try:
            model_name = "ViT-B/32"
            output_file = "E:/Đồ án chuyên ngành/9_12_2024/embedding/image_embeddings.npy"
            extract_and_save_embeddings_from_folder(frame_dir, model_name, output_file)
        except Exception as e:
            yield f"data:{{'error': 'Error extracting embeddings: {e}'}}\n\n"
            return
        step += 1
        yield f"data:{{'step': 'extracting_embeddings', 'progress': {int((step / total_steps) * 100)}}}\n\n"
        yield f"data:{{'step': 'processing_json', 'progress': {int((step / total_steps) * 100)}}}\n\n"
        time.sleep(0.5)
        try:
            json_output_path = "output_samples.json"
            process_images_in_folder(frame_dir_process, json_output_path,path_save_video)  # Xử lý metadata cho frames
        except Exception as e:
            yield f"data:{{'error': 'Error processing JSON: {e}'}}\n\n"
            return
        step += 1
        yield f"data:{{'step': 'processing_json', 'progress': {int((step / total_steps) * 100)}}}\n\n"
        yield f"data:{{'step': 'completed'}}\n\n"
    return Response(stream_with_context(run_progress()), mimetype="text/event-stream")

# def save_frames_permanently(temp_output_dir):
#     """Copy các frame từ thư mục tạm sang thư mục chính."""
#     permanent_dir = "static/processed_frames"
#     os.makedirs(permanent_dir, exist_ok=True)

#     for file in tqdm(os.listdir(temp_output_dir), desc="Saving Frames"):
#         if file.endswith(".jpg"):
#             src = os.path.join(temp_output_dir, file)
#             dst = os.path.join(permanent_dir, file)
#             shutil.copy(src, dst)

#     print(f"Frames have been saved to {permanent_dir}")

@app.route('/get_unique_videos')
def get_unique_videos():
    try:
        with open("output_samples.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Lấy danh sách video duy nhất
        videos = {}
        for frame_data in data:
            video_path = frame_data.get('video')
            if video_path and video_path not in videos:
                # Lấy tên video từ đường dẫn
                video_name = os.path.basename(os.path.dirname(video_path))
                video_filename = os.path.basename(video_path)
                # Tạo đường dẫn web cho video
                web_video_path = url_for('serve_video', video_path=video_filename)
                
                videos[video_path] = {
                    'name': video_name,
                    'path': web_video_path,
                    'original_path': video_path,
                    'thumbnail': frame_data.get('filepath')  # Sử dụng frame đầu tiên làm thumbnail
                }
        
        return jsonify(list(videos.values()))
    except Exception as e:
        print(f"Error getting unique videos: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/search_video_frames', methods=["POST"])
def search_video_frames():
    try:
        search_type = request.form.get("search_type")
        
        # Xử lý tìm kiếm bằng hình ảnh
        if search_type == "image":
            image_files = request.files.getlist("image_files")
            image_url = request.form.get("image_url")
            top_k = int(request.form.get("top_k", 3))
            
            # Xử lý hình ảnh và trích xuất đặc trưng
            image_features_list = []
            
            # Xử lý hình ảnh từ file tải lên
            if image_files and image_files[0].filename:
                for image_file in image_files:
                    try:
                        image = Image.open(image_file).convert("RGB")
                        image_input = preprocess(image).unsqueeze(0).to(device)
                        with torch.no_grad():
                            image_features = model.encode_image(image_input).cpu().numpy()
                        image_features = image_features / np.linalg.norm(image_features, axis=-1, keepdims=True)
                        image_features_list.append(image_features)
                    except Exception as e:
                        print(f"Error processing uploaded image: {e}")
            
            # Xử lý hình ảnh từ URL
            elif image_url:
                try:
                    if re.match(r"^data:image\/[a-zA-Z]+;base64,", image_url):
                        base64_str = image_url.split(",", 1)[1]
                        image_data = base64.b64decode(base64_str)
                        image = Image.open(BytesIO(image_data)).convert("RGB")
                    else:
                        response = requests.get(image_url)
                        response.raise_for_status()
                        image = Image.open(BytesIO(response.content)).convert("RGB")
                    
                    image_input = preprocess(image).unsqueeze(0).to(device)
                    with torch.no_grad():
                        image_features = model.encode_image(image_input).cpu().numpy()
                    image_features = image_features / np.linalg.norm(image_features, axis=-1, keepdims=True)
                    image_features_list.append(image_features)
                except Exception as e:
                    print(f"Error processing image from URL: {e}")
                    return jsonify({"error": f"Error processing image from URL: {e}"}), 400
            else:
                return jsonify({"error": "No image provided"}), 400
            
            if not image_features_list:
                return jsonify({"error": "Failed to process image"}), 400
            
            # Kết hợp các đặc trưng hình ảnh nếu có nhiều hình ảnh
            image_features_combined = np.mean(np.vstack(image_features_list), axis=0, keepdims=True)
            
            # Tìm kiếm các frame phù hợp
            top_frames = search_top_frames_by_image(image_features_combined, top_k * 10)  # Lấy nhiều frame hơn để lọc theo video
            
            # Đọc dữ liệu từ file JSON
            with open("output_samples.json", "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Nhóm các frame theo video
            video_frames = {}
            for frame_name in top_frames:
                frame_idx = int(Path(frame_name).stem)
                
                # Tìm thông tin frame trong dữ liệu
                frame_data = None
                for item in data:
                    if item.get('frameidx') == frame_idx:
                        frame_data = item
                        break
                
                if frame_data:
                    video_path = frame_data.get('video')
                    if video_path:
                        video_name = os.path.basename(os.path.dirname(video_path))
                        video_filename = os.path.basename(video_path)
                        # Tạo đường dẫn web cho video
                        web_video_path = url_for('serve_video', video_path=video_filename)
                        
                        if video_path not in video_frames:
                            video_frames[video_path] = {
                                'name': video_name,
                                'path': web_video_path,
                                'original_path': video_path,
                                'frames': []
                            }
                        
                        # Thêm frame vào danh sách của video
                        video_frames[video_path]['frames'].append({
                            'name': frame_name,
                            'path': frame_data.get('filepath'),
                            'frameidx': frame_idx,
                            'confidence': 0.0  # Không có confidence cho tìm kiếm hình ảnh
                        })
            
            # Lấy top 3 frame cho mỗi video
            result = []
            for video_path, video_info in video_frames.items():
                video_info['frames'] = video_info['frames'][:3]  # Lấy top 3 frame
                result.append(video_info)
            
            return jsonify(result)
        
        # Xử lý tìm kiếm bằng văn bản (code hiện tại)
        else:
            query = request.form.get("query")
            keyword = request.form.get("keyword")
            min_confidence = float(request.form.get("min_confidence", 0.0))
            
            if not query and not keyword:
                return jsonify({"error": "Query or keyword is required"}), 400
            
            # Đọc dữ liệu từ file JSON
            with open("output_samples.json", "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Lấy tất cả các frame
            all_frames = sorted(load_frames_from_json(FRAMES_JSON), key=lambda x: int(Path(x).stem))
            
            # Tìm kiếm theo query
            top_frames_query = []
            if query:
                processor = VietnameseTextProcessor()
                processed_text = processor.preprocess_and_translate(query)
                print("Câu truy vấn đã xử lý:", processed_text)
                query_frames = search_top_frames(processed_text, len(all_frames))
                top_frames_query = query_frames[:50]  # Lấy top 50 để sau đó lọc theo video
            
            # Tìm kiếm theo keyword
            top_frames_keyword = []
            if keyword:
                keyword_frames = search_frames_by_keyword(keyword, len(all_frames))
                top_frames_keyword = [
                    frame for frame in all_frames
                    if filter_frame_by_keyword_and_confidence(frame, keyword, keyword_frames, min_confidence)
                ][:50]  # Lấy top 50 để sau đó lọc theo video
            
            # Kết hợp kết quả
            if query and keyword:
                combined_frames = list(set(top_frames_query + top_frames_keyword))
            elif query:
                combined_frames = top_frames_query
            elif keyword:
                combined_frames = top_frames_keyword
            else:
                combined_frames = []
            
            # Nhóm các frame theo video và lấy top 3 frame cho mỗi video
            video_frames = {}
            for frame_name in combined_frames:
                frame_idx = int(Path(frame_name).stem)
                
                # Tìm thông tin frame trong dữ liệu
                frame_data = None
                for item in data:
                    if item.get('frameidx') == frame_idx:
                        frame_data = item
                        break
                
                if frame_data:
                    video_path = frame_data.get('video')
                    if video_path:
                        video_name = os.path.basename(os.path.dirname(video_path))
                        video_filename = os.path.basename(video_path)
                        # Tạo đường dẫn web cho video
                        web_video_path = url_for('serve_video', video_path=video_filename)
                        
                        if video_path not in video_frames:
                            video_frames[video_path] = {
                                'name': video_name,
                                'path': web_video_path,
                                'original_path': video_path,
                                'frames': []
                            }
                        
                        confidence = 0
                        if query:
                            confidence = extract_query_confidence(frame_name, processed_text)
                        
                        # Thêm frame vào danh sách của video
                        video_frames[video_path]['frames'].append({
                            'name': frame_name,
                            'path': frame_data.get('filepath'),
                            'frameidx': frame_idx,
                            'confidence': float(confidence)
                        })

            result = []
            for video_path, video_info in video_frames.items():
                video_info['frames'].sort(key=lambda x: x['confidence'], reverse=True)
                video_info['frames'] = video_info['frames'][:3] 
                result.append(video_info)
            
            return jsonify(result)
    except Exception as e:
        print(f"Error searching video frames: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/video_search')
def video_search():
    return render_template("video_search.html")

@app.route('/serve_video/<path:video_path>')
def serve_video(video_path):
    try:
        # Lấy đường dẫn video từ output_samples.json
        with open("output_samples.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Tìm video có đường dẫn trùng khớp
        video_path_full = None
        for frame_data in data:
            video = frame_data.get('video')
            if video and os.path.basename(video) == video_path:
                video_path_full = video
                break
        
        if video_path_full and os.path.exists(video_path_full):
            print(f"Serving video: {video_path_full}")
            return send_file(video_path_full, mimetype='video/mp4')
        else:
            print(f"Video not found: {video_path}")
            return jsonify({"error": "Video not found"}), 404
    except Exception as e:
        print(f"Error serving video: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000,debug=True)