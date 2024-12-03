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

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# FRAMES_DIR = "E:\\THIHE\\testfitty one\\SegmentVideo\\seg1\\SegmentVideo"
FRAMES_JSON = "E:\\Đồ án chuyên ngành\\resource\\DACN_modelCLIP\\output_samples.json" 
EMBEDDINGS_FILE = "E:\\Đồ án chuyên ngành\\resource\\DACN_modelCLIP\\embedding\\image_embeddings.npy"
text_processor = VietnameseTextProcessor()

embeddings = np.load(EMBEDDINGS_FILE)

def load_frames_from_json(json_path):
    """Load danh sách tên file từ file JSON."""
    with open(json_path, 'r', encoding='utf-8') as file:
        samples = json.load(file)
    # Trả về danh sách tên file (không bao gồm đường dẫn đầy đủ)
    return [os.path.basename(sample["filepath"]) for sample in samples if "filepath" in sample]
def load_frames_mapping_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        # Trả về mapping {filename: full_filepath}
        return {Path(sample["filepath"]).name: sample["filepath"] for sample in data}

FRAMES_MAPPING = load_frames_mapping_from_json(FRAMES_JSON)
app = Flask(__name__)

# @app.route('/frames/<path:filename>')
# def serve_frame(filename):
#     a = send_from_directory(FRAMES_DIR, filename)
#     print(a)
#     return send_from_directory(FRAMES_DIR, filename)
def load_frame_data():
    with open('DACN_modelCLIP/output_samples.json', 'r') as file:
        return json.load(file)

# Lưu dữ liệu frames vào một biến toàn cục
frames_data = load_frame_data()

@app.route('/get_frame_info/<frameidx>')
def get_frame_info(frameidx):
    # Tìm frame tương ứng theo frameidx
    frame_info = next((frame for frame in frames_data if str(frame['frameidx']) == frameidx), None)

    if frame_info:
        return jsonify(frame_info)
    else:
        return jsonify({"error": "Frame not found"}), 404

@app.route('/frames/<path:filename>')
def serve_frame(filename):
    if filename in FRAMES_MAPPING:
        full_path = FRAMES_MAPPING[filename]
        return send_file(full_path, mimetype="image/jpeg")  
    else:
        abort(404, description=f"File {filename} not found.")

@app.route('/reset')
def reset():
    all = sorted(load_frames_from_json(FRAMES_JSON), key=lambda x: int(Path(x).stem))
    tags = load_tags_from_file()
    return render_template("index.html", query=None, top_frames=all, tags=[t['tag'] for t in tags])

def load_tags_from_file():
    with open('DACN_modelCLIP/tags.json', 'r') as file:
        return json.load(file)

@app.route("/", methods=["GET", "POST"])
def index():
    all = sorted(load_frames_from_json(FRAMES_JSON), key=lambda x: int(Path(x).stem))
    tags = load_tags_from_file()

    if request.method == "POST":
        query = request.form.get("query")
        top_k = request.form.get("top_k", 50)

        try:
            top_k = int(float(top_k))
        except ValueError:
            top_k = 50

        total_frames = len(all)    
        top_k = min(max(1, top_k), total_frames)
        processed_text = text_processor.preprocess_and_translate(query)
        print("Câu truy vấn đã xử lý:", processed_text)
        top_frames = search_top_frames(processed_text, top_k)
        return render_template("index.html", query=query, top_frames=top_frames,tags=[t['tag'] for t in tags])
    
    # Xử lý khi có GET request (lấy `tag_name` từ URL)
    tag_name = request.args.get("tag_name")
    if tag_name:
        # Lọc ảnh theo tag
        tag = next((t for t in tags if t["tag"] == tag_name), None)
        if tag:
            filtered_frames = tag["frames"]
        else:
            filtered_frames = []  
        return render_template("index.html", query=None, top_frames=filtered_frames, current_tag=tag_name, tags=[t['tag'] for t in tags])
    # Nếu không có tag_name, hiển thị tất cả các frame
    return render_template("index.html", query=None, top_frames=all, tags=[t['tag'] for t in tags])
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

    embeddings = np.load("E:\\Đồ án chuyên ngành\\resource\\DACN_modelCLIP\\embedding\\image_embeddings.npy")
    embeddings = embeddings / np.linalg.norm(embeddings, axis=-1, keepdims=True)

    similarities = np.dot(embeddings, text_features.T).flatten()

    top_indices = np.argsort(similarities)[-top_k:][::-1] 

    all_files = load_frames_from_json(FRAMES_JSON)
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
    similarities = np.dot(embeddings, image_features.T).flatten()
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    all_files = load_frames_from_json(FRAMES_JSON)
    return [all_files[i] for i in top_indices]
@app.route('/video_popup')
def video_popup():
    frameidx = request.args.get('frameidx', default='0')
    return render_template('video_popup.html', frameidx=frameidx)

@app.route('/serve_video')
def serve_video():
    return send_file("E:\\THIHE\\testfitty one\\videotesst.mp4", mimetype='video/mp4')

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
        return jsonify({"message": "Tags saved successfully", "redirect_url": url_for('index', tag_name=tag_name)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/load-images', methods=['GET'])
def load_images():
    tag_name = request.args.get('tag_name')
    if tag_name:
        tags = load_tags_from_file()
        tag = next((t for t in tags if t['tag'] == tag_name), None)
        if tag:
            # Trả về danh sách tên ảnh theo tag
            return jsonify({'images': tag['frames']})
    return jsonify({'images': []})

TAGS_FILE = "E:\\Đồ án chuyên ngành\\resource\\DACN_modelCLIP\\tags.json"
# Lưu trữ tags toàn cục để sử dụng trong các route
TAGS = load_tags_from_file()
@app.route('/get-tags', methods=['GET'])
def get_tags():
    with open('DACN_modelCLIP/tags.json', 'r') as file:
        data = json.load(file)
    return jsonify(data)

def save_tags_to_file(tags):
    with open(TAGS_FILE, "w") as f:
        json.dump(tags, f)

if __name__ == "__main__":
    app.run(debug=True)
