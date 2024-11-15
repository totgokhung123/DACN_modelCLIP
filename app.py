from flask import Flask, request, render_template, send_from_directory, Response,send_file,url_for
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
from word_processing import VietnameseTextProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

FRAMES_DIR = "E:\\THIHE\\testfitty one\\SegmentVideo\\seg1\\SegmentVideo"
EMBEDDINGS_FILE = "E:\\Đồ án chuyên ngành\\resource\\DACN_modelCLIP\\embedding\\image_embeddings.npy"
text_processor = VietnameseTextProcessor()

embeddings = np.load(EMBEDDINGS_FILE)

app = Flask(__name__)

@app.route('/frames/<path:filename>')
def serve_frame(filename):
    return send_from_directory(FRAMES_DIR, filename)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        query = request.form.get("query")
        top_k = request.form.get("top_k", 50)  
        # image_file = request.files.get("image_file")
        # image_url = request.form.get("image_url")
        try:
            top_k = int(float(top_k))  
        except ValueError:
            top_k = 50  

        total_frames = len([f for f in os.listdir(FRAMES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        top_k = min(max(1, top_k), total_frames)  
        # if image_file or image_url:
        #     image_features = process_image_query(image_file, image_url)
        #     top_frames = search_top_frames_by_image(image_features, top_k)
        # else:
        processed_text = text_processor.preprocess_and_translate(query)
        print("Cau truy van da xu ly:", processed_text)
        top_frames = search_top_frames(processed_text, top_k)
       
        
        return render_template("index.html", query=query, top_frames=top_frames)
    return render_template("index.html", query=None, top_frames=[])
@app.route("/search_image", methods=["POST"])
def search_image():
    top_k = int(request.form.get("top_k", 50))
    try:
        top_k = int(float(top_k))  
    except ValueError:
        top_k = 50  
    total_frames = len([f for f in os.listdir(FRAMES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    top_k = min(max(1, top_k), total_frames)  
    image_files = request.files.getlist("image_files")  # Retrieve multiple files
    image_url = request.form.get("image_url")
    print(image_url)
    image_features_list = []

    # # Process uploaded image files
    # if image_files:
    #     for image_file in image_files:
    #         image = Image.open(image_file).convert("RGB")
    #         image_input = preprocess(image).unsqueeze(0).to(device)
    #         with torch.no_grad():
    #             image_features = model.encode_image(image_input).cpu().numpy()
    #         image_features = image_features / np.linalg.norm(image_features, axis=-1, keepdims=True)
    #         image_features_list.append(image_features)

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

    return render_template("index.html", query=None, top_frames=top_frames)
#def search_top_frames(query, top_k):
    
    text_input = clip.tokenize([query]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_input).cpu().numpy()
    
    similarities = np.dot(embeddings, text_features.T).flatten()
    top_indices = np.argsort(similarities)[-top_k:][::-1]  

    all_files = [f for f in os.listdir(FRAMES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    return [all_files[i] for i in top_indices]

def search_top_frames(query, top_k):
    
    text_input = clip.tokenize([query]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_input).cpu().numpy()

    text_features = text_features / np.linalg.norm(text_features, axis=-1, keepdims=True)

    embeddings = np.load("E:\\Đồ án chuyên ngành\\resource\\DACN_modelCLIP\\embedding\\image_embeddings.npy")
    embeddings = embeddings / np.linalg.norm(embeddings, axis=-1, keepdims=True)

    similarities = np.dot(embeddings, text_features.T).flatten()

    top_indices = np.argsort(similarities)[-top_k:][::-1] 

    all_files = [f for f in os.listdir(FRAMES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
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

    all_files = [f for f in os.listdir(FRAMES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    return [all_files[i] for i in top_indices]
@app.route('/video_popup')
def video_popup():
    frameidx = request.args.get('frameidx', default='0')
    return render_template('video_popup.html', frameidx=frameidx)

@app.route('/serve_video')
def serve_video():
    return send_file("E:\\THIHE\\testfitty one\\videotesst.mp4", mimetype='video/mp4')
if __name__ == "__main__":
    app.run(debug=True)
