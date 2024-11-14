from flask import Flask, request, render_template, send_from_directory, Response,send_file
import os
import numpy as np
import clip
import torch
from PIL import Image
from tqdm import tqdm
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
        try:
            top_k = int(float(top_k))  
        except ValueError:
            top_k = 50  

        total_frames = len([f for f in os.listdir(FRAMES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        top_k = min(max(1, top_k), total_frames)  
        
        processed_text = text_processor.preprocess_and_translate(query)
        print("Cau truy van da xu ly:", processed_text)
        top_frames = search_top_frames(processed_text, top_k)
        
        return render_template("index.html", query=query, top_frames=top_frames)
    return render_template("index.html", query=None, top_frames=[])

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

@app.route('/video_popup')
def video_popup():
    frameidx = request.args.get('frameidx', default='0')
    return render_template('video_popup.html', frameidx=frameidx)

@app.route('/video')
def video():
    return render_template('video_popup.html')

@app.route('/serve_video')
def serve_video():

    return send_file("E:\\THIHE\\testfitty one\\videotesst.mp4", mimetype='video/mp4')
if __name__ == "__main__":
    app.run(debug=True)
