
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU
 
from collections import Counter
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import io
from PIL import Image
 
app = Flask(__name__)
CORS(app, origins="*")

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "🚀 Falcon Technology API is running!",
        "endpoints": {
            "detect": "/detect (POST)",
            "summary": "/detect_summary (POST)"
        }
    })
 
# ── Load your local model ──
model = YOLO("Eagle-1(Chocolate)\my_model.pt")
 
 
# ── Route 1: Returns annotated image ──
@app.route('/detect', methods=['POST', 'OPTIONS'])
def detect():
    if request.method == 'OPTIONS':
        return '', 200
 
    file = request.files['image']
    img_bytes = file.read()
 
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
 
    results = model(frame)
    annotated_frame = results[0].plot()
 
    annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(annotated_rgb)
 
    img_io = io.BytesIO()
    pil_img.save(img_io, 'JPEG')
    img_io.seek(0)
 
    return send_file(img_io, mimetype='image/jpeg')
 
 
# ── Route 2: Returns JSON summary (count per class) ──
@app.route('/detect_summary', methods=['POST', 'OPTIONS'])
def detect_summary():
    if request.method == 'OPTIONS':
        return '', 200
 
    file = request.files['image']
    img_bytes = file.read()
 
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
 
    results = model(frame)
 
    names = model.names
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int).tolist()
    counts = Counter(names[i] for i in class_ids)
 
    return jsonify({
        "total": len(class_ids),
        "labels": dict(counts)
    })
 
 
if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')