from flask import Flask, request, render_template, send_file
from flask_cors import CORS
import os
import uuid

from inference import run_inference_on_image
from inference_uninet import run_uninet_inference
from inference_fastflow import run_fastflow_inference
from inference_voting import run_voting_inference

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "static/outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def get_next_available_img_id(folder):
    existing = {f.split(".")[0] for f in os.listdir(folder) if f.endswith(".png")}
    i = 1
    while str(i) in existing:
        i += 1
    return str(i)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return "No image uploaded", 400

    image = request.files["image"]
    model_choice = request.form.get("model_choice", "efficientad")
    
    # 📌 Boşta olan ID'yi bul
    img_id = get_next_available_img_id(UPLOAD_FOLDER)
    
    input_path = os.path.join(UPLOAD_FOLDER, f"{img_id}.png")
    image.save(input_path)

    # Eski çıktıları temizle (gerekirse)
    for suffix in ["_a.png", "_b.png"]:
        try:
            os.remove(os.path.join(OUTPUT_FOLDER, f"{img_id}{suffix}"))
        except FileNotFoundError:
            pass

    # 🔍 Model çalıştır
    if model_choice == "uninet":
        result = run_uninet_inference(input_path, img_id, OUTPUT_FOLDER)
    elif model_choice == "fastflow":
        config_path = os.path.join("Model", "FastFlow", "config.yaml")
        ckpt_path = os.path.join("Model", "FastFlow", "best_model.pth")
        result = run_fastflow_inference(input_path, img_id, config_path, ckpt_path, OUTPUT_FOLDER)
    elif model_choice == "voting":
        result = run_voting_inference(input_path, img_id, OUTPUT_FOLDER)
    else:
        result = run_inference_on_image(input_path, img_id, OUTPUT_FOLDER)

    return render_template("index.html", result=result)


@app.route("/image/<filename>")
def get_image(filename):
    path = os.path.join(OUTPUT_FOLDER, filename)
    if not os.path.exists(path):
        return f"Image {filename} not found", 404
    return send_file(path, mimetype='image/png')

if __name__ == "__main__":
    app.run(debug=True, port=5000)
