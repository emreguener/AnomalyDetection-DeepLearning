from flask import Flask, request, render_template, send_file
from flask_cors import CORS
import os
import uuid

from inference_efficient import run_inference_efficient
from inference_uninet import run_uninet_inference
from inference_fastflow import run_fastflow_inference
from inference_pbas import run_pbas_inference
from inference_voting import run_voting_inference
from inference_inp import run_inpformer_inference


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

def get_overlay_threshold(model, level):
    mappings = {
        "uninet": {"low": 2.50, "medium": 2.35, "high": 2.0},
        "fastflow": {"low": -0.70, "medium": -0.55, "high": -0.30},
        "efficientad": {"low": 0.40, "medium": 0.25, "high": 0.18},
        "pbas": {"low": 0.65, "medium": 0.75, "high": 0.35},
        "inp-former": {"low": 0.65, "medium": 0.5840, "high": 0.33},
    }
    return mappings.get(model, {}).get(level, None)



@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return "No image uploaded", 400

    image = request.files["image"]
    model_choice = request.form.get("model_choice", "efficientad")
    threshold_level = request.form.get("threshold", "medium")

    img_id = get_next_available_img_id(UPLOAD_FOLDER)
    input_path = os.path.join(UPLOAD_FOLDER, f"{img_id}.png")
    image.save(input_path)

    # Önceki çıktıları temizle
    for suffix in ["_a.png", "_b.png"]:
        output_file = os.path.join(OUTPUT_FOLDER, f"{img_id}{suffix}")
        if os.path.exists(output_file):
            os.remove(output_file)

    result = None

    if model_choice == "uninet":
        overlay_threshold = get_overlay_threshold("uninet", threshold_level)
        result = run_uninet_inference(input_path, img_id, OUTPUT_FOLDER, overlay_threshold)

    elif model_choice == "fastflow":
        config_path = os.path.join("Model", "FastFlow", "config.yaml")
        ckpt_path = os.path.join("Model", "FastFlow", "best_model.pth")
        overlay_threshold = get_overlay_threshold("fastflow", threshold_level)
        result = run_fastflow_inference(input_path, img_id, config_path, ckpt_path, OUTPUT_FOLDER, overlay_threshold)

    elif model_choice == "efficientad":
        overlay_threshold = get_overlay_threshold("efficientad", threshold_level)
        result = run_inference_efficient(input_path, img_id, OUTPUT_FOLDER, overlay_threshold)

    elif model_choice == "pbas":
        overlay_threshold = get_overlay_threshold("pbas", threshold_level)
        ckpt_path = os.path.join("Model", "PBAS", "model.pth")  # varsa yolunu düzelt
        result = run_pbas_inference(input_path, img_id, ckpt_path, OUTPUT_FOLDER, overlay_threshold)

    elif model_choice == "inp-former":
        config_path = os.path.join("Model", "INPFormer", "config.yaml")
        ckpt_path = os.path.join("Model", "INPFormer", "model.pth")
        overlay_threshold = get_overlay_threshold("inpformer", threshold_level)
        result = run_inpformer_inference(input_path, img_id, config_path, ckpt_path, OUTPUT_FOLDER, overlay_threshold)

    elif model_choice == "voting":
        overlay_threshold_u = get_overlay_threshold("uninet", threshold_level)
        overlay_threshold_f = get_overlay_threshold("fastflow", threshold_level)
        overlay_threshold_e = get_overlay_threshold("efficientad", threshold_level)
        result = run_voting_inference(input_path, img_id, OUTPUT_FOLDER,overlay_threshold_f,overlay_threshold_u,overlay_threshold_e)

    return render_template("index.html", result=result)

@app.route("/image/<filename>")
def get_image(filename):
    path = os.path.join(OUTPUT_FOLDER, filename)
    if not os.path.exists(path):
        return f"Image {filename} not found", 404
    return send_file(path, mimetype='image/png')


@app.route("/aboutus.html")  # ← Bu şekilde olursa .html URL çalışır
def aboutus():
    return render_template("aboutus.html")

if __name__ == "__main__":
    app.run(debug=True, port=5000)
