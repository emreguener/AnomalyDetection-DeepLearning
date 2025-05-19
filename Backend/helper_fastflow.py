import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F
import FrEIA.framework as Ff
import FrEIA.modules as Fm


# === Yol Ayarları ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FASTFLOW_DIR = os.path.join(BASE_DIR, "Model", "FastFlow")
sys.path.append(FASTFLOW_DIR)  # FastFlow içindeki fastflow.py dosyasına erişim için

from fastflow import FastFlow
import yaml
import constants as const


def load_fastflow_model(config_path, ckpt_path, device):
    config = yaml.safe_load(open(config_path, "r"))
    model = FastFlow(
        backbone_name=config["backbone_name"],
        flow_steps=config["flow_step"],
        input_size=config["input_size"],
        conv3x3_only=config["conv3x3_only"],
        hidden_ratio=config["hidden_ratio"],
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.eval()
    print("✅ FastFlow modeli başarıyla yüklendi.")
    return model, config


def preprocess_image(image_path, input_size):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Görüntü bulunamadı: {image_path}")
    img = cv2.resize(img, (input_size, input_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    transform = T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(pil_img).unsqueeze(0)
    return tensor.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")), pil_img


def predict_fastflow(model, tensor):
    with torch.no_grad():
        output = model(tensor)
        score_map = output["anomaly_map"]
        score = score_map.view(score_map.shape[0], -1).max(dim=1)[0].item()
        score_map = score_map.squeeze().cpu().numpy()
    return score, score_map


def generate_heatmap(original_image, anomaly_map, threshold=None):
    # === Normalize anomaly map to [0, 1] ===
    norm_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)

    # === Create heatmap for visualization (as reference only) ===
    heatmap = (norm_map * 255).astype("uint8")
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # === Convert original image to OpenCV format
    overlay = np.array(original_image.convert("RGB"))  # shape: (H, W, 3)

    # === Optional: Draw contours on original image
    if threshold is not None:
        binary_mask = (norm_map > threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(overlay, contours, -1, (255, 0, 0), 2)  # RED on original

    return overlay, heatmap_color

def test_single_image_fastflow(image_path, config_path, ckpt_path, threshold=-0.3205):
    import torch
    import cv2
    from helper_fastflow import (
        load_fastflow_model,
        preprocess_image,
        predict_fastflow,
        generate_heatmap
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_fastflow_model(config_path, ckpt_path, device)
    tensor, original_image = preprocess_image(image_path, config["input_size"])
    score, anomaly_map = predict_fastflow(model, tensor)

    prediction = "Anomali" if score > threshold else "Normal"
    print(f"📊 Skor: {score:.4f} → Tahmin: {prediction}")

    # generate_heatmap uses normalized anomaly_map, threshold should also be in [0, 1] now
    overlay, heatmap = generate_heatmap(original_image, anomaly_map, threshold)

    # Ensure saved image sizes match input image
    overlay_resized = cv2.resize(overlay, original_image.size[::-1])
    heatmap_resized = cv2.resize(heatmap, original_image.size[::-1])

    cv2.imwrite("fastflow_overlay.png", cv2.cvtColor(overlay_resized, cv2.COLOR_RGB2BGR))
    cv2.imwrite("fastflow_heatmap.png", cv2.cvtColor(heatmap_resized, cv2.COLOR_RGB2BGR))
    return score, prediction
