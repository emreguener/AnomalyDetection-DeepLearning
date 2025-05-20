import sys
import os
from PIL import Image
from helpers_uninet import (
    load_model_from_ckpt, preprocess_input_image, transform_image,
    predict, create_overlay, c, device
)

# Backend klasörünü import path'e ekle
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(BASE_DIR, "Model")
sys.path.append(BACKEND_DIR)

# Modeli yükle
model = load_model_from_ckpt(c, device)


def run_uninet_inference(img_path, img_id, output_dir,overlay=2.35):
    # Görüntüleri hazırla
    preprocessed_image = preprocess_input_image(img_path)
    image_tensor, _ = transform_image(preprocessed_image, c)
    raw_image = Image.open(img_path).convert("RGB")

    # Tahmin yap
    score, anomaly_map = predict(model, image_tensor, c)
    label = "Anomali" if score > 2 else "Normal"

    # Görsel çıktıları oluştur
    overlay, heatmap = create_overlay(preprocessed_image, anomaly_map,overlay)

    # Kayıt dizinini oluştur
    os.makedirs(output_dir, exist_ok=True)
    raw_image.save(os.path.join(output_dir, f"{img_id}_original.png"))
    preprocessed_image.save(os.path.join(output_dir, f"{img_id}_preprocessed.png"))
    Image.fromarray(overlay).save(os.path.join(output_dir, f"{img_id}_overlay.png"))
    Image.fromarray(heatmap).save(os.path.join(output_dir, f"{img_id}_heatmap.png"))

    return {
        "score": round(float(score), 4),
        "prediction": label,
        "original_url": f"/image/{img_id}_original.png",
        "preprocessed_url": f"/image/{img_id}_preprocessed.png",
        "overlay_url": f"/image/{img_id}_overlay.png",
        "heatmap_url": f"/image/{img_id}_heatmap.png"
    }
