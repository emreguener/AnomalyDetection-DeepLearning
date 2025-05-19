# inference.py
# Author: Baturhan Çağatay
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image
from helpers import transform_image, load_models, predict, create_overlay, preprocess_input_image
import os
import numpy as np

teacher, student, autoencoder, quantiles, stats = load_models()


def run_inference_on_image(img_path, img_id, output_dir):
    # Preprocess: Maskeleme, threshold, normalize (model ile tutarlı)
    preprocessed_pil_image = preprocess_input_image(img_path)

    # Transform: Tensor + normalize
    image_tensor, processed_for_tensor = transform_image(preprocessed_pil_image)

    # Ham orijinal resmi RGB olarak al
    raw_image = Image.open(img_path).convert("RGB")

    # Tahmin yap
    anomaly_map, score, label = predict(image_tensor, teacher, student, autoencoder, quantiles, stats)

    # Overlay ve heatmap oluştur
    overlay, heatmap = create_overlay(processed_for_tensor, anomaly_map)

    # Klasör oluşturulmamışsa yarat
    os.makedirs(output_dir, exist_ok=True)

    # Ham orijinal görüntü
    orig_save_path = os.path.join(output_dir, f"{img_id}_original.png")
    raw_image.save(orig_save_path)

    # Otsu sonrası işlenmiş görüntü
    otsu_save_path = os.path.join(output_dir, f"{img_id}_preprocessed.png")
    preprocessed_pil_image.save(otsu_save_path)

    # Overlay kaydet
    overlay_path = os.path.join(output_dir, f"{img_id}_overlay.png")
    Image.fromarray(overlay).save(overlay_path)

    # Heatmap kaydet
    heatmap_path = os.path.join(output_dir, f"{img_id}_heatmap.png")
    Image.fromarray(heatmap).save(heatmap_path)

    # Ground truth (placeholder siyah görsel)
    gt_save_path = os.path.join(output_dir, f"{img_id}_ground_truth.png")
    dummy_gt = Image.new("RGB", preprocessed_pil_image.size, (0, 0, 0))
    dummy_gt.save(gt_save_path)

    return {
        "score": round(score, 4),
        "prediction": label,
        "original_url": f"/image/{img_id}_original.png",
        "preprocessed_url": f"/image/{img_id}_preprocessed.png",
        "overlay_url": f"/image/{img_id}_overlay.png",
        "heatmap_url": f"/image/{img_id}_heatmap.png",
        
    }