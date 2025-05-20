from helpers_pbas import (
    load_pbas_model,
    preprocess_input_image_pbas,
    transform_image,
    predict_pbas,
    create_overlay_pbas
)
from PIL import Image
import os
import torch

model_pbas = None

def run_pbas_inference(image_path, image_id, ckpt_path, output_dir, threshold=0.5):
    global model_pbas
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_pbas is None:
        model_pbas = load_pbas_model(ckpt_path, device)

    preprocessed_image = preprocess_input_image_pbas(image_path)
    image_tensor = transform_image(preprocessed_image)
    raw_image = Image.open(image_path).convert("RGB")

    score, anomaly_map = predict_pbas(model_pbas, image_tensor)
    label = "Anomali" if score > threshold else "Normal"
    overlay, heatmap = create_overlay_pbas(preprocessed_image, anomaly_map, threshold)

    os.makedirs(output_dir, exist_ok=True)
    raw_image.save(os.path.join(output_dir, f"{image_id}_original.png"))
    preprocessed_image.save(os.path.join(output_dir, f"{image_id}_preprocessed.png"))
    Image.fromarray(overlay).save(os.path.join(output_dir, f"{image_id}_overlay.png"))
    Image.fromarray(heatmap).save(os.path.join(output_dir, f"{image_id}_heatmap.png"))

    return {
        "score": round(float(score), 4),
        "prediction": label,
        "original_url": f"/image/{image_id}_original.png",
        "preprocessed_url": f"/image/{image_id}_preprocessed.png",
        "overlay_url": f"/image/{image_id}_overlay.png",
        "heatmap_url": f"/image/{image_id}_heatmap.png"
    }
