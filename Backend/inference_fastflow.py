import os
from PIL import Image
from helper_fastflow import (
    load_fastflow_model,
    preprocess_image,
    predict_fastflow,
    generate_heatmap
)


def run_fastflow_inference(image_path, image_id, config_path, ckpt_path, output_dir, threshold=-0.3205, overlay_threshold=0.55):
    device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

    # Model ve konfigürasyonu yükle
    model, config = load_fastflow_model(config_path, ckpt_path, device)

    # Görüntüyü hazırla
    image_tensor, original_image = preprocess_image(image_path, config["input_size"])
    raw_image = Image.open(image_path).convert("RGB")

    # Tahmin yap
    score, anomaly_map = predict_fastflow(model, image_tensor)
    label = "Anomali" if score > threshold else "Normal"

    # Görsel çıktıları oluştur
    overlay, heatmap = generate_heatmap(original_image, anomaly_map, overlay_threshold)

    # Kayıt dizinini oluştur
    os.makedirs(output_dir, exist_ok=True)
    raw_image.save(os.path.join(output_dir, f"{image_id}_original.png"))
    original_image.save(os.path.join(output_dir, f"{image_id}_preprocessed.png"))
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
