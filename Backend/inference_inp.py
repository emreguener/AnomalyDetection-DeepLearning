import os
from PIL import Image
from helpers_inp import (
    load_inpformer_model,
    preprocess_image,
    predict_inpformer,
    generate_heatmap
)

def run_inpformer_inference(image_path, image_id, config_path, ckpt_path, output_dir, threshold=0.5, overlay_threshold=0.5):
    device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

    model, config = load_inpformer_model(config_path, ckpt_path, device)

    image_tensor, original_image = preprocess_image(image_path, config["input_size"])
    raw_image = Image.open(image_path).convert("RGB")

    score, anomaly_map = predict_inpformer(model, image_tensor)
    label = "Anomali" if score > threshold else "Normal"

    overlay, heatmap = generate_heatmap(original_image, anomaly_map, overlay_threshold)

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
