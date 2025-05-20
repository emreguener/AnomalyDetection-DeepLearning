import os
from inference_uninet import run_uninet_inference
from inference_fastflow import run_fastflow_inference
from inference_efficient import run_inference_efficient

def run_voting_inference(input_path, img_id, output_dir,overlay_f,overlay_u,overlay_e):
    config_path = os.path.join("Model", "FastFlow", "config.yaml")
    ckpt_path = os.path.join("Model", "FastFlow", "best_model.pth")

    result_uninet = run_uninet_inference(input_path, img_id, output_dir,overlay_u)
    result_fastflow = run_fastflow_inference(input_path, img_id, config_path, ckpt_path, output_dir,overlay_f)
    result_efficientad = run_inference_efficient(input_path, img_id, output_dir,overlay_e)

    # Voting: 2 veya daha fazla Anomali varsa Anomali
    votes = [result_uninet["prediction"], result_fastflow["prediction"], result_efficientad["prediction"]]
    final_label = "Anomali" if votes.count("Anomali") >= 2 else "Normal"

    return {
        "score": None,  # Voting'e özel skor kullanılmayabilir
        "prediction": final_label,
        "original_url": result_uninet.get("original_url", ""),
        "preprocessed_url": result_uninet.get("preprocessed_url", ""),
        "overlay_url": result_uninet.get("overlay_url", ""),
        "heatmap_url": result_uninet.get("heatmap_url", ""),
        "details": {
            "UniNet": result_uninet["prediction"],
            "FastFlow": result_fastflow["prediction"],
            "EfficientAD": result_efficientad["prediction"]
        }
    }
