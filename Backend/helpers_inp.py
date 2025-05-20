import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as T
import yaml


# === Yol Ayarları ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPFORMER_DIR = os.path.join(BASE_DIR, "Model", "INPFormer")
sys.path.append(INPFORMER_DIR)

from inpformer import INP_Former as INPFormer 

def load_inpformer_model(config_path, ckpt_path, device):
    # Gerekli bileşenleri oluştur
    encoder = ViTEncoder(name="dinov2_vitl14_reg4", pretrained=True)
    bottleneck = Bottleneck(in_channels=1024, out_channels=256)
    aggregation = AggregationModule()  # ya da parametreli
    decoder = SimpleDecoder(in_channels=256, out_channels=1)

    # Modeli oluştur
    model = INP_Former(
        encoder=encoder,
        bottleneck=bottleneck,
        aggregation=aggregation,
        decoder=decoder
    ).to(device)

    # Ağırlıkları yükle
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.eval()

    print("✅ INPFormer modeli başarıyla yüklendi.")
    return model, {"input_size": 256}  # sadece preprocess için


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


def predict_inpformer(model, tensor):
    with torch.no_grad():
        output = model(tensor)
        # Aşağıdaki yapı modelin çıkışına göre özelleştirilmeli
        score_map = output["anomaly_map"] if isinstance(output, dict) else output
        score = score_map.view(score_map.shape[0], -1).max(dim=1)[0].item()
        score_map = score_map.squeeze().cpu().numpy()
    return score, score_map


def generate_heatmap(original_image, anomaly_map, threshold=None):
    norm_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)

    heatmap = (norm_map * 255).astype("uint8")
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = np.array(original_image.convert("RGB"))

    if threshold is not None:
        binary_mask = (norm_map > threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(overlay, contours, -1, (255, 0, 0), 2)

    return overlay, heatmap_color
