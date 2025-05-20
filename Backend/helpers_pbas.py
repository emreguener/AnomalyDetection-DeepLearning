import torch
import os
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import torchvision.transforms as T
import torch.nn.functional as F
import torchvision.models as models
import torch.nn as nn
import sys
from skimage import measure

# === PATH AYARI ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PBAS_DIR = os.path.join(BASE_DIR, "Model", "PBAS")
sys.path.append(PBAS_DIR)

from pbas import PBAS
import metrics

# === Cihaz seçimi ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_pbas_model(ckpt_path, device):
    import torchvision.models as models

    # PBAS modelini kur
    model = PBAS(device=device, patchsize=3, patchstride=1, pre_proj=1)

    # Backbone olarak ResNet50'in sadece conv katmanları
    resnet = models.resnet50(pretrained=True)
    model.backbone = nn.Sequential(*list(resnet.children())[:-2])  # [B, 2048, 8, 8]

    # Patch boyutundan gelen input dim
    patch_dim = 2048 * 3 * 3
    proj_dim = 256

    model.proj_layer = nn.Linear(patch_dim, proj_dim)
    model.decoder = nn.Linear(proj_dim, patch_dim)

    # Ağırlıkları yükle (varsa)
    if os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()
    return model


# === Görsel oku ve hazırla ===
def preprocess_input_image_pbas(image_path, size=256, output_dir=None):
    """
    Görüntüyü oku, Gaussian + Otsu + maskeleme uygula, normalize et ve PIL Image olarak döndür.
    output_dir verilirse işlemli görseli diske de kaydeder.
    """
    # 1. Görüntüyü oku
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Görüntü bulunamadı: {image_path}")

    # 2. Griye çevir (Otsu için)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Gaussian blur ve Otsu eşikleme
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. Maskeleme (renkli resme uygula)
    masked = cv2.bitwise_and(img, img, mask=thresh)

    # 5. Resize
    resized = cv2.resize(masked, (size, size), interpolation=cv2.INTER_AREA)

    # 6. Normalize et ve tekrar 0-255'e getir (isteğe bağlı)
    normalized = resized.astype(np.float32) / 255.0
    output_img = (normalized * 255).astype(np.uint8)

    # 7. Kaydetmek istersen
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, output_img)
        print(f"✅ Kaydedildi: {output_path}")

    # 8. RGB formatta PIL Image olarak döndür
    return Image.fromarray(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))


# === Görsel tensor'a çevir ===
def transform_image(image):
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image).unsqueeze(0).to(device)


# === Tahmin (inference) ===
def predict_pbas(model, tensor):
    with torch.no_grad():
        output = model(tensor)
        print("📐 Output shape:", output.shape)

        if output.ndim == 4:
            output = output.squeeze(0).squeeze(0)
        elif output.ndim == 3:
            output = output.squeeze(0)
        elif output.ndim != 2:
            raise ValueError(f"Beklenmeyen çıktı boyutu: {output.shape}")

        amap = output.cpu().numpy()
        amap = gaussian_filter(amap, sigma=4)
        amap = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
        score = np.max(amap)
        return score, amap


# === Overlay + heatmap üret ===
def create_overlay_pbas(original_image, anomaly_map, threshold=0.40):
    orig_np = np.array(original_image.convert('RGB'))
    anomaly_map = cv2.resize(anomaly_map, (orig_np.shape[1], orig_np.shape[0]))
    anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)

    heatmap = cv2.normalize(anomaly_map, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    binary_mask = (anomaly_map > threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    overlay = orig_np.copy()
    if contours:
        cv2.drawContours(overlay, contours, -1, (255, 0, 0), 2)

    return overlay, heatmap_color


# === Test fonksiyonu ===
def test_single_image_pbas(model, image_path, threshold=0.4):
    image = preprocess_input_image_pbas(image_path)
    tensor = transform_image(image)
    score, amap = predict_pbas(model, tensor)
    label = "Anomali" if score > threshold else "Normal"
    print(f"📊 Skor: {score:.4f} → Tahmin: {label}")
    overlay, heatmap = create_overlay_pbas(image, amap, threshold)

    # Kaydet
    cv2.imwrite("pbas_overlay.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite("pbas_heatmap.png", cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
    return score, label
