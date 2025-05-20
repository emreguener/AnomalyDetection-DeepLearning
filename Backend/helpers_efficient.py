import os
import torch
from torch.serialization import add_safe_globals
from torchvision.transforms import transforms as tv_transforms
import numpy as np
import cv2
from PIL import Image
from collections import OrderedDict
import torch.nn as nn
import torchvision

# === PyTorch 2.6+ global allow ===
add_safe_globals([
    torch.nn.Sequential,
    torch.nn.Conv2d,
    torch.nn.ReLU,
    torch.nn.Linear,
    torch.nn.BatchNorm2d,
    torch.nn.AvgPool2d,
    torch.nn.MaxPool2d,
    torch.nn.Flatten,
    torchvision.transforms.Compose,
    torch.nn.modules.upsampling.Upsample,
])

# === Güvenli Tensor Dönüştürme ===
def to_tensor_safe(v):
    if isinstance(v, torch.Tensor):
        return v
    elif isinstance(v, np.ndarray):
        return torch.from_numpy(v)
    elif isinstance(v, (float, int)):
        return torch.tensor(v)
    else:
        raise ValueError(f"Unsupported type in conversion to tensor: {type(v)}")

# === Model Yükleyici ===
def load_models():
    base_path = os.path.join("Model", "EfficientAD")
    model_path = os.path.join(base_path, "efficientad_complete_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    print("🔄 efficientad_complete_model.pth yükleniyor...")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    teacher = checkpoint["teacher"]
    student = checkpoint["student"]
    autoencoder = checkpoint["autoencoder"]

    quantiles_raw = checkpoint.get("map_quantiles", {})
    stats_raw = checkpoint.get("normalization_params", {})

    map_quantiles = {k: to_tensor_safe(v) for k, v in quantiles_raw.items()}
    stats = {
        "mean": to_tensor_safe(stats_raw.get("mean", torch.zeros(1, 384, 1, 1))),
        "std": to_tensor_safe(stats_raw.get("std", torch.ones(1, 384, 1, 1)))
    }

    teacher.eval()
    student.eval()
    autoencoder.eval()

    print("✅ Model bileşenleri başarıyla yüklendi.")
    return teacher, student, autoencoder, map_quantiles, stats

# === Görsel Dönüştürücü ===
def transform_image(image, image_size=256):
    transform = tv_transforms.Compose([
        tv_transforms.Resize((image_size, image_size)),
        tv_transforms.ToTensor(),
        tv_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    ])
    original = image.copy()
    return transform(image).unsqueeze(0), original

# === Tahmin Fonksiyonu ===
def predict(image_tensor, teacher, student, autoencoder, quantiles, stats, on_gpu=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_tensor = image_tensor.to(device)
    teacher = teacher.to(device)
    student = student.to(device)
    autoencoder = autoencoder.to(device)

    for key in quantiles:
        quantiles[key] = to_tensor_safe(quantiles[key]).to(device)

    stats["mean"] = to_tensor_safe(stats.get("mean")).to(device)
    stats["std"] = to_tensor_safe(stats.get("std")).to(device)

    with torch.no_grad():
        teacher_output = teacher(image_tensor)
        teacher_output = (teacher_output - stats['mean']) / stats['std']

        student_output = student(image_tensor)
        autoencoder_output = autoencoder(image_tensor)

        map_st = torch.mean((teacher_output - student_output[:, :384])**2, dim=1, keepdim=True)
        map_ae = torch.mean((autoencoder_output - student_output[:, 384:])**2, dim=1, keepdim=True)

        if all(k in quantiles for k in ['q_st_start', 'q_st_end', 'q_ae_start', 'q_ae_end']):
            map_st = 0.1 * (map_st - quantiles['q_st_start']) / (quantiles['q_st_end'] - quantiles['q_st_start'] + 1e-6)
            map_ae = 0.1 * (map_ae - quantiles['q_ae_start']) / (quantiles['q_ae_end'] - quantiles['q_ae_start'] + 1e-6)

        anomaly_map = 0.5 * map_st + 0.5 * map_ae

        anomaly_map = torch.nn.functional.interpolate(anomaly_map, size=(256, 256), mode='bilinear', align_corners=False)

        anomaly_map = anomaly_map.clamp(0, 1).cpu().squeeze().numpy()

        score = float(anomaly_map.max())
        label = "Anomali" if score > 0.385 else "Normal"

        return anomaly_map, score, label

# === Görsel Üzerine Overlay Çizici ===
def create_overlay(original_image, anomaly_map, threshold):
    if isinstance(original_image, Image.Image):
        orig_np = np.array(original_image.convert('RGB'))
    elif isinstance(original_image, np.ndarray):
        orig_np = original_image.copy()
    else:
        raise TypeError("original_image must be a PIL.Image or a numpy.ndarray")

    heatmap = cv2.normalize(anomaly_map, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    binary_mask = (anomaly_map > threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    overlay = orig_np.copy()
    if contours:
        cv2.drawContours(overlay, contours, -1, (255, 0, 0), 2)

    return overlay, heatmap_color


def preprocess_input_image(image_path, target_size=(256, 256)):
    """
    Görüntüyü modelin beklediği şekilde işler:
    - Grayscale olarak yükler
    - Gaussian Blur + Otsu Threshold ile maske çıkarır
    - Maskeyi uygular
    - Yeniden boyutlandırır
    - Normalleştirip 3 kanallı hale getirir
    - PIL.Image formatında döndürür
    """
    # 1. Grayscale yükle
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None or img.size == 0:
        raise ValueError(f"⚠️ Görüntü okunamadı veya bozuk: {image_path}")

    # 2. Gaussian Blur ve Otsu Threshold
    try:
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    except Exception as e:
        raise RuntimeError(f"Otsu threshold uygulanamadı: {e}")

    # 3. Maskeleme
    masked = cv2.bitwise_and(img, img, mask=thresh)

    # 4. Boyut kontrolü
    if masked.shape[0] < 10 or masked.shape[1] < 10:
        raise ValueError(f"Görüntü çok küçük: {masked.shape}")

    # 5. Yeniden boyutlandır
    try:
        resized = cv2.resize(masked, target_size, interpolation=cv2.INTER_AREA)
    except Exception as e:
        raise RuntimeError(f"Boyutlandırma hatası: {e}")

    # 6. Normalize (0-1)
    normalized = resized.astype(np.float32) / 255.0

    # 7. 3 kanala çıkar (model RGB bekliyor)
    normalized_3ch = np.repeat(normalized[:, :, np.newaxis], 3, axis=2)  # shape: (H,W,3)

    # 8. PIL Image'a dönüştür
    try:
        final_image = Image.fromarray((normalized_3ch * 255).astype(np.uint8))
    except Exception as e:
        raise RuntimeError(f"PIL'e dönüştürme hatası: {e}")

    return final_image

