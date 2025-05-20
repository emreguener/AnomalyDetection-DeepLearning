import os
import sys
import torch
import numpy as np
import cv2
import copy
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.metrics import f1_score, accuracy_score
import torchvision.transforms as T
import torch.nn.functional as F

# === Yol Ayarları ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UNINET_DIR = os.path.join(BASE_DIR, "Model", "UniNet")
sys.path.append(os.path.join(BASE_DIR, "Model"))

from UniNet.model import UniNet
from UniNet.resnet import wide_resnet50_2
from UniNet.de_resnet import de_wide_resnet50_2
from UniNet.DFS import DomainRelated_Feature_Selection
from UniNet.mechanism import weighted_decision_mechanism


# === Konfigürasyon ===
class Config:
    dataset = "MVTec AD"
    _class_ = "wood"
    setting = "oc"
    domain = "industrial"
    image_size = 256
    batch_size = 1
    T = 2
    alpha = 0.01
    beta = 3e-5
    ckpt_suffix = "BEST_P_PRO"


# === Aygıt Seçimi ===
c = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === Ağırlık Yükleme ===
def load_model_from_ckpt(c, device):
    tt, bn = wide_resnet50_2(c, pretrained=True)
    tt.layer4 = None
    tt.fc = None
    st = de_wide_resnet50_2(pretrained=False)
    dfs = DomainRelated_Feature_Selection()
    [tt, bn, st, dfs] = [m.to(device) for m in [tt, bn, st, dfs]]

    model = UniNet(c, tt, tt, bn, st, DFS=dfs).to(device).eval()
    ckpt_path = os.path.join(UNINET_DIR, f"{c.ckpt_suffix}.pth")
    weights = torch.load(ckpt_path, map_location=device)

    model.t.t_t.load_state_dict(weights["tt"])
    model.bn.bn.load_state_dict(weights["bn"])
    model.s.s1.load_state_dict(weights["st"])
    model.dfs.load_state_dict(weights["dfs"])

    print("✅ Model başarıyla yüklendi.")
    return model


# === Görsel Hazırlama ve Tahmin ===
def preprocess_input_image(image_path, image_size=256):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Görüntü bulunamadı: {image_path}")
    img = cv2.resize(img, (image_size, image_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


def transform_image(image, c):
    transform = T.Compose([
        T.Resize((c.image_size, c.image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device), image


def predict(model, tensor, c):
    model.train_or_eval(type='eval')
    with torch.no_grad():
        t_feats, s_feats = model(tensor)
        output_list = [[] for _ in range(c.T * 3)]
        for l, (t, s) in enumerate(zip(t_feats, s_feats)):
            sim = 1 - F.cosine_similarity(t, s)
            output_list[l].append(sim)
        score, amap = weighted_decision_mechanism(1, output_list, c.alpha, c.beta)

        score = score[0] if isinstance(score, (list, tuple)) else score
        amap = gaussian_filter(amap, sigma=4).squeeze()
        return score, amap


# === Isı Haritası ve Overlay Oluşturma ===
def create_overlay(original_image, anomaly_map, threshold):
    orig_np = np.array(original_image.convert('RGB'))
    anomaly_map = cv2.resize(anomaly_map, (256, 256))
    heatmap = cv2.normalize(anomaly_map, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    binary_mask = (anomaly_map > threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    overlay = orig_np.copy()
    if contours:
        cv2.drawContours(overlay, contours, -1, (255, 0, 0), 2)

    return overlay, heatmap_color


# === Tek Görsel Test Fonksiyonu ===
def test_single_image(model, image_path, c, threshold=0.185):
    image = preprocess_input_image(image_path, c.image_size)
    tensor, _ = transform_image(image, c)
    score, amap = predict(model, tensor, c)
    label = "Anomali" if score > threshold else "Normal"
    print(f"✅ Skor: {score:.4f} → Tahmin: {label}")
    overlay, heatmap = create_overlay(image, amap, threshold)

    # Görseli kaydet
    cv2.imwrite("overlay_result.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite("heatmap_result.png", cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
    return score, label
