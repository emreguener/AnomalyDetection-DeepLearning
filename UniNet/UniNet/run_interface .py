import os
import sys
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms as T
import torch.nn.functional as F
import pandas as pd
import glob
from sklearn.metrics import f1_score, accuracy_score

# === Yol ayarları ===
sys.path.append(os.path.abspath("."))
from UniNet_lib.model import UniNet
from UniNet_lib.DFS import DomainRelated_Feature_Selection
from UniNet_lib.resnet import wide_resnet50_2
from UniNet_lib.de_resnet import de_wide_resnet50_2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_root = "/content/drive/MyDrive/Wood_Dataset/mvtec/wood/test"
gt_dir = "/content/drive/MyDrive/Wood_Dataset/mvtec/wood/ground_truth/defect"
save_dir = "./visuals"
ckpt_path = "./ckpts/MVTec AD/wood/BEST_P_PRO.pth"
os.makedirs(save_dir, exist_ok=True)

# === Görselleri al ===
image_paths = sorted(glob.glob(os.path.join(image_root, "good", "*.jpg")) +
                     glob.glob(os.path.join(image_root, "defect", "*.jpg")))

# === Model Yükle ===
def to_device(modules, device):
    return [m.to(device) for m in modules]

def load_model():
    class DummyConfig:
        _class_ = "wood"
        T = 2

    c = DummyConfig()
    tt, bn = wide_resnet50_2(c, pretrained=True)
    tt.layer4 = None
    tt.fc = None
    st = de_wide_resnet50_2(pretrained=False)
    dfs = DomainRelated_Feature_Selection()
    [tt, bn, st, dfs] = to_device([tt, bn, st, dfs], device)

    model = UniNet(c, tt, tt, bn, st, DFS=dfs).to(device).eval()

    weights = torch.load(ckpt_path, map_location=device)
    model.t.t_t.load_state_dict(weights["tt"])
    model.bn.bn.load_state_dict(weights["bn"])
    model.s.s1.load_state_dict(weights["st"])
    model.dfs.load_state_dict(weights["dfs"])
    print("✅ Model ağırlıkları başarıyla yüklendi.")
    return model

# === Inference ===
def predict_anomaly(model, image_tensor):
    image_tensor = image_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        t_feats, s_feats = model(image_tensor)
        s_selected = model.feature_selection(t_feats, s_feats, max=True)

        anomaly_maps = []
        for t, s in zip(t_feats, s_selected):
            amap = torch.mean(torch.abs(t - s), dim=1, keepdim=True)
            amap_resized = F.interpolate(amap, size=(256, 256), mode='bilinear', align_corners=False)
            anomaly_maps.append(amap_resized)

        anomaly_map = torch.mean(torch.stack(anomaly_maps), dim=0).squeeze().cpu().numpy()
        return anomaly_map

# === Yardımcılar ===
def t2np(tensor):
    return tensor.detach().cpu().numpy() if torch.is_tensor(tensor) else tensor

def preprocess_image_for_visualization(img_tensor):
    img_np = t2np(img_tensor)
    if img_np.ndim == 3 and img_np.shape[0] == 3:
        img_np = np.transpose(img_np, (1, 2, 0))
    if img_np.dtype in [np.float32, np.float64]:
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
    elif img_np.dtype != np.uint8:
        img_np = img_np.astype(np.uint8)
    return img_np

def overlay_heatmap_on_image(image, anomaly_map, alpha=0.6):
    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    anomaly_map_uint8 = (np.clip(anomaly_map, 0, 1) * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(anomaly_map_uint8, cv2.COLORMAP_JET)
    return cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)

def draw_contours_on_image(image, binary_mask, color=(0, 255, 0), thickness=2):
    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = image.copy()
    cv2.drawContours(out, contours, -1, color, thickness)
    return out

def load_gt_mask(image_path, gt_root):
    if "defect" in image_path:
        image_name = os.path.basename(image_path)
        gt_path = os.path.join(gt_root, image_name)
        if os.path.exists(gt_path):
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            return cv2.resize(gt, (256, 256), interpolation=cv2.INTER_NEAREST).astype(np.float32) / 255.0
    return np.zeros((256, 256), dtype=np.float32)

def visualize_and_analyze(img_tensor, anomaly_map, gt_mask, save_path, threshold=0.38):
    img = preprocess_image_for_visualization(img_tensor)
    anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
    pred_bin = (anomaly_map > threshold).astype(np.uint8)
    gt_bin = (gt_mask > 0.1).astype(np.uint8)

    heat_overlay = overlay_heatmap_on_image(img, anomaly_map)
    pred_contours = draw_contours_on_image(img, pred_bin)

    num_plots = 4 if gt_mask.sum() > 0 else 3
    fig, axs = plt.subplots(1, num_plots, figsize=(20, 5))
    axs[0].imshow(img)
    axs[0].set_title("Original")
    axs[1].imshow(heat_overlay)
    axs[1].set_title("Anomaly Map")
    axs[2].imshow(pred_contours)
    axs[2].set_title("Contours")
    if gt_mask.sum() > 0:
        gt_viz = (gt_bin * 255).astype(np.uint8)
        axs[3].imshow(gt_viz, cmap='gray')
        axs[3].set_title("Ground Truth")
    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved: {save_path}")

# === Ana Döngü ===
print(f"Toplam {len(image_paths)} görsel bulundu.")
print("Örnek görseller:", image_paths[:3])

model = load_model()
transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])
results = []

# 1. Pass – Skorları hesapla
for image_path in image_paths:
    image_name = os.path.basename(image_path)
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img)
    gt_mask = load_gt_mask(image_path, gt_dir)
    anomaly_map = predict_anomaly(model, img_tensor)

    norm_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
    avg_score = np.mean(norm_map)
    max_score = np.max(norm_map)
    binary_gt = (gt_mask > 0.1).astype(np.uint8)
    has_defect = int(binary_gt.sum() > 0)

    results.append({
        "image": image_name,
        "avg_score": avg_score,
        "max_score": max_score,
        "label": has_defect
    })

# 2. Threshold optimizasyonu
df = pd.DataFrame(results)
thresholds = np.arange(0.0, 1.0, 0.01)
best_f1, best_thresh = 0, 0
for t in thresholds:
    preds = (df["avg_score"] > t).astype(int)
    f1 = f1_score(df["label"], preds, zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

# 3. Tahmin sonuçlarını ekle
df["predicted_label"] = (df["avg_score"] > best_thresh).astype(int)
acc = accuracy_score(df["label"], df["predicted_label"])
print(f"🏆 En iyi threshold: {best_thresh:.2f}")
print(f"📈 F1-score: {best_f1:.4f} | ✅ Accuracy: {acc:.4f}")

# 4. CSV kaydet
csv_path = os.path.join(save_dir, "anomaly_scores_with_predictions.csv")
df.to_csv(csv_path, index=False)
print(f"📄 Güncellenmiş CSV kaydedildi: {csv_path}")

# 5. Visualization (threshold'a göre)
for image_path in image_paths:
    image_name = os.path.basename(image_path)
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img)
    gt_mask = load_gt_mask(image_path, gt_dir)
    anomaly_map = predict_anomaly(model, img_tensor)
    save_path = os.path.join(save_dir, f"{os.path.splitext(image_name)[0]}_viz.png")
    visualize_and_analyze(img_tensor, anomaly_map, gt_mask, save_path, threshold=best_thresh)
