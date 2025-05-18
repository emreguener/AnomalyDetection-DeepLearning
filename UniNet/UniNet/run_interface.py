import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms as T
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score
from scipy.ndimage import gaussian_filter
import glob

from UniNet_lib.model import UniNet
from UniNet_lib.resnet import wide_resnet50_2
from UniNet_lib.de_resnet import de_wide_resnet50_2
from UniNet_lib.DFS import DomainRelated_Feature_Selection
from UniNet_lib.mechanism import weighted_decision_mechanism
from utils import to_device, load_weights

# === Config ===
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

c = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Model Yükle ===
def load_model_from_ckpt(c, device):
    tt, bn = wide_resnet50_2(c, pretrained=True)
    tt.layer4 = None
    tt.fc = None
    st = de_wide_resnet50_2(pretrained=False)
    dfs = DomainRelated_Feature_Selection()
    [tt, bn, st, dfs] = to_device([tt, bn, st, dfs], device)

    model = UniNet(c, tt, tt, bn, st, DFS=dfs).to(device).eval()

    ckpt_path = os.path.join("./ckpts", c.dataset, c._class_, f"{c.ckpt_suffix}.pth")
    weights = torch.load(ckpt_path, map_location=device)

    model.t.t_t.load_state_dict(weights["tt"])
    model.bn.bn.load_state_dict(weights["bn"])
    model.s.s1.load_state_dict(weights["st"])
    model.dfs.load_state_dict(weights["dfs"])

    print("✅ Model başarıyla yüklendi.")
    return model

# === Preprocessing + Predict
def preprocess_image(image_path, c):
    transform = T.Compose([
        T.Resize((c.image_size, c.image_size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0).to(device), image

def predict_score_and_map(model, tensor, c):
    model.train_or_eval(type='eval')
    with torch.no_grad():
        t_feats, s_feats = model(tensor)
        output_list = [[] for _ in range(c.T * 3)]
        for l, (t, s) in enumerate(zip(t_feats, s_feats)):
            sim = 1 - F.cosine_similarity(t, s)
            output_list[l].append(sim)
        score, amap = weighted_decision_mechanism(1, output_list, c.alpha, c.beta)

        # 💡 Eğer score bir liste ise ilkini al
        score = score[0] if isinstance(score, (list, tuple)) else score
        amap = gaussian_filter(amap, sigma=4).squeeze()
        return score, amap

# === Görselleştirme ===
def visualize(image_pil, anomaly_map, gt_mask, save_path):
    anomaly_map_norm = cv2.normalize(anomaly_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(anomaly_map_norm, cv2.COLORMAP_JET)

    img_bgr = cv2.cvtColor(np.array(image_pil.resize((256, 256))), cv2.COLOR_RGB2BGR)
    _, bin_thresh = cv2.threshold(anomaly_map_norm, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bin_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = cv2.drawContours(img_bgr.copy(), contours, -1, (0, 255, 0), 2)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 4, 1); plt.imshow(image_pil.resize((256, 256))); plt.title("Original"); plt.axis('off')
    plt.subplot(1, 4, 2); plt.imshow(anomaly_map, cmap='jet'); plt.title("Anomaly Map"); plt.axis('off')
    plt.subplot(1, 4, 3); plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB)); plt.title("Contours"); plt.axis('off')
    plt.subplot(1, 4, 4); plt.imshow(gt_mask, cmap='gray'); plt.title("Ground Truth"); plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"📷 Saved: {save_path}")
    plt.close()

# === Tek Görsel Test ===
def test_single_image():
    image_path = "/content/drive/MyDrive/Wood_Dataset/mvtec/wood/test/defect/100000003.jpg"
    gt_path = "/content/drive/MyDrive/Wood_Dataset/mvtec/wood/ground_truth/defect/100000003_mask.jpg"
    tensor, image_pil = preprocess_image(image_path, c)
    score, amap = predict_score_and_map(model, tensor, c)
    gt = Image.open(gt_path).convert("L").resize((256, 256)) if os.path.exists(gt_path) else np.zeros((256, 256))
    visualize(image_pil, amap, gt, "visual_result.png")

# === Tüm Veride Test ve F1 ===
def test_all_images():
    image_root = "/content/drive/MyDrive/Wood_Dataset/mvtec/wood/test"
    gt_dir = "/content/drive/MyDrive/Wood_Dataset/mvtec/wood/ground_truth/defect"
    image_paths = sorted(glob.glob(os.path.join(image_root, "good", "*.jpg")) +
                         glob.glob(os.path.join(image_root, "defect", "*.jpg")))

    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    results = []
    for path in image_paths:
        name = os.path.basename(path)
        label = 1 if "defect" in path else 0
        image = Image.open(path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)
        score, _ = predict_score_and_map(model, tensor, c)
        results.append({"image": name, "score": score, "label": label})

    df = pd.DataFrame(results)
    best_f1, best_thresh = 0, 0
    for t in np.arange(min(df["score"]), max(df["score"]), 0.01):
        preds = (df["score"] > t).astype(int)
        f1 = f1_score(df["label"], preds)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    df["predicted"] = (df["score"] > best_thresh).astype(int)
    acc = accuracy_score(df["label"], df["predicted"])
    print(f"📈 Best Threshold: {best_thresh:.4f} | F1: {best_f1:.4f} | Accuracy: {acc:.4f}")
    df.to_csv("anomaly_scores.csv", index=False)
    



# === Ana Çalıştırma ===
if __name__ == "__main__":
    model = load_model_from_ckpt(c, device)
    test_all_images()
    test_single_image()
