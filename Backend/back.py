import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

def debug_predict(image_tensor, model, device):
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        output = model(image_tensor)

        # 🔍 Model çıktısının yapısını incele
        if isinstance(output, tuple) and isinstance(output[0], list):
            anomaly_map = output[0][0]
        else:
            raise ValueError("⚠️ Model çıktısı tuple[list[tensor]] yapısında değil.")

        if not isinstance(anomaly_map, torch.Tensor):
            raise TypeError(f"❌ anomaly_map bir Tensor değil: {type(anomaly_map)}")

        # 🔍 Anomaly map işleme
        anomaly_map = anomaly_map.mean(dim=1, keepdim=True)
        anomaly_map = torch.nn.functional.interpolate(anomaly_map, size=(256, 256), mode="bilinear", align_corners=False)
        anomaly_map = anomaly_map.clamp(0, 1).cpu().squeeze().numpy()

        # 🔍 Skor analizleri
        raw_max = float(anomaly_map.max())
        raw_mean = float(anomaly_map.mean())
        raw_percentile = float(np.percentile(anomaly_map, 99))

        # 🔍 DFS ağırlıkları varsa kontrol et
        if hasattr(model, 'dfs'):
            try:
                print("theta1 mean:", model.dfs.theta1.mean().item())
                print("theta2 mean:", model.dfs.theta2.mean().item())
                print("theta3 mean:", model.dfs.theta3.mean().item())
            except Exception as e:
                print("(⚠️ DFS parametreleri okunamadı)", e)

        print(f"\n🧠 Skorlar:\n - Max: {raw_max:.4f}\n - Mean: {raw_mean:.4f}\n - 99th %: {raw_percentile:.4f}")

        # 🔍 Karar
        threshold = 0.15  # yeni düşük eşik
        label = "Anomali" if raw_max > threshold else "Normal"
        print(f"🔎 Tahmin: {label} (eşik: {threshold})")

        # 🔍 Haritayı görselleştir
        plt.imshow(anomaly_map, cmap='jet')
        plt.colorbar()
        plt.title(f"Anomaly Map\nMax: {raw_max:.3f}, Mean: {raw_mean:.3f}, 99%: {raw_percentile:.3f}\nLabel: {label}")
        plt.tight_layout()
        plt.show()

        return anomaly_map, raw_max, label


from helpers_uninet import load_uninet_model, preprocess_input_image, transform_image

import torch

# BEST_P_PRO.pth dosyasının yolunu belirt
ckpt_path = "C:\\Users\\Bentego Personel\\Documents\\Baturhan\\Neural_Project\\Backend\\Model\\UniNet\\BEST_P_PRO.pth"  # Gerekirse tam yolu düzelt

checkpoint = torch.load(ckpt_path, map_location="cpu")

print("📦 Checkpoint tipi:", type(checkpoint))

# Eğer dict ise:
if isinstance(checkpoint, dict):
    print("🔑 Anahtarlar:", list(checkpoint.keys())[:10])

    # Eğer doğrudan dfs.weight gibi şeyler varsa
    if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        print("✅ Doğrudan state_dict (tek düzeyli)")
        dfs_weights = {k: v.shape for k, v in checkpoint.items() if 'dfs' in k.lower()}
        print("🔍 DFS Ağırlıkları:", dfs_weights)

    # Eğer alt modüller varsa (dict içinde dict)
    elif all(isinstance(v, dict) for v in checkpoint.values()):
        print("✅ dict içinde dict (modül bazlı kaydedilmiş)")
        for module_name in checkpoint.keys():
            print(f"📂 {module_name}: {list(checkpoint[module_name].keys())[:5]}")

else:
    print("❌ Beklenmeyen format:", type(checkpoint))
