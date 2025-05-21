# 📊 AnomalyDetection-DeepLearning

Bu depo, ahşap yüzeylerdeki kusurları tespit etmek için derin öğrenme tabanlı farklı modelleri kullanarak anomali tespiti yapar. MVTEC AD veri setinin sadece **wood** alt veri kümesi kullanılmıştır.

---

## 🔗 Repo Klonlama ve Ortam Kurulumu

```bash
git clone https://github.com/emreguener/AnomalyDetection-DeepLearning.git
cd AnomalyDetection-DeepLearning
pip install -r requirements.txt
```

> **Not:** Google Colab ortamında çalışacak şekilde notebook dosyaları optimize edilmiştir.

---

## 📂 Veri Seti Yapısı

Bu projede yalnızca **wood** alt veri kümesi kullanılmaktadır. Lütfen aşağıdaki dizin yapısına dikkat ederek veri setini yerleştiriniz:

```
Wood_dataset/
├── wood/
│   ├── train/
│   │   └── good/
│   ├── test/
│   │   ├── good/
│   │   └── defect/
│   └── ground_truth/
│       └── defect/
```

**Veri Yolu örneği (notebook içinde):**

```python
dataset_path = "/content/drive/MyDrive/Wood_dataset/wood"
```

---

## 📄 Gerekli Kütüphaneler (requirements.txt)

```txt
Pillow
collections
functools
glob
google
joblib
main
matplotlib
models
numpy
opencv-python
optuna
os
pandas
pickle
random
scikit-learn
seaborn
shutil
skimage
sys
tifffile
time
torch
torchvision
tqdm
traceback
utils
warnings
xgboost
yaml
zipfile
git+https://github.com/VLL-HD/FrEIA.git
```

---

## 🔧 Modellerin Kullanımı (Notebook Yolları)

Her modelin `.ipynb` dosyası ayrıdır ve tam çalışabilir haldedir.

### 1. 🧠 EfficientAD
- Student-Teacher yapısı ile anomaly segmentasyonu  
- [`EfficientAD_Run.ipynb`](./EfficientAD/EfficientAD_Run.ipynb)

### 2. ⚡ FastFlow
- Normal yüzeylerin akış haritalarını tersine çevirerek kusur tespiti  
- [`FastFlow_Run.ipynb`](./FastFlow_Run%20%281%29.ipynb)

### 3. 🔬 INP-Former
- Transformer tabanlı bilgi yoğunlaştırma  
- [`INP_Former_Run.ipynb`](./INP_Former_Run%20%281%29.ipynb)

### 4. 🧪 PBAS
- Patch-tabanlı skor üretimi  
- [`PBAS_Run.ipynb`](./PBAS_Run%20%281%29.ipynb)

### 5. 🔹 SimpleNet
- Basit ama etkili segmentasyon modeli  
- [`SimpleNet_Run.ipynb`](./SimpleNet_Run.ipynb)

### 6. 🔸 UniNet
- DFS + Student + Teacher birleşimli çok bölümlü model  
- [`UniNet_Run.ipynb`](./UniNet_Run.ipynb)
<pre><code>```bash !python '/content/UniNet/main.py' \ --dataset "MVTec AD" \ --setting oc \ --train_and_test_all \ --is_saved \ --save_dir "./results" \ --epoch 90 ``` </code></pre>
---

## ⚠️ Uyarılar

* Kodlar yalnızca `wood` alt veri kümesiyle çalışacak şekilde optimize edilmelidir.
* Tüm modeller aynı klasör yapısını bekler. Lütfen veri yollarını notebook içinde doğrulayın.

---


