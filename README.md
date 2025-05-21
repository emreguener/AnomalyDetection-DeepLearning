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
<pre><code>!python '/content/AnomalyDetection-DeepLearning/EfficientAD/efficientad.py'' --mvtec_ad_path "/content/drive/MyDrive/wood_otsu" --subdataset "wood" --train_steps 1500 -o "/content/drive/MyDrive/EfficientAD_/Deneme 5 Ağırlıklar"</code></pre>

### 2. ⚡ FastFlow
- Normal yüzeylerin akış haritalarını tersine çevirerek kusur tespiti  
- [`FastFlow_Run.ipynb`](./FastFlow_Run%20%281%29.ipynb) 
<pre><code> !python /content/AnomalyDetection-DeepLearning/FastFlow/main.py --cfg /content/FastFlow/configs/densenet121.yaml --data /content/fast_flow_dataset --cat wood</code></pre>

### 3. 🔬 INP-Former
- Transformer tabanlı bilgi yoğunlaştırma  
- [`INP_Former_Run.ipynb`](./INP_Former_Run%20%281%29.ipynb)
<pre><code>'''
  !python /content/AnomalyDetection-DeepLearning/INP_FORMER/INP_Former_Single_Class.py \
  --dataset MVTec-AD \
  --data_path /content/drive/MyDrive/wood_dataset \
  --item wood \
  --batch_size 8\
  --total_epochs 100 \
  --phase train \
  --save_dir /content/inp_result \
  --encoder dinov2reg_vit_large_14 \
  --input_size 252 \
  --crop_size 252 \
  --INP_num 6'''</code></pre>

### 4. 🧪 PBAS
- Patch-tabanlı skor üretimi  
- [`PBAS_Run.ipynb`](./PBAS_Run%20%281%29.ipynb)
  <pre><code>!python /content/AnomalyDetection-DeepLearning/PBAS/pbas.py \
  --gpu 0 \
  --seed 0 \
  net \
    -b wideresnet50 \
    -le layer2 \
    -le layer3 \
    --pretrain_embed_dimension 1536 \
    --target_embed_dimension 1536 \
    --patchsize 3 \
    --meta_epochs 20 \
    --eval_epochs 5 \
    --dsc_layers 2 \
    --dsc_hidden 1024 \
    --pre_proj 1 \
    --k 0.25 \
  dataset \
    --batch_size 8 \
    --resize 256 \
    --imagesize 256 \
    {flags} mvtec {datapath}
"""
</code></pre>

### 5. 🔹 SimpleNet
- Basit ama etkili segmentasyon modeli  
- [`SimpleNet_Run.ipynb`](./SimpleNet_Run.ipynb)
<pre><code>```
!python /content/AnomalyDetection-DeepLearning/SimpleNet/main.py \
--gpu 0 \
--seed 0 \
--log_group simplenet_wood_fast \
--log_project WoodOnly \
--results_path results \
--run_name wood_fast_run \
net \
-b wideresnet50 \
-le layer2 \
-le layer3 \
--pretrain_embed_dimension 1024 \
--target_embed_dimension 1024 \
--patchsize 3 \
--meta_epochs 15 \
--embedding_size 256 \
--gan_epochs 5 \
--noise_std 0.015 \
--dsc_hidden 768 \
--dsc_layers 2 \
--dsc_margin .5 \
--pre_proj 2 \
dataset \
--batch_size 8 \
--resize 256 \
--imagesize 256 \
-d wood \
mvtec /content/drive/MyDrive/wood_dataset
```</code></pre>

### 6. 🔸 UniNet
- DFS + Student + Teacher birleşimli çok bölümlü model  
- [`UniNet_Run.ipynb`](./UniNet_Run.ipynb)
<pre><code>!python '/content/AnomalyDetection-DeepLearning/UniNet/UniNet/main.py' \ --dataset "MVTec AD" \ --setting oc \ --train_and_test_all \ --is_saved \ --save_dir "./results" \ --epoch 90 </code></pre>
---

## ⚠️ Uyarılar

* Kodlar yalnızca `wood` alt veri kümesiyle çalışacak şekilde optimize edilmelidir.
* Tüm modeller aynı klasör yapısını bekler. Yukarıda gözüken çalışma kodlarında lütfen veri yollarını uygun formatta verin. Veri yollarını notebook içinde doğrulayın.

---


