# AnomalyDetection-DeepLearning

Anomaly Detection with Deep Learning

Bu repository, çeşitli derin öğrenme modellerini kullanarak endüstriyel anomali tespiti için uygulamalar içermektedir.

## Setup

### Gerekli Paketler

```bash
Python==3.10
torch==1.13.0
torchvision==0.14.0
tifffile==2021.7.30
tqdm==4.56.0
scikit-learn==1.2.2
```

### Mvtec AD Değerlendirme Paketleri

```bash
numpy==1.18.5
Pillow==7.0.0
scipy==1.7.1
tabulate==0.8.7
tifffile==2021.7.30
tqdm==4.56.0
```

## Veri Seti Yapısı

Tüm modeller için genel veri seti yapısı aşağıdaki gibi olmalıdır:

```bash
dataset/
├── train/
│   └── good/         # Sadece normal (kusursuz) örnekler
├── test/
│   ├── good/         # Normal test örnekleri
│   └── defect/       # Kusurlu test örnekleri (anomaliler)
└── ground_truth/
    └── defect/       # Anomaliler için maske görüntüleri (*_mask.jpg)
```

---

## EfficientAD

EfficientAD, gerçek zamanlı uygulamalarda kullanılabilen, hızlı ve doğru anomali tespiti yapabilen bir derin öğrenme modelidir. Model, öğretmen-öğrenci mimarisi ve otokodlayıcı kullanarak hem yapısal hem de mantıksal anomalileri tespit edebilmektedir.

### Modeli Çalıştırma Adımları

```bash
1. Google Colab'da EfficientAD_13_05.ipynb dosyasını açın
2. GPU hızlandırıcıyı etkinleştirin:
   Çalışma Zamanı > Çalışma zamanı türünü değiştir > GPU
3. Google Drive'ı bağlayın:
```

```python
from google.colab import drive
drive.mount('/content/drive')
import os
os.chdir('/content/drive/MyDrive/')
```

```bash
4. Dosya yollarını kendi dizin yapınıza göre güncelleyin
5. Notebook hücrelerini sırayla çalıştırın
```

---

## FastFlow

FastFlow, normal verilerin özellik dağılımını öğrenerek anomalileri aykırılık olarak tespit eden bir yaklaşım sunar. Gerçek zamanlı anomali tespiti için optimize edilmiştir.

### Modeli Çalıştırma Adımları

```bash
- FastFlow klasörü altındaki notebook dosyasını açın
- Colab veya lokal ortamda çalıştırabilirsiniz
- Veri yollarını kendi klasör yapınıza göre ayarlayın
- Tüm adımlar kod içerisinde yorumlarla belirtilmiştir
```

---

## INP\_FORMER

INP\_FORMER, Transformer tabanlı bir yaklaşımla endüstriyel anomali tespiti yapmayı hedefler. Görsel dizi öğrenme mantığını kullanarak karmaşık yapıları algılayabilir.

### Modeli Çalıştırma Adımları

```bash
- INP_FORMER klasöründe inp_former_10_05.ipynb dosyasını inceleyin
- Ortamınızın gerekli paketleri içerdiğinden emin olun
- Kendi verisetinize göre parametreleri ayarlayarak notebook'u çalıştırın
```

---

## PBAS

PBAS (Pixel-Based Adaptive Segmenter), geleneksel görüntü işleme teknikleriyle geliştirilmiş bir anomali tespiti yaklaşımıdır. Hafif yapısıyla çok düşük sistemlerde bile çalışabilir.

### Modeli Çalıştırma Adımları

```bash
- PBAS klasörü altındaki script dosyalarını inceleyin
- Her adımda girdi-çıktı yollarını doğru belirleyin
- Çalıştırmak için: python main.py
```

---

## SimpleNet

SimpleNet, minimal bir sinir ağı modeliyle baseline anomali tespiti performansı sağlamak için tasarlanmıştır. Anlaması ve modifiye etmesi kolay bir yapıya sahiptir.

### Modeli Çalıştırma Adımları

```bash
- SimpleNet/SimpleNetRunpynb.ipynb notebook'unu açın
- Verisetini belirttikten sonra, tüm kod sırasıyla çalıştırılabilir
- Basit yapısı sayesinde öğrenme amaçlı kullanımlar için uygundur
```

---

## UniNet

UniNet, çoklu anomaly type'ları için tek bir çözüm sunan esnek bir mimaridir. Birleştirilmiş öğrenme yetenekleri sayesinde farklı alanlarda uygulanabilir.

### Modeli Çalıştırma Adımları

```bash
- UniNet klasöründeki script dosyalarını çalıştırın
- GPU desteklidir, CUDA uyumlu ortamlarda kullanım tavsiye edilir
- Parametreleri README veya script içinde belirtilen şekilde düzenleyin
```

---

## Katkıda Bulunma

Proje katkılarına açıktır. Her model klasörü kendi özgün kod yapısına sahiptir. Yeni modeller ekleyebilir veya mevcut modelleri geliştirebilirsiniz.

Pull request'leriniz değerlendirilmek üzere beklenmektedir.

---
