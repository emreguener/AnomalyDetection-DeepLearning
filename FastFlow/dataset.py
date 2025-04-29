import os
from glob import glob

import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms


class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, category, input_size, is_train=True):
        self.image_transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.is_train = is_train

        if is_train:
            # Eğitim verileri sadece kusursuz (good) görseller
            self.image_files = glob(os.path.join(root, category, "train", "good", "*.jpg"))
            print(f"[TRAIN] Eğitim verisi bulundu: {len(self.image_files)} dosya")
        else:
            # Test verileri: good + defect (recursive=True ile tüm alt klasörleri tarar)
            self.image_files = glob(os.path.join(root, category, "test", "**", "*.jpg"), recursive=True)
            print(f"[TEST] Test verisi bulundu: {len(self.image_files)} dosya")
            if len(self.image_files) > 0:
                print(f"[TEST] Örnek dosya: {self.image_files[0]}")
            else:
                print("❌ Test verisi bulunamadı. Yol yapılandırmasını kontrol et!")

            self.target_transform = transforms.Compose([
                transforms.Resize(input_size),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_file = self.image_files[index]

        # Görüntüyü yükle
        image = Image.open(image_file).convert("RGB")
        image = self.image_transform(image)

        if self.is_train:
            return image
        else:
            if os.path.dirname(image_file).endswith("good"):
                target = Image.new("L", (image.shape[2], image.shape[1]))
                target = self.target_transform(target)
            else:
                target_path = image_file.replace("/test/defect/", "/ground/defect/")
                target_path = os.path.splitext(target_path)[0] + ".jpg"

                if not os.path.exists(target_path):
                    # Eğer maske bulunamazsa boş bir maske döndür
                    print(f"❌ Maske bulunamadı, boş maske kullanılıyor: {target_path}")
                    target = Image.new("L", (image.shape[2], image.shape[1]))
                else:
                    target = Image.open(target_path).convert("L")

                target = self.target_transform(target)

            return image, target
