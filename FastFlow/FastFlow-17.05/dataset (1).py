import os
from glob import glob

import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms


class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, category, input_size, is_train=True):
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.target_transform = transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.ToTensor(),
            ]
        )
        self.is_train = is_train

        if is_train:
            self.image_files = glob(
                os.path.join(root, category, "train", "good", "*.*")
            )
            print(f"[TRAIN] Eğitim verisi bulundu: {len(self.image_files)} dosya")
        else:
            self.image_files = glob(os.path.join(root, category, "test", "*", "*.*"))
            print(f"[TEST] Test verisi bulundu: {len(self.image_files)} dosya")
            if len(self.image_files) > 0:
                print(f"[TEST] Örnek dosya: {self.image_files[0]}")

    def __getitem__(self, index):
        image_file = self.image_files[index]

        # Görseli oku ve 3 kanala dönüştür
        image = Image.open(image_file).convert("RGB")
        image = self.image_transform(image)

        if self.is_train:
            return image
        else:
            # GOOD örneklerde maske yok: tüm sıfırlardan oluşan maske
            if os.path.dirname(image_file).endswith("good"):
                target = torch.zeros([1, image.shape[-2], image.shape[-1]])
            else:
                # Maske dosya yolunu üret
                mask_path = (
                    image_file.replace("/test/", "/ground/")
                    .replace(".jpg", ".jpg")
                    .replace(".jpeg", ".jpg")
                    .replace(".png", ".jpg")
                )
                if not os.path.exists(mask_path):
                    print(f"❌ Maske dosyası bulunamadı: {mask_path}")
                    target = torch.zeros([1, image.shape[-2], image.shape[-1]])
                else:
                    mask = Image.open(mask_path).convert("L")
                    target = self.target_transform(mask)

            return image, target

    def __len__(self):
        return len(self.image_files)
