from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
from model import Discriminator, Projection, PatchMaker

import numpy as np
import torch.nn.functional as F

import logging
import os
import torch
import tqdm
import common
import metrics
import cv2
import utils
import glob
import shutil

LOGGER = logging.getLogger(__name__)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class TBWrapper:
    def __init__(self, log_dir):
        self.g_iter = 0
        self.logger = SummaryWriter(log_dir=log_dir)

    def step(self):
        self.g_iter += 1


class PBAS(torch.nn.Module):
    def __init__(self, device):
        super(PBAS, self).__init__()
        self.device = device

    def load(
            self,
            backbone,
            layers_to_extract_from,
            device,
            input_shape,
            pretrain_embed_dimension,
            target_embed_dimension,
            patchsize=3,
            patchstride=1,
            meta_epochs=640,
            eval_epochs=1,
            dsc_layers=2,
            dsc_hidden=1024,
            dsc_margin=0.5,
            train_backbone=False,
            pre_proj=1,
            k=0.25,
            lr=0.0001,
            **kwargs,
    ):
        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape
        self.device = device

        self.forward_modules = torch.nn.ModuleDict({})
        feature_aggregator = common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device, train_backbone
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        preprocessing = common.Preprocessing(feature_dimensions, pretrain_embed_dimension)
        self.forward_modules["preprocessing"] = preprocessing
        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = common.Aggregator(target_dim=target_embed_dimension)
        preadapt_aggregator.to(self.device)
        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.meta_epochs = meta_epochs
        self.lr = lr
        self.train_backbone = train_backbone
        if self.train_backbone:
            self.backbone_opt = torch.optim.AdamW(self.forward_modules["feature_aggregator"].backbone.parameters(), lr)

        self.pre_proj = pre_proj
        if self.pre_proj > 0:
            self.pre_projection = Projection(self.target_embed_dimension, self.target_embed_dimension, pre_proj)
            self.pre_projection.to(self.device)
            self.proj_opt = torch.optim.Adam(self.pre_projection.parameters(), lr, weight_decay=1e-5)

        self.dsc_lr = lr * 2
        self.eval_epochs = eval_epochs
        self.dsc_layers = dsc_layers
        self.dsc_hidden = dsc_hidden
        self.discriminator = Discriminator(self.target_embed_dimension, n_layers=dsc_layers, hidden=dsc_hidden)
        self.discriminator.to(self.device)
        self.dsc_opt = torch.optim.AdamW(self.discriminator.parameters(), lr=self.dsc_lr)
        self.dsc_margin = dsc_margin

        self.c = torch.tensor(0)
        self.k = k

        self.patch_maker = PatchMaker(patchsize, stride=patchstride)
        self.anomaly_segmentor = common.RescaleSegmentor(device=self.device, target_size=input_shape[-2:])
        self.model_dir = ""
        self.dataset_name = ""
        self.logger = None

    def set_model_dir(self, model_dir, dataset_name):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.ckpt_dir = os.path.join(self.model_dir, dataset_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.tb_dir = os.path.join(self.ckpt_dir, "tb")
        os.makedirs(self.tb_dir, exist_ok=True)
        self.logger = TBWrapper(self.tb_dir)

    def _embed(self, images, detach=True, provide_patch_shapes=False, evaluation=False):
        """Returns feature embeddings for images."""
        if not evaluation and self.train_backbone:
            self.forward_modules["feature_aggregator"].train()
            features = self.forward_modules["feature_aggregator"](images, eval=evaluation)
        else:
            self.forward_modules["feature_aggregator"].eval()
            with torch.no_grad():
                features = self.forward_modules["feature_aggregator"](images)

        features = [features[layer] for layer in self.layers_to_extract_from]

        features = [self.patch_maker.patchify(x, return_spatial_info=True) for x in features]
        patch_shapes = [x[1] for x in features]
        patch_features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(patch_features)):
            feature = patch_features[i]
            patch_dims = patch_shapes[i]

            feature = feature.reshape(
                feature.shape[0], patch_dims[0], patch_dims[1], *feature.shape[2:]
            )
            feature = feature.permute(0, 3, 4, 5, 1, 2)
            perm_base_shape = feature.shape
            feature = feature.reshape(-1, *feature.shape[-2:])
            feature = F.interpolate(
                feature.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            feature = feature.squeeze(1)
            feature = feature.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            feature = feature.permute(0, 4, 5, 1, 2, 3)
            feature = feature.reshape(len(feature), -1, *feature.shape[-3:])
            patch_features[i] = feature

        patch_features = [x.reshape(-1, *x.shape[-3:]) for x in patch_features]
        patch_features = self.forward_modules["preprocessing"](patch_features)
        patch_features = self.forward_modules["preadapt_aggregator"](patch_features)

        return patch_features, patch_shapes

    def trainer(self, train_data, test_data, name):
        state_dict = {}
        ckpt_path = glob.glob(self.ckpt_dir + '/ckpt_best*')
        if len(ckpt_path) != 0:
            LOGGER.info("Start testing, ckpt file found!")
            return 0., 0., 0., 0., 0., -1.

        def update_state_dict():
            state_dict["discriminator"] = OrderedDict({
                k: v.detach().cpu()
                for k, v in self.discriminator.state_dict().items()})
            if self.pre_proj > 0:
                state_dict["pre_projection"] = OrderedDict({
                    k: v.detach().cpu()
                    for k, v in self.pre_projection.state_dict().items()})

        with torch.no_grad():
            for i, data in enumerate(train_data):
                img = data["image"]
                img = img.to(torch.float).to(self.device)
                if self.pre_proj > 0:
                    outputs = self.pre_projection(self._embed(img, evaluation=False)[0])
                else:
                    outputs = self._embed(img, evaluation=False)[0]
                outputs = outputs.reshape(img.shape[0], -1, outputs.shape[-1])

                batch_mean = torch.mean(outputs, dim=0)
                if i == 0:
                    self.c = batch_mean
                else:
                    similarity = torch.mm(F.normalize(batch_mean, dim=1), F.normalize(self.c, dim=1).T)
                    indices = torch.argmax(similarity, dim=1)
                    for j, index in enumerate(indices):
                        self.c[index] = self.c[index] * 0.9 + batch_mean[j] * 0.1
            center_path = os.path.join(self.ckpt_dir, "center.pth")
            torch.save(self.c, center_path)

        pbar = tqdm.tqdm(range(self.meta_epochs), unit='epoch')
        pbar_str1 = ""
        best_record = None

        for i_epoch in pbar:
            pbar_str, pt, pg = self._train_discriminator(train_data, i_epoch, pbar, pbar_str1)
            update_state_dict()

            if (i_epoch + 1) % self.eval_epochs == 0:
                images, scores, segmentations, labels_gt, masks_gt = self.predict(test_data)
                image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro = self._evaluate(images, scores, segmentations,
                                                                                         labels_gt, masks_gt, name)
                self.logger.logger.add_scalar("i-auroc", image_auroc, i_epoch)
                self.logger.logger.add_scalar("p-auroc", pixel_auroc, i_epoch)

                eval_path = './results/eval/' + name + '/'
                train_path = './results/training/' + name + '/'
                if best_record is None or image_auroc + pixel_auroc > best_record[0] + best_record[2]:
                    if best_record is not None:
                        os.remove(ckpt_path_best)
                    best_record = [image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro, i_epoch]
                    ckpt_path_best = os.path.join(self.ckpt_dir, "ckpt_best_{}.pth".format(i_epoch))
                    torch.save(state_dict, ckpt_path_best)
                    shutil.rmtree(eval_path, ignore_errors=True)
                    shutil.copytree(train_path, eval_path)

                pbar_str1 = f" IAUC:{round(image_auroc * 100, 2)}({round(best_record[0] * 100, 2)})" \
                            f" PAUC:{round(pixel_auroc * 100, 2)}({round(best_record[2] * 100, 2)})" \
                            f" E:{i_epoch}({best_record[-1]})"
                pbar_str += pbar_str1
                pbar.set_description_str(pbar_str)

        return best_record

    def _train_discriminator(self, train_data, cur_epoch, pbar, pbar_str1):
        self.forward_modules.eval()
        if self.pre_proj > 0:
            self.pre_projection.train()
        self.discriminator.train()

        all_loss, all_s_loss, all_b_loss = [], [], []
        all_p_true, all_p_fake, all_r_t, all_r_f, all_r_g = [], [], [], [], []
        sample_num = 0
        for i_iter, data_item in enumerate(train_data):
            self.dsc_opt.zero_grad()
            if self.pre_proj > 0:
                self.proj_opt.zero_grad()

            img = data_item["image"]
            img = img.to(torch.float).to(self.device)
            if self.pre_proj > 0:
                true_feats = self.pre_projection(self._embed(img, evaluation=False)[0])
            else:
                true_feats = self._embed(img, evaluation=False)[0]

            similarity = torch.mm(F.normalize(true_feats, dim=1), F.normalize(self.c, dim=1).T)
            indices = torch.argmax(similarity, dim=1)
            center_feats = self.c[indices]

            direct = true_feats - center_feats
            d_ct = torch.norm(direct, dim=1, keepdim=True)
            r_ct = d_ct.mean()
            fake_feats = true_feats + ((direct / d_ct) * r_ct * self.k).detach()

            d_tf = torch.norm(fake_feats - true_feats, dim=1, keepdim=True)
            r_tf = d_tf.mean()
            d_cf = torch.norm(fake_feats - center_feats, dim=1, keepdim=True)
            r_cf = d_cf.mean()
            svdd_loss = r_ct

            scores = self.discriminator(torch.concat([true_feats, fake_feats]))
            true_scores = scores[:len(true_feats)]
            fake_scores = scores[len(true_feats):]
            true_loss = torch.nn.BCELoss()(true_scores, torch.zeros_like(true_scores))
            fake_loss = torch.nn.BCELoss()(fake_scores, torch.ones_like(fake_scores))
            bce_loss = true_loss + fake_loss

            loss = svdd_loss + bce_loss
            loss.backward()
            if self.pre_proj > 0:
                self.proj_opt.step()
            if self.train_backbone:
                self.backbone_opt.step()
            self.dsc_opt.step()

            pix_true = true_scores.detach()
            pix_fake = fake_scores.detach()
            p_t = (pix_true < self.dsc_margin).sum() / pix_true.shape[0]
            p_g = (pix_fake >= self.dsc_margin).sum() / pix_fake.shape[0]

            self.logger.logger.add_scalar("total_loss", loss, self.logger.g_iter)
            self.logger.logger.add_scalar("svdd_loss", svdd_loss, self.logger.g_iter)
            self.logger.logger.add_scalar("bce_loss", bce_loss, self.logger.g_iter)
            self.logger.logger.add_scalar(f"p_true", p_t, self.logger.g_iter)
            self.logger.logger.add_scalar(f"p_fake", p_g, self.logger.g_iter)
            self.logger.logger.add_scalar(f"r_true", r_ct, self.logger.g_iter)
            self.logger.logger.add_scalar(f"r_fake", r_cf, self.logger.g_iter)
            self.logger.step()

            all_loss.append(loss.detach().cpu().item())
            all_s_loss.append(svdd_loss.detach().cpu().item())
            all_b_loss.append(bce_loss.detach().cpu().item())
            all_p_true.append(p_t.cpu().item())
            all_p_fake.append(p_g.cpu().item())
            all_r_t.append(r_ct.cpu().item())
            all_r_g.append(r_cf.cpu().item())
            all_r_f.append(r_tf.cpu().item())

            np.mean(all_loss)
            all_s_loss_ = np.mean(all_s_loss)
            all_b_loss_ = np.mean(all_b_loss)
            all_p_true_ = np.mean(all_p_true)
            all_p_fake_ = np.mean(all_p_fake)
            all_r_t_ = np.mean(all_r_t)
            all_r_g_ = np.mean(all_r_g)
            all_r_f_ = np.mean(all_r_f)
            sample_num += img.shape[0]

            pbar_str = f"epoch:{cur_epoch}"
            pbar_str += f" sl:{all_s_loss_:.2e}"
            pbar_str += f" bl:{all_b_loss_:.2e}"
            pbar_str += f" pt:{all_p_true_ * 100:.2f}"
            pbar_str += f" pf:{all_p_fake_ * 100:.2f}"
            pbar_str += f" c->t:{all_r_t_:.2f}"
            pbar_str += f" c->f:{all_r_g_:.2f}"
            pbar_str += f" t->f:{all_r_f_:.2f}"
            pbar_str += f" sample:{sample_num}"
            pbar_str2 = pbar_str
            pbar_str += pbar_str1
            pbar.set_description_str(pbar_str)

        return pbar_str2, all_p_true_, all_p_fake_

    def tester(self, test_data, name):
        ckpt_path = glob.glob(self.ckpt_dir + '/ckpt_best*')
        if len(ckpt_path) != 0:
            state_dict = torch.load(ckpt_path[0], map_location=self.device)
            if 'discriminator' in state_dict:
                self.discriminator.load_state_dict(state_dict['discriminator'])
                if "pre_projection" in state_dict:
                    self.pre_projection.load_state_dict(state_dict["pre_projection"])
            else:
                self.load_state_dict(state_dict, strict=False)

            images, scores, segmentations, labels_gt, masks_gt = self.predict(test_data)
            image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro = self._evaluate(images, scores, segmentations,
                                                                                     labels_gt, masks_gt, name, path='eval')
            epoch = int(ckpt_path[0].split('_')[-1].split('.')[0])
        else:
            LOGGER.info("No ckpt file found!")
            return 0., 0., 0., 0., 0., -1.

        return image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro, epoch

    def predict(self, test_dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""
        self.forward_modules.eval()

        img_paths = []
        images = []
        scores = []
        masks = []
        labels_gt = []
        masks_gt = []

        with tqdm.tqdm(test_dataloader, desc="Inferring...", leave=False, unit='batch') as data_iterator:
            for data in data_iterator:
                if isinstance(data, dict):
                    labels_gt.extend(data["is_anomaly"].numpy().tolist())
                    if data.get("mask_gt", None) is not None:
                        masks_gt.extend(data["mask_gt"].numpy().tolist())
                    image = data["image"]
                    images.extend(image.numpy().tolist())
                    img_paths.extend(data["image_path"])
                _scores, _masks = self._predict(image)
                for score, mask in zip(_scores, _masks):
                    scores.append(score)
                    masks.append(mask)

        return images, scores, masks, labels_gt, masks_gt

    def _predict(self, img):
        """Infer score and mask for a batch of images."""
        img = img.to(torch.float).to(self.device)
        self.forward_modules.eval()

        if self.pre_proj > 0:
            self.pre_projection.eval()
        self.discriminator.eval()

        with torch.no_grad():
            patch_features, patch_shapes = self._embed(img, provide_patch_shapes=True, evaluation=True)
            if self.pre_proj > 0:
                patch_features = self.pre_projection(patch_features)
                patch_scores = image_scores = self.discriminator(patch_features)

            patch_scores = self.patch_maker.unpatch_scores(patch_scores, batchsize=img.shape[0])
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(img.shape[0], scales[0], scales[1])
            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)

            image_scores = self.patch_maker.unpatch_scores(image_scores, batchsize=img.shape[0])
            image_scores = self.patch_maker.score(image_scores)
            if isinstance(image_scores, torch.Tensor):
                image_scores = image_scores.cpu().numpy()

        return list(image_scores), list(masks)
    

    def _evaluate(self, images, scores, segmentations, labels_gt, masks_gt, name, path='training'):
    # Skorları ve sonuçları hesaplama
      scores = np.squeeze(np.array(scores))
      image_scores = metrics.compute_imagewise_retrieval_metrics(scores, labels_gt, path)
      image_auroc = image_scores["auroc"]
      image_ap = image_scores["ap"]
      
      # Ek olarak: threshold ile binary sınıflandırma yap ve F1 hesapla
      from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
      best_f1 = 0
      best_thresh = 0.0
      for t in np.linspace(min(scores), max(scores), 100):
          preds = [1 if s > t else 0 for s in scores]
          f1 = f1_score(labels_gt, preds)
          if f1 > best_f1:
              best_f1 = f1
              best_thresh = t
      final_preds = [1 if s > best_thresh else 0 for s in scores]
      precision = precision_score(labels_gt, final_preds)
      recall = recall_score(labels_gt, final_preds)
      cm = confusion_matrix(labels_gt, final_preds)

      print("🔍 Optimal threshold:", round(best_thresh, 4))
      print("✅ Image-level F1 Score:", round(best_f1, 4))
      print("📊 Precision:", round(precision, 4))
      print("📈 Recall:", round(recall, 4))
      print("🧮 Confusion Matrix:\n", cm)

      segmentations = np.array(segmentations)
      pixel_scores = metrics.compute_pixelwise_retrieval_metrics(segmentations, masks_gt, path)
      pixel_auroc = pixel_scores["auroc"]
      pixel_ap = pixel_scores["ap"]
      if path == 'eval':
        try:
            pixel_pro = metrics.compute_pro(np.squeeze(np.array(masks_gt)), segmentations)
        except:
            pixel_pro = 0.
      else:
        pixel_pro = 0.
      def compute_iou(pred_mask, gt_mask, threshold=0.5):
            pred_binary = (pred_mask > threshold).astype(np.uint8)
            if len(gt_mask.shape) == 3 and gt_mask.shape[2] == 3:
                gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2GRAY)
            gt_binary = (gt_mask > 0).astype(np.uint8)

            intersection = np.logical_and(pred_binary, gt_binary).sum()
            union = np.logical_or(pred_binary, gt_binary).sum()
            return intersection / union if union != 0 else 0.0

      # F1 skorlarını hesapla
      def compute_f1_score(pred_mask, gt_mask, threshold=0.5):
          """
          Prediction ve ground truth maskeleri arasındaki F1 skorunu hesapla
          """
          pred_binary = (pred_mask > threshold).astype(np.uint8)
          
          # Eğer gt_mask 3 kanallı ise, tek kanala dönüştür
          if len(gt_mask.shape) == 3 and gt_mask.shape[2] == 3:
              gt_mask_gray = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2GRAY)
          else:
              gt_mask_gray = gt_mask
              
          gt_binary = (gt_mask_gray > 0).astype(np.uint8)
          
          # Şekilleri kontrol et
          if pred_binary.shape != gt_binary.shape:
              raise ValueError(f"Shapes do not match: pred_binary.shape={pred_binary.shape}, gt_binary.shape={gt_binary.shape}")
          
          # True Positive, False Positive, False Negative sayılarını hesapla
          tp = np.sum(np.logical_and(pred_binary == 1, gt_binary == 1))
          fp = np.sum(np.logical_and(pred_binary == 1, gt_binary == 0))
          fn = np.sum(np.logical_and(pred_binary == 0, gt_binary == 1))
          
          # Precision ve recall hesapla
          precision = tp / (tp + fp) if (tp + fp) > 0 else 0
          recall = tp / (tp + fn) if (tp + fn) > 0 else 0
          
          # F1 skoru hesapla
          f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
          
          return f1, precision, recall

    # Görselleştirme ve anomali bölgelerinin işaretlenmesi
      defects = np.array(images)
      targets = np.array(masks_gt)
    
    # Sonuçların kaydedileceği dizini oluştur
      full_path = './results/' + path + '/' + name + '/'
      utils.del_remake_dir(full_path, del_flag=False)
    
    # Her bir görüntü için anomali tespiti ve görselleştirme
      for i in range(len(defects)):
        # Görüntüleri doğru formata dönüştür
        defect = utils.torch_format_2_numpy_img(defects[i])
        target = utils.torch_format_2_numpy_img(targets[i])

        # Isı haritasını görüntü boyutuna yeniden boyutlandır
        heatmap = cv2.resize(segmentations[i], (defect.shape[1], defect.shape[0]))
        iou_score = compute_iou(heatmap, target)

        # Tüm görüntü için F1 skoru hesapla
        try:
            f1, precision, recall = compute_f1_score(heatmap, target)
        except Exception as e:
            print(f"Error computing overall F1 score for image {i+1}: {str(e)}")
            f1, precision, recall = 0.0, 0.0, 0.0
        
        # Isı haritasını renklendirme - JET renk haritası
        heatmap_vis = cv2.cvtColor((heatmap * 255).astype('uint8'), cv2.COLOR_GRAY2BGR)
        colored_heatmap = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)
        
        # Yüksek eşik değeri ile anomali bölgelerini tespit et (0.8 olarak ayarlanmış)
        high_threshold = 0.80  # Bu değeri 0 ile 1 arasında ihtiyaca göre ayarlayabilirsiniz
        binary_mask = (heatmap > high_threshold).astype(np.uint8) * 255
        
        # Gürültü azaltma için morfolojik işlemler (opsiyonel)
        kernel = np.ones((3, 3), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        
        # Anomali konturlarını bul
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Orijinal görüntü üzerine konturları çiz
        defect_with_contours = defect.copy()
        # Prediction ve Ground Truth yazısı ekle
        true_label = labels_gt[i]
        pred_label = final_preds[i]

        gt_text = "Ground Truth: Anomaly" if true_label == 1 else "Ground Truth: Normal"
        pred_text = "Prediction: Anomaly" if pred_label == 1 else "Prediction: Normal"

        # Doğruysa yeşil, yanlışsa kırmızı
        color = (0, 255, 0) if true_label == pred_label else (0, 0, 255)

        # Yazıları ayrı ayrı çiz (çakışmasın diye satır satır)
        cv2.putText(
            defect_with_contours,
            pred_text,
            (10, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1
        )

        cv2.putText(
            defect_with_contours,
            gt_text,
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1
        )


        
        # Konturları çiz ve kontur içine anomali skorunu yaz
        for j, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < 100:  # Çok küçük anomalileri filtrele
                continue
                
            # Konturun orijinal görüntü üzerine çizimi
            cv2.drawContours(defect_with_contours, [contour], -1, (0, 255, 0), 2)  # Yeşil konturlar
            
            # Anomali bölgesinin ortalama skorunu hesapla
            mask = np.zeros_like(heatmap)
            cv2.drawContours(mask, [contour], 0, 1, -1)
            mean_score = np.mean(heatmap[mask == 1])
            
            # Konturun ağırlık merkezini hesapla
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0
                
            # Anomali skoru yazısı için arka plan dikdörtgeni
            text = f"{mean_score:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Anomali skorunu konturun ağırlık merkezine yaz
            cv2.putText(
                defect_with_contours, 
                text, 
                (cX - text_width // 2, cY + text_height // 2), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 0, 255),  # Kırmızı renk
                1
            )
        
        # Görüntü üzerine daireler (circles) çizerek anomalileri işaretle
        defect_with_circles = defect.copy()
        
        # Anomali özelliklerini kaydetmek için bir metin dosyası oluştur
        with open(full_path + str(i + 1).zfill(3) + '_anomalies.txt', 'w') as f:
            f.write(f"Image: {str(i + 1).zfill(3)}\n")
            f.write(f"Threshold used: {high_threshold}\n")
            f.write(f"Total anomalies detected: {len(contours)}\n")
            f.write(f"Overall F1 Score: {f1:.4f}\n")
            f.write(f"Overall Precision: {precision:.4f}\n")
            f.write(f"Overall Recall: {recall:.4f}\n\n")
        
        # Her bir anomali için özelliklerini hesapla ve görselleştir
        contour_data = []  # Kontur bilgilerini saklamak için liste
        
        for j, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # Çok küçük anomalileri filtrele (alanı 100 pikselden küçük olanları atla)
            if area < 100:  
                continue
                
            # Daire çizmek için anomalinin minimum çevreleyen daireyi bul
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            
            # Anomali bölgesinin ortalama skoru hesapla
            mask = np.zeros_like(heatmap)
            cv2.drawContours(mask, [contour], 0, 1, -1)
            mean_score = np.mean(heatmap[mask == 1])
            
            # Bu kontur bölgesi için F1 skoru hesapla
            contour_mask = np.zeros_like(heatmap)
            cv2.drawContours(contour_mask, [contour], 0, 1, -1)
            
            # Bu kontur bölgesi için F1 skoru hesapla
            try:
                contour_f1, contour_prec, contour_rec = compute_f1_score(contour_mask, target)
            except Exception as e:
                print(f"Error computing F1 score for contour {j+1}: {str(e)}")
                contour_f1, contour_prec, contour_rec = 0.0, 0.0, 0.0
            
            # Kontur verilerini sakla
            if radius > 5:
                contour_data.append({
                    'center': center,
                    'radius': radius,
                    'mean_score': mean_score,
                    'id': j+1,
                    'area': area,
                    'f1': contour_f1,
                    'precision': contour_prec,
                    'recall': contour_rec
                })
            
            # Anomali bilgilerini dosyaya kaydet
            with open(full_path + str(i + 1).zfill(3) + '_anomalies.txt', 'a') as f:
                f.write(f"Anomaly #{j+1}:\n")
                f.write(f"  Center: ({int(x)}, {int(y)})\n")
                f.write(f"  Radius: {radius} pixels\n")
                f.write(f"  Area: {area:.1f} pixels²\n")
                f.write(f"  Mean score: {mean_score:.4f}\n")
                f.write(f"  F1 Score: {contour_f1:.4f}\n")
                f.write(f"  Precision: {contour_prec:.4f}\n")
                f.write(f"  Recall: {contour_rec:.4f}\n\n")
            
        # Yarı-saydam ısı haritası ile orijinal görüntüyü birleştir
        overlay = defect.copy()
        alpha = 0.6  # Saydamlık faktörü
        cv2.addWeighted(colored_heatmap, alpha, defect, 1-alpha, 0, overlay)
        
        # Kompozit görüntü oluştur (tüm görselleştirmeleri tek bir görüntüde birleştir)
        h, w = defect.shape[:2]
        composite = np.zeros((h, w*4, 3), dtype=np.uint8)
        
        # İlk üç bölüm: orijinal, heatmap ve konturlar
        composite[:, 0:w] = defect
        composite[:, w:w*2] = colored_heatmap
        composite[:, w*2:w*3] = defect_with_contours
        
        # Dördüncü bölüm: daireler
        # Orijinal görüntüyü kopyala
        defect_with_circles = defect.copy()
        
        # Daireleri çiz
        for data in contour_data:
            cv2.circle(defect_with_circles, data['center'], data['radius'], (0, 0, 255), 2)
            # Sadece anomaly skoru yazdır
            cv2.putText(
                defect_with_circles, 
                f"{data['mean_score']:.2f}", 
                (data['center'][0] - 10, data['center'][1] + 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 0, 255), 
                1
            )
        
        # Daire görüntüsünü kompozitin dördüncü bölümüne yerleştir
        composite[:, w*3:w*4] = defect_with_circles
        
        # Kompozit görüntüye açıklayıcı etiketler ekle
        labels = ["Original", "Heatmap", "Contours", "Circles"]
        for idx, label in enumerate(labels):
            cv2.putText(
                composite, 
                label, 
                (w*idx + 10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, 
                (255, 255, 255), 
                2
            )
        
        
        
        # Tüm görselleştirmeleri kaydet
        cv2.imwrite(full_path + str(i + 1).zfill(3) + '_input.png', defect)
        cv2.imwrite(full_path + str(i + 1).zfill(3) + '_gt.png', target)
        cv2.imwrite(full_path + str(i + 1).zfill(3) + '_heatmap.png', colored_heatmap)
        cv2.imwrite(full_path + str(i + 1).zfill(3) + '_contours.png', defect_with_contours)
        cv2.imwrite(full_path + str(i + 1).zfill(3) + '_circles.png', defect_with_circles)
        cv2.imwrite(full_path + str(i + 1).zfill(3) + '_overlay.png', overlay)
        cv2.imwrite(full_path + str(i + 1).zfill(3) + '_composite.png', composite)
        cv2.imwrite(full_path + str(i + 1).zfill(3) + '_binary_mask.png', binary_mask)

      return image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro
