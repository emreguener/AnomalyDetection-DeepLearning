#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import tifffile
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import itertools
import os
import random
from tqdm import tqdm
from common import get_autoencoder, get_pdn_small, get_pdn_medium, \
    ImageFolderWithoutTarget, ImageFolderWithPath, InfiniteDataloader
from sklearn.metrics import roc_auc_score

best_iou_threshold = 0.3

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='mvtec_ad',
                        choices=['mvtec_ad', 'mvtec_loco'])
    parser.add_argument('-s', '--subdataset', default='bottle',
                        help='One of 15 sub-datasets of Mvtec AD or 5' +
                             'sub-datasets of Mvtec LOCO')
    parser.add_argument('-o', '--output_dir', default='output/1')
    parser.add_argument('-m', '--model_size', default='small',
                        choices=['small', 'medium'])
    parser.add_argument('-w', '--weights', default='/content/AnomalyDetection-DeepLearning/EfficientAD/models/teacher_small.pth')
    parser.add_argument('-i', '--imagenet_train_path',
                        default='none',
                        help='Set to "none" to disable ImageNet' +
                             'pretraining penalty. Or see README.md to' +
                             'download ImageNet and set to ImageNet path')
    parser.add_argument('-a', '--mvtec_ad_path',
                        default='./mvtec_anomaly_detection',
                        help='Downloaded Mvtec AD dataset')
    parser.add_argument('-b', '--mvtec_loco_path',
                        default='./mvtec_loco_anomaly_detection',
                        help='Downloaded Mvtec LOCO dataset')
    parser.add_argument('-t', '--train_steps', type=int, default=70000)
    return parser.parse_args()

# constants
seed = 42
on_gpu = torch.cuda.is_available()
out_channels = 384
image_size = 256

# data loading
default_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
transform_ae = transforms.RandomChoice([
    transforms.ColorJitter(brightness=0.2),
    transforms.ColorJitter(contrast=0.2),
    transforms.ColorJitter(saturation=0.2)
])

def train_transform(image):
    return default_transform(image), default_transform(transform_ae(image))

def main():
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    config = get_argparse()

    train_losses = []
    val_losses = []

    if config.dataset == 'mvtec_ad':
        dataset_path = config.mvtec_ad_path
    elif config.dataset == 'mvtec_loco':
        dataset_path = config.mvtec_loco_path
    else:
        raise Exception('Unknown config.dataset')

    pretrain_penalty = config.imagenet_train_path != 'none'

    train_output_dir = os.path.join(config.output_dir, 'trainings', config.dataset, config.subdataset)
    test_output_dir = os.path.join(config.output_dir, 'anomaly_maps', config.dataset, config.subdataset, 'test')
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)

    full_train_set = ImageFolderWithoutTarget(
        os.path.join(dataset_path, config.subdataset, 'train'),
        transform=transforms.Lambda(train_transform))
    test_set = ImageFolderWithPath(
        os.path.join(dataset_path, config.subdataset, 'test'))

    if config.dataset == 'mvtec_ad':
        train_size = int(0.9 * len(full_train_set))
        validation_size = len(full_train_set) - train_size
        rng = torch.Generator().manual_seed(seed)
        train_set, validation_set = torch.utils.data.random_split(full_train_set, [train_size, validation_size], rng)
    else:
        train_set = full_train_set
        validation_set = ImageFolderWithoutTarget(
            os.path.join(dataset_path, config.subdataset, 'validation'),
            transform=transforms.Lambda(train_transform))

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    train_loader_infinite = InfiniteDataloader(train_loader)
    validation_loader = DataLoader(validation_set, batch_size=1)

    if pretrain_penalty:
        penalty_transform = transforms.Compose([
            transforms.Resize((2 * image_size, 2 * image_size)),
            transforms.RandomGrayscale(0.3),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        penalty_set = ImageFolderWithoutTarget(config.imagenet_train_path, transform=penalty_transform)
        penalty_loader = DataLoader(penalty_set, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
        penalty_loader_infinite = InfiniteDataloader(penalty_loader)
    else:
        penalty_loader_infinite = itertools.repeat(None)

    # Model oluşturma
    if config.model_size == 'small':
        teacher = get_pdn_small(out_channels)
        student = get_pdn_small(2 * out_channels)
    else:
        teacher = get_pdn_medium(out_channels)
        student = get_pdn_medium(2 * out_channels)

    state_dict = torch.load(config.weights, map_location='cpu')
    teacher.load_state_dict(state_dict)
    autoencoder = get_autoencoder(out_channels)

    teacher.eval()
    student.train()
    autoencoder.train()

    if on_gpu:
        teacher.cuda()
        student.cuda()
        autoencoder.cuda()

    teacher_mean, teacher_std = teacher_normalization(teacher, train_loader)
    
    # Save teacher normalization parameters
    normalization_params = {
        'teacher_mean': teacher_mean.cpu(),
        'teacher_std': teacher_std.cpu()
    }
    torch.save(normalization_params, os.path.join(train_output_dir, 'normalization_params.pth'))

    optimizer = torch.optim.Adam(itertools.chain(student.parameters(), autoencoder.parameters()),
                                 lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.95 * config.train_steps), gamma=0.1)

    tqdm_obj = tqdm(range(config.train_steps))
    for iteration in tqdm_obj:
        (image_st, image_ae) = next(train_loader_infinite)
        image_penalty = next(penalty_loader_infinite)

        if on_gpu:
            image_st = image_st.cuda()
            image_ae = image_ae.cuda()
            if image_penalty is not None:
                image_penalty = image_penalty.cuda()

        with torch.no_grad():
            teacher_output_st = teacher(image_st)
            teacher_output_st = (teacher_output_st - teacher_mean) / teacher_std

        student_output_st = student(image_st)[:, :out_channels]
        distance_st = (teacher_output_st - student_output_st) ** 2
        d_hard = torch.quantile(distance_st, q=0.999)
        loss_hard = torch.mean(distance_st[distance_st >= d_hard])

        if image_penalty is not None:
            student_output_penalty = student(image_penalty)[:, :out_channels]
            loss_penalty = torch.mean(student_output_penalty ** 2)
            loss_st = loss_hard + loss_penalty
        else:
            loss_st = loss_hard

        ae_output = autoencoder(image_ae)
        with torch.no_grad():
            teacher_output_ae = teacher(image_ae)
            teacher_output_ae = (teacher_output_ae - teacher_mean) / teacher_std

        student_output_ae = student(image_ae)[:, out_channels:]
        distance_ae = (teacher_output_ae - ae_output) ** 2
        distance_stae = (ae_output - student_output_ae) ** 2
        loss_ae = torch.mean(distance_ae)
        loss_stae = torch.mean(distance_stae)
        loss_total = loss_st + loss_ae + loss_stae

        train_losses.append(loss_total.item())

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        scheduler.step()

        if iteration % 10 == 0:
            tqdm_obj.set_description(f"Current loss: {loss_total.item():.4f}")

        if iteration % 1000 == 0:
            torch.save(teacher, os.path.join(train_output_dir, 'teacher_tmp.pth'))
            torch.save(student, os.path.join(train_output_dir, 'student_tmp.pth'))
            torch.save(autoencoder, os.path.join(train_output_dir, 'autoencoder_tmp.pth'))

        if iteration % 10000 == 0 and iteration > 0:
            # Validation loss hesaplama
            val_total_loss = 0
            val_batches = 0
            for val_image, _ in validation_loader:
                if on_gpu:
                    val_image = val_image.cuda()
                with torch.no_grad():
                    teacher_output_val = teacher(val_image)
                    teacher_output_val = (teacher_output_val - teacher_mean) / teacher_std
                    student_output_val = student(val_image)[:, :out_channels]
                    ae_output_val = autoencoder(val_image)
                    distance_val = (teacher_output_val - student_output_val) ** 2
                    d_val = torch.quantile(distance_val, q=0.999)
                    loss_val_hard = torch.mean(distance_val[distance_val >= d_val])

                    teacher_output_ae_val = teacher(val_image)
                    teacher_output_ae_val = (teacher_output_ae_val - teacher_mean) / teacher_std
                    student_output_ae_val = student(val_image)[:, out_channels:]
                    distance_ae_val = (teacher_output_ae_val - ae_output_val)**2
                    distance_stae_val = (ae_output_val - student_output_ae_val)**2
                    loss_val_ae = torch.mean(distance_ae_val)
                    loss_val_stae = torch.mean(distance_stae_val)

                    loss_val_total = loss_val_hard + loss_val_ae + loss_val_stae
                    val_total_loss += loss_val_total.item()
                    val_batches += 1
            val_losses.append(val_total_loss / val_batches)

            # Ara değerlendirme
            teacher.eval()
            student.eval()
            autoencoder.eval()

            q_st_start, q_st_end, q_ae_start, q_ae_end, map_quantiles = map_normalization(
                validation_loader=validation_loader, teacher=teacher,
                student=student, autoencoder=autoencoder,
                teacher_mean=teacher_mean, teacher_std=teacher_std,
                desc='Intermediate map normalization')
                
            # Save the quantiles at each checkpoint
            torch.save(map_quantiles, os.path.join(train_output_dir, f'map_quantiles_{iteration}.pth'))
                
            auc, f1, precision, recall, cm = test(
                test_set=test_set, teacher=teacher, student=student,
                autoencoder=autoencoder, teacher_mean=teacher_mean,
                teacher_std=teacher_std, q_st_start=q_st_start,
                q_st_end=q_st_end, q_ae_start=q_ae_start, q_ae_end=q_ae_end,
                test_output_dir=None, desc='Intermediate inference')
            print(f"\n📍 Ara Değerlendirme (Adım: {iteration})")
            print(f"AUC Score        : {auc:.4f}")
            print(f"F1 Score         : {f1:.4f}")
            print(f"Precision        : {precision:.4f}")
            print(f"Recall           : {recall:.4f}")
            print("Confusion Matrix :\n", cm)

            teacher.train()
            student.train()
            autoencoder.train()

    # Eğitim sonrası
    teacher.eval()
    student.eval()
    autoencoder.eval()

    torch.save(teacher, os.path.join(train_output_dir, 'teacher_final.pth'))
    torch.save(student, os.path.join(train_output_dir, 'student_final.pth'))
    torch.save(autoencoder, os.path.join(train_output_dir, 'autoencoder_final.pth'))

    q_st_start, q_st_end, q_ae_start, q_ae_end, map_quantiles = map_normalization(
        validation_loader=validation_loader, teacher=teacher, student=student,
        autoencoder=autoencoder, teacher_mean=teacher_mean,
        teacher_std=teacher_std, desc='Final map normalization')
        
    # Save the final quantiles
    torch.save(map_quantiles, os.path.join(train_output_dir, 'map_quantiles_final.pth'))
    
    auc, f1, precision, recall, cm = test(
        test_set=test_set, teacher=teacher, student=student,
        autoencoder=autoencoder, teacher_mean=teacher_mean,
        teacher_std=teacher_std, q_st_start=q_st_start, q_st_end=q_st_end,
        q_ae_start=q_ae_start, q_ae_end=q_ae_end,
        test_output_dir=test_output_dir, desc='Final inference')
    print("\n✅ Final Anomaly Detection Sonuçları")
    print(f"Final AUC Score        : {auc:.4f}")
    print(f"Final F1 Score         : {f1:.4f}")
    print(f"Final Precision        : {precision:.4f}")
    print(f"Final Recall           : {recall:.4f}")
    print("Final Confusion Matrix :\n", cm)

    # Grafik çizimi
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', alpha=0.7)
    plt.plot(np.linspace(0, len(train_losses), len(val_losses)), val_losses, label='Validation Loss', color='orange', marker='o')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(train_output_dir, 'loss_curve.png'))
    plt.close()


def load_model_with_quantiles(model_dir):
    """
    Load models and quantiles from a directory
    """
    teacher = torch.load(os.path.join(model_dir, 'teacher_final.pth'))
    student = torch.load(os.path.join(model_dir, 'student_final.pth'))
    autoencoder = torch.load(os.path.join(model_dir, 'autoencoder_final.pth'))
    map_quantiles = torch.load(os.path.join(model_dir, 'map_quantiles_final.pth'))
    normalization_params = torch.load(os.path.join(model_dir, 'normalization_params.pth'))
    
    return (teacher, student, autoencoder, 
            normalization_params['teacher_mean'], normalization_params['teacher_std'],
            map_quantiles['q_st_start'], map_quantiles['q_st_end'], 
            map_quantiles['q_ae_start'], map_quantiles['q_ae_end'])


def test(test_set, teacher, student, autoencoder, teacher_mean, teacher_std,
         q_st_start, q_st_end, q_ae_start, q_ae_end, test_output_dir=None,
         desc='Running inference'):
    import cv2
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, roc_curve, confusion_matrix

    y_true = []
    y_score = []

    threshold_candidates = np.linspace(0.1, 0.9, 9)
    threshold_iou_results = {}

    ground_truths = []
    scores = []

    for image, target, path in tqdm(test_set, desc=desc):
        orig_width = image.width
        orig_height = image.height
        image = default_transform(image)
        image = image[None]
        if torch.cuda.is_available():
            image = image.cuda()

        map_combined, map_st, map_ae = predict(
            image=image, teacher=teacher, student=student,
            autoencoder=autoencoder, teacher_mean=teacher_mean,
            teacher_std=teacher_std, q_st_start=q_st_start, q_st_end=q_st_end,
            q_ae_start=q_ae_start, q_ae_end=q_ae_end)

        map_combined = torch.nn.functional.pad(map_combined, (4, 4, 4, 4))
        map_combined = torch.nn.functional.interpolate(
            map_combined, (orig_height, orig_width), mode='bilinear')
        map_combined = map_combined[0, 0].cpu().numpy()

        defect_class = os.path.basename(os.path.dirname(path))
        if test_output_dir is not None:
            img_nm = os.path.split(path)[1].split('.')[0]
            os.makedirs(os.path.join(test_output_dir, defect_class), exist_ok=True)
            tifffile.imwrite(os.path.join(test_output_dir, defect_class, img_nm + '.tiff'), map_combined)

        y_true_image = 0 if defect_class == 'good' else 1
        y_score_image = np.max(map_combined)
        y_true.append(y_true_image)
        y_score.append(y_score_image)

        # IoU için sadece defect örnekleri
        if y_true_image == 1:
            gt_mask_path = path.replace('/test/', '/ground_truth/').replace('.png', '_mask.jpg').replace('.jpg', '_mask.jpg')
            if not os.path.exists(gt_mask_path):
                print(f"🚫 Mask file not found: {gt_mask_path}")
                continue

            gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
            if gt_mask is None:
                print(f"🚫 Failed to load mask: {gt_mask_path}")
                continue

            gt_mask_bin = (gt_mask > 127).astype(np.uint8)
            if np.sum(gt_mask_bin) == 0:
                print(f"⚠️ Mask is empty (all 0s): {gt_mask_path}")
                continue

            gt_mask_resized = cv2.resize(gt_mask_bin, (map_combined.shape[1], map_combined.shape[0]))
            ground_truths.append(gt_mask_resized.flatten())
            scores.append(map_combined.flatten())

    # Eğer hiç geçerli maske yoksa uyarı ver ve erken çık
    if len(ground_truths) == 0 or len(scores) == 0:
        print("\n❌ IoU hesaplanamadı: Hiçbir geçerli ground-truth maske bulunamadı.")
        print("Lütfen ground_truth klasöründe doğru adlandırılmış, gri tonlamalı maskeler olduğundan emin olun.")
    else:
        best_iou = -1
        best_thresh = 0.0

        for thresh in threshold_candidates:
            ious = []
            for gt_mask, score_map in zip(ground_truths, scores):
                pred_mask = (score_map >= thresh).astype(np.uint8)
                intersection = np.logical_and(gt_mask, pred_mask).sum()
                union = np.logical_or(gt_mask, pred_mask).sum()
                iou = intersection / union if union > 0 else (1.0 if intersection == 0 else 0.0)
                ious.append(iou)
            avg_iou = np.mean(ious)
            threshold_iou_results[thresh] = avg_iou
            if avg_iou > best_iou:
                best_iou = avg_iou
                best_thresh = thresh

        print("\n🔍 IoU Threshold Optimization Results:")
        for thresh, iou in threshold_iou_results.items():
            print(f"Threshold: {thresh:.2f} | Mean IoU: {iou:.4f}")
        print(f"\n✅ Best IoU: {best_iou:.4f} @ Threshold = {best_thresh:.2f}")

    # Anomaly tespiti metrikleri (image-level)
    auc = roc_auc_score(y_true=y_true, y_score=y_score)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    youden_index = tpr - fpr
    best_threshold = thresholds[np.argmax(youden_index)]
    y_pred = [1 if s >= best_threshold else 0 for s in y_score]
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print("\n✅ Anomaly Detection Sonucları")
    print(f"AUC Score          : {auc:.4f}")
    print(f"F1 Score           : {f1:.4f}")
    print(f"Precision          : {precision:.4f}")
    print(f"Recall             : {recall:.4f}")
    print(f"Optimal Threshold  : {best_threshold:.4f}")
    print("Confusion Matrix:\n", cm)

    return auc, f1, precision, recall, cm




@torch.no_grad()
def predict(image, teacher, student, autoencoder, teacher_mean, teacher_std,
            q_st_start=None, q_st_end=None, q_ae_start=None, q_ae_end=None):
    teacher_output = teacher(image)
    teacher_output = (teacher_output - teacher_mean) / teacher_std
    student_output = student(image)
    autoencoder_output = autoencoder(image)
    map_st = torch.mean((teacher_output - student_output[:, :out_channels])**2,
                        dim=1, keepdim=True)
    map_ae = torch.mean((autoencoder_output -
                         student_output[:, out_channels:])**2,
                        dim=1, keepdim=True)
    if q_st_start is not None:
        map_st = 0.1 * (map_st - q_st_start) / (q_st_end - q_st_start)
    if q_ae_start is not None:
        map_ae = 0.1 * (map_ae - q_ae_start) / (q_ae_end - q_ae_start)
    map_combined = 0.5 * map_st + 0.5 * map_ae
    return map_combined, map_st, map_ae

@torch.no_grad()
def map_normalization(validation_loader, teacher, student, autoencoder,
                      teacher_mean, teacher_std, desc='Map normalization'):
    maps_st = []
    maps_ae = []
    # ignore augmented ae image
    for image, _ in tqdm(validation_loader, desc=desc):
        if on_gpu:
            image = image.cuda()
        map_combined, map_st, map_ae = predict(
            image=image, teacher=teacher, student=student,
            autoencoder=autoencoder, teacher_mean=teacher_mean,
            teacher_std=teacher_std)
        maps_st.append(map_st)
        maps_ae.append(map_ae)
    maps_st = torch.cat(maps_st)
    maps_ae = torch.cat(maps_ae)
    q_st_start = torch.quantile(maps_st, q=0.9)
    q_st_end = torch.quantile(maps_st, q=0.995)
    q_ae_start = torch.quantile(maps_ae, q=0.9)
    q_ae_end = torch.quantile(maps_ae, q=0.995)
    
    # Create a dictionary to store the quantiles
    map_quantiles = {
        'q_st_start': q_st_start.cpu().item(),
        'q_st_end': q_st_end.cpu().item(),
        'q_ae_start': q_ae_start.cpu().item(),
        'q_ae_end': q_ae_end.cpu().item()
    }
    
    return q_st_start, q_st_end, q_ae_start, q_ae_end, map_quantiles

@torch.no_grad()
def teacher_normalization(teacher, train_loader):
    mean_outputs = []
    for train_image, _ in tqdm(train_loader, desc='Computing mean of features'):
        if on_gpu:
            train_image = train_image.cuda()
        teacher_output = teacher(train_image)
        mean_output = torch.mean(teacher_output, dim=[0, 2, 3])
        mean_outputs.append(mean_output)
    channel_mean = torch.mean(torch.stack(mean_outputs), dim=0)
    channel_mean = channel_mean[None, :, None, None]

    mean_distances = []
    for train_image, _ in tqdm(train_loader, desc='Computing std of features'):
        if on_gpu:
            train_image = train_image.cuda()
        teacher_output = teacher(train_image)
        distance = (teacher_output - channel_mean) ** 2
        mean_distance = torch.mean(distance, dim=[0, 2, 3])
        mean_distances.append(mean_distance)
    channel_var = torch.mean(torch.stack(mean_distances), dim=0)
    channel_var = channel_var[None, :, None, None]
    channel_std = torch.sqrt(channel_var)

    return channel_mean, channel_std

if __name__ == '__main__':
    main()
