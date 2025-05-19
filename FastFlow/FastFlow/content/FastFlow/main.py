import argparse
import os

import torch
import yaml
from ignite.contrib import metrics
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, jaccard_score
import numpy as np

import constants as const
import dataset
import fastflow
import utils

# === Helper Functions ===
def find_optimal_threshold(scores, labels):
    scores = np.array(scores)
    labels = np.array(labels).astype(int)
    if len(np.unique(labels)) < 2:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_scores = (2 * precision * recall) / (precision + recall + 1e-10)
    optimal_idx = np.argmax(f1_scores[:-1]) if len(f1_scores) > 1 else 0
    optimal_threshold = thresholds[optimal_idx] if len(thresholds) > 0 else 0.5
    return optimal_threshold

def calculate_f1(pred_mask, gt_mask):
    pred_flat = pred_mask.astype(bool).flatten()
    gt_flat = gt_mask.astype(bool).flatten()
    try:
        return f1_score(gt_flat, pred_flat, zero_division=1)
    except Exception as e:
        print(f"Error in F1 calculation: {e}")
        return 0.0

def calculate_iou(pred_mask, gt_mask):
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 1.0
    return intersection / union

# === Main Training / Eval Functions ===
def build_train_data_loader(args, config):
    train_dataset = dataset.MVTecDataset(
        root=args.data,
        category=args.category,
        input_size=config["input_size"],
        is_train=True,
    )
    return torch.utils.data.DataLoader(
        train_dataset,
        batch_size=const.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

def build_test_data_loader(args, config):
    test_dataset = dataset.MVTecDataset(
        root=args.data,
        category=args.category,
        input_size=config["input_size"],
        is_train=False,
    )
    return torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )

def build_model(config):
    model = fastflow.FastFlow(
        backbone_name=config["backbone_name"],
        flow_steps=config["flow_step"],
        input_size=config["input_size"],
        conv3x3_only=config["conv3x3_only"],
        hidden_ratio=config["hidden_ratio"],
    )
    print("Model A.D. Param#: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    return model

def build_optimizer(model):
    return torch.optim.Adam(
        model.parameters(), lr=const.LR, weight_decay=const.WEIGHT_DECAY
    )

def eval_once(dataloader, model):
    model.eval()
    auroc_metric = metrics.ROC_AUC()
    all_scores, all_labels = [], []
    px_f1_total, px_iou_total = 0, 0
    count = 0

    with torch.no_grad():
        for data, targets in dataloader:
            data, targets = data.cuda(), targets.cuda()
            ret = model(data)
            anomaly_map = ret["anomaly_map"].cpu()
            outputs = anomaly_map.flatten()
            gt = targets.flatten()
            auroc_metric.update((outputs, gt))

            scores = anomaly_map.view(anomaly_map.shape[0], -1).max(dim=1)[0]
            labels = (targets.view(targets.shape[0], -1).max(dim=1).values > 0).float()
            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Pixel-level F1/IoU
            for i in range(anomaly_map.size(0)):
                pred_mask = (anomaly_map[i] > 0.5).numpy()
                gt_mask = targets[i].cpu().numpy()
                px_f1_total += calculate_f1(pred_mask, gt_mask)
                px_iou_total += calculate_iou(pred_mask, gt_mask)
                count += 1

    auroc = auroc_metric.compute()
    best_thresh = find_optimal_threshold(all_scores, all_labels)
    pred_labels = (np.array(all_scores) > best_thresh).astype(int)
    f1 = f1_score(all_labels, pred_labels)
    cm = confusion_matrix(all_labels, pred_labels)
    px_f1_avg = px_f1_total / count
    px_iou_avg = px_iou_total / count

    print(f"\n📈 AUROC: {auroc:.4f}")
    print(f"🔍 Optimal Threshold (F1-based): {best_thresh:.4f}")
    print(f"📊 Image-Level F1 Score: {f1:.4f}")
    print(f"🧮 Confusion Matrix:\n{cm}")
    print(f"📏 Pixel-Level F1 Score: {px_f1_avg:.4f}")
    print(f"📐 Pixel-Level IoU: {px_iou_avg:.4f}\n")
    return f1

def train_one_epoch(dataloader, model, optimizer, epoch):
    model.train()
    loss_meter = utils.AverageMeter()
    for step, data in enumerate(dataloader):
        data = data.cuda()
        ret = model(data)
        loss = ret["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item())
        if (step + 1) % const.LOG_INTERVAL == 0 or (step + 1) == len(dataloader):
            print("Epoch {} - Step {}: loss = {:.3f}({:.3f})".format(epoch + 1, step + 1, loss_meter.val, loss_meter.avg))

def train(args):
    os.makedirs(const.CHECKPOINT_DIR, exist_ok=True)
    checkpoint_dir = os.path.join(const.CHECKPOINT_DIR, f"exp{len(os.listdir(const.CHECKPOINT_DIR))}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    config = yaml.safe_load(open(args.config, "r"))
    model = build_model(config)
    optimizer = build_optimizer(model)
    train_dataloader = build_train_data_loader(args, config)
    test_dataloader = build_test_data_loader(args, config)
    model.cuda()

    best_f1 = 0
    early_stop_counter = 0

    for epoch in range(const.NUM_EPOCHS):
        train_one_epoch(train_dataloader, model, optimizer, epoch)
        if (epoch + 1) % const.EVAL_INTERVAL == 0:
            f1 = eval_once(test_dataloader, model)
            if f1 > best_f1:
                best_f1 = f1
                early_stop_counter = 0
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))
                print("✅ Best model saved.")
            else:
                early_stop_counter += 1
                print("⏸️ No improvement, counter:", early_stop_counter)
            if early_stop_counter >= 5:
                print("🛑 Early stopping triggered.")
                break

def evaluate(args):
    config = yaml.safe_load(open(args.config, "r"))
    model = build_model(config)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint)
    test_dataloader = build_test_data_loader(args, config)
    model.cuda()
    eval_once(test_dataloader, model)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", "--config", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("-cat", "--category", type=str, choices=const.MVTEC_CATEGORIES, required=True)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("-ckpt", "--checkpoint", type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.eval:
        evaluate(args)
    else:
        train(args)