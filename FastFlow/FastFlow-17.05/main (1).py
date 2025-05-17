import argparse
import os

import torch
import yaml
from ignite.contrib import metrics
from sklearn.metrics import roc_auc_score
import csv

import constants as const
import dataset
import fastflow
import utils

from sklearn.metrics import roc_auc_score, f1_score, jaccard_score
import numpy as np


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
        batch_size=const.BATCH_SIZE,
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
    print(
        "Model A.D. Param#: {}".format(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        )
    )
    return model


def build_optimizer(model):
    return torch.optim.Adam(
        model.parameters(), lr=const.LR, weight_decay=const.WEIGHT_DECAY
    )


def train_one_epoch(dataloader, model, optimizer, epoch):
    model.train()
    loss_meter = utils.AverageMeter()
    for step, data in enumerate(dataloader):
        # forward
        data = data.cuda()
        ret = model(data)
        loss = ret["loss"]
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # log
        loss_meter.update(loss.item())
        if (step + 1) % const.LOG_INTERVAL == 0 or (step + 1) == len(dataloader):
            print(
                "Epoch {} - Step {}: loss = {:.3f}({:.3f})".format(
                    epoch + 1, step + 1, loss_meter.val, loss_meter.avg
                )
            )




def find_best_threshold(scores, labels):
    best_f1 = 0
    best_thresh = None
    for t in np.linspace(min(scores), max(scores), num=100):
        preds = (np.array(scores) > t).astype(int)
        f1 = f1_score(labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    return best_thresh, best_f1


def eval_once(dataloader, model, save_csv_path="eval_results.csv", fixed_threshold=0.485):
    model.eval()
    all_img_scores = []
    all_img_labels = []
    results = []

    print(f"🧪 Test loader uzunluğu: {len(dataloader)}")

    for idx, (data, targets) in enumerate(dataloader):
        data, targets = data.cuda(), targets.cuda()
        with torch.no_grad():
            ret = model(data)

        anomaly_maps = ret["anomaly_map"].detach().cpu()  # (B,1,H,W)
        targets = targets.cpu()  # (B,1,H,W)

        # Görsel skor ve label
        raw_scores = anomaly_maps.mean(dim=(1, 2, 3)).numpy()
        scores = -raw_scores  # tersle
        scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-8)  # normalize
        labels = (targets.mean(dim=(1, 2, 3)) > 0.01).float().numpy()

        all_img_scores.extend(scores)
        all_img_labels.extend(labels)

        # Sonuçları geçici olarak topla
        for i in range(len(scores)):
            results.append({
                "index": idx * len(scores) + i,
                "score": scores[i],
                "gt_label": int(labels[i]),
                "image_index": idx * len(scores) + i
            })

    # NumPy'ye dönüştür
    all_img_scores = np.array(all_img_scores)
    all_img_labels = np.array(all_img_labels)

    print("📊 Anomaly skor aralığı: min =", np.min(all_img_scores), "max =", np.max(all_img_scores))
    print("📊 Anomali içeren test görseli sayısı:", int(np.sum(all_img_labels)))

    # Tahminler
    preds = (all_img_scores > fixed_threshold).astype(int)

    # Skorlar üzerinden metrikler
    try:
        auroc = roc_auc_score(all_img_labels, all_img_scores)
    except:
        auroc = -1
    try:
        f1 = f1_score(all_img_labels, preds)
    except:
        f1 = -1
    try:
        iou = jaccard_score(all_img_labels, preds)
    except:
        iou = -1

    print(f"✅ AUROC (image-level): {auroc:.4f}")
    print(f"✅ F1 Score (image-level, t={fixed_threshold:.4f}): {f1:.4f}")
    print(f"✅ IoU (image-level, t={fixed_threshold:.4f}): {iou:.4f}")

    # 📁 CSV’ye yaz
    with open(save_csv_path, mode="w", newline="") as f:
        fieldnames = ["image_index", "score", "gt_label", "prediction", "correct", "false_positive", "false_negative"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(results):
            gt = row["gt_label"]
            pred = int(all_img_scores[i] > fixed_threshold)
            correct = pred == gt
            fp = (gt == 0 and pred == 1)
            fn = (gt == 1 and pred == 0)
            writer.writerow({
                "image_index": row["image_index"],
                "score": row["score"],
                "gt_label": gt,
                "prediction": pred,
                "correct": correct,
                "false_positive": fp,
                "false_negative": fn
            })

    print(f"📁 CSV kaydedildi: {os.path.abspath(save_csv_path)}")

    


def train(args):
    os.makedirs(const.CHECKPOINT_DIR, exist_ok=True)
    checkpoint_dir = os.path.join(
        const.CHECKPOINT_DIR, "exp%d" % len(os.listdir(const.CHECKPOINT_DIR))
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    config = yaml.safe_load(open(args.config, "r"))
    model = build_model(config)
    optimizer = build_optimizer(model)

    train_dataloader = build_train_data_loader(args, config)
    test_dataloader = build_test_data_loader(args, config)
    model.cuda()

    for epoch in range(const.NUM_EPOCHS):
        train_one_epoch(train_dataloader, model, optimizer, epoch)
        if (epoch + 1) % const.EVAL_INTERVAL == 0:
            eval_once(test_dataloader, model)
        if (epoch + 1) % const.CHECKPOINT_INTERVAL == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(checkpoint_dir, "%d.pt" % epoch),
            )


def evaluate(args):
    config = yaml.safe_load(open(args.config, "r"))
    model = build_model(config)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_dataloader = build_test_data_loader(args, config)
    model.cuda()
    eval_once(test_dataloader, model)


def parse_args():
    parser = argparse.ArgumentParser(description="Train FastFlow on MVTec-AD dataset")
    parser.add_argument(
        "-cfg", "--config", type=str, required=True, help="path to config file"
    )
    parser.add_argument("--data", type=str, required=True, help="path to mvtec folder")
    parser.add_argument(
        "-cat",
        "--category",
        type=str,
        choices=const.MVTEC_CATEGORIES,
        required=True,
        help="category name in mvtec",
    )
    parser.add_argument("--eval", action="store_true", help="run eval only")
    parser.add_argument(
        "-ckpt", "--checkpoint", type=str, help="path to load checkpoint"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.eval:
        evaluate(args)
    else:
        train(args)