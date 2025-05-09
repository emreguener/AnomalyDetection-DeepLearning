import torch
import torch.nn as nn
import numpy as np
import os
from functools import partial
import warnings
from tqdm import tqdm
from torch.nn.init import trunc_normal_
import argparse
from optimizers import StableAdamW
from utils import evaluation_batch, WarmCosineScheduler, global_cosine_hm_adaptive, setup_seed, get_logger

# Dataset-Related Modules
from dataset import MVTecDataset, RealIADDataset
from dataset import get_data_transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Model-Related Modules
from models import vit_encoder
from models.uad import INP_Former
from models.vision_transformer import Mlp, Aggregation_Block, Prototype_Block

warnings.filterwarnings("ignore")

# 🔧 Validation Loss Fonksiyonu
# 🔧 Validation Loss Fonksiyonu
def evaluate_val_loss(model, val_dataloader):
    model.eval()
    val_loss_list = []
    with torch.no_grad():
        for batch in val_dataloader:
            val_img = batch[0].to(device)
            en, de, _ = model(val_img)
            
            # en ve de liste olarak işleme
            if isinstance(en, list):
                # Listedeki tüm tensorları birleştir
                en_concat = torch.cat([e.flatten(1) for e in en], dim=1)
                de_concat = torch.cat([d.flatten(1) for d in de], dim=1)
                
                # Birleştirilmiş tensorlar ile loss hesapla
                loss = torch.nn.functional.cosine_embedding_loss(
                    en_concat,
                    de_concat,
                    torch.ones(en_concat.size(0)).to(device)
                )
            else:
                # en ve de zaten tensor ise orijinal yaklaşım
                loss = torch.nn.functional.cosine_embedding_loss(
                    en.view(en.size(0), -1),
                    de.view(de.size(0), -1),
                    torch.ones(en.size(0)).to(device)
                )
                
            val_loss_list.append(loss.item())
    return np.mean(val_loss_list)



def main(args):
    setup_seed(1)

    data_transform, gt_transform = get_data_transforms(args.input_size, args.crop_size)

    train_path = os.path.join(args.data_path, args.item, 'train')
    test_path = os.path.join(args.data_path, args.item)

    train_data = ImageFolder(root=train_path, transform=data_transform)
    test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    encoder = vit_encoder.load(args.encoder)
    if 'small' in args.encoder:
        embed_dim, num_heads = 384, 6
    elif 'base' in args.encoder:
        embed_dim, num_heads = 768, 12
    elif 'large' in args.encoder:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
    else:
        raise "Architecture not in small, base, large."

    Bottleneck = nn.ModuleList([Mlp(embed_dim, embed_dim * 4, embed_dim, drop=0.)])
    INP = nn.ParameterList([nn.Parameter(torch.randn(args.INP_num, embed_dim)) for _ in range(1)])
    INP_Extractor = nn.ModuleList([
        Aggregation_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                          qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
    ])
    INP_Guided_Decoder = nn.ModuleList([
        Prototype_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
        for _ in range(8)
    ])

    model = INP_Former(
        encoder=encoder, bottleneck=Bottleneck, aggregation=INP_Extractor,
        decoder=INP_Guided_Decoder, target_layers=target_layers, remove_class_token=True,
        fuse_layer_encoder=fuse_layer_encoder, fuse_layer_decoder=fuse_layer_decoder,
        prototype_token=INP
    ).to(device)

    if args.phase == 'train':
        trainable = nn.ModuleList([Bottleneck, INP_Guided_Decoder, INP_Extractor, INP])
        for m in trainable.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        optimizer = StableAdamW([{'params': trainable.parameters()}], lr=1e-3, betas=(0.9, 0.999),
                                weight_decay=1e-4, amsgrad=True, eps=1e-10)
        lr_scheduler = WarmCosineScheduler(
            optimizer, base_value=1e-3, final_value=1e-4,
            total_iters=args.total_epochs * len(train_dataloader),
            warmup_iters=100
        )

        print_fn(f'train image number: {len(train_data)}')

        for epoch in range(args.total_epochs):
            model.train()
            loss_list = []
            for img, _ in tqdm(train_dataloader, ncols=80):
                img = img.to(device)
                en, de, g_loss = model(img)
                loss = global_cosine_hm_adaptive(en, de, y=3)
                loss += 0.2 * g_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm(trainable.parameters(), max_norm=0.1)
                optimizer.step()
                loss_list.append(loss.item())
                lr_scheduler.step()

            avg_train_loss = np.mean(loss_list)
            print_fn(f'epoch [{epoch+1}/{args.total_epochs}], loss: {avg_train_loss:.4f}')
            
            # 🔍 Validation Loss hesapla
            val_loss = evaluate_val_loss(model, test_dataloader)
            print_fn(f'Validation loss: {val_loss:.4f}')

        results = evaluation_batch(model, test_dataloader, device, max_ratio=0.01, resize_mask=256)
        auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results
        print_fn(
            f'{args.item}: I-Auroc:{auroc_sp:.4f}, I-AP:{ap_sp:.4f}, I-F1:{f1_sp:.4f}, '
            f'P-AUROC:{auroc_px:.4f}, P-AP:{ap_px:.4f}, P-F1:{f1_px:.4f}, P-AUPRO:{aupro_px:.4f}'
        )

        os.makedirs(os.path.join(args.save_dir, args.save_name, args.item), exist_ok=True)
        torch.save(model.state_dict(), os.path.join(args.save_dir, args.save_name, args.item, 'model.pth'))
        return results

    elif args.phase == 'test':
        model.load_state_dict(torch.load(os.path.join(args.save_dir, args.save_name, args.item, 'model.pth')), strict=True)
        model.eval()
        results = evaluation_batch(model, test_dataloader, device, max_ratio=0.01, resize_mask=256)
        return results


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    parser = argparse.ArgumentParser(description='')

    # dataset info
    parser.add_argument('--dataset', type=str, default='MVTec-AD')
    parser.add_argument('--data_path', type=str, default='/content/drive/MyDrive/wood_dataset')
    parser.add_argument('--item', type=str, default='wood')

    # save info
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str, default='INP-Former-Single-Class')

    # model info
    parser.add_argument('--encoder', type=str, default='dinov2reg_vit_base_14')
    parser.add_argument('--input_size', type=int, default=252)
    parser.add_argument('--crop_size', type=int, default=252)
    parser.add_argument('--INP_num', type=int, default=6)

    # training info
    parser.add_argument('--total_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--phase', type=str, default='train')

    args = parser.parse_args()
    args.save_name = args.save_name + f'_dataset={args.dataset}_Encoder={args.encoder}_Resize={args.input_size}_Crop={args.crop_size}_INP_num={args.INP_num}'
    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    main(args)
