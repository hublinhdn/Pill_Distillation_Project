import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import module nội bộ
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.pill_retrieval_model import PillRetrievalModel
from utils.dataset_ogyei import build_ogyei_df_strict_split, OGYEICropDataset, LetterboxResize

USE_TRAIN_AS_GALLERY = True # use valid (6) or train(28) as gallery for query

def evaluate_model(model, dataloader, device):
    model.eval()
    all_feats, all_labels, all_is_ref = [], [], []

    with torch.no_grad():
        for imgs, _, labels, is_ref, _ in tqdm(dataloader, desc="Extracting", leave=False):
            imgs = imgs.to(device)
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(imgs)
                feats = outputs[-1] if isinstance(outputs, tuple) else outputs
            all_feats.append(feats.cpu())
            all_labels.append(labels.cpu())
            all_is_ref.append(is_ref.cpu())

    all_feats = F.normalize(torch.cat(all_feats, dim=0), p=2, dim=1)
    all_labels = torch.cat(all_labels, dim=0)
    all_is_ref = torch.cat(all_is_ref, dim=0)

    g_feats, g_labels = all_feats[all_is_ref == 1], all_labels[all_is_ref == 1]
    q_feats, q_labels = all_feats[all_is_ref == 0], all_labels[all_is_ref == 0]

    S = torch.mm(q_feats, g_feats.T)
    
    ranks = []
    for i in range(len(q_feats)):
        match_idx = (g_labels[torch.argsort(S[i], descending=True)] == q_labels[i]).nonzero(as_tuple=True)[0]
        ranks.append(match_idx[0].item() + 1 if len(match_idx) > 0 else float('inf'))

    ranks = np.array(ranks)
    valid_mask = ranks != float('inf')
    rank1 = (ranks[valid_mask] == 1).sum() / len(q_feats)
    mAP = (1.0 / ranks[valid_mask]).sum() / len(q_feats)
    return rank1, mAP

def generate_report(results, output_dir, output_img_path='performance_comparison.png', output_txt_report='evaluation_report.txt'):
    os.makedirs(output_dir, exist_ok=True)
    names = [r['name'] for r in results]
    maps = [r['mAP'] for r in results]
    rank1s = [r['rank1'] * 100 for r in results]

    # --- Vẽ biểu đồ ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    
    bars1 = ax1.bar(names, maps, color=colors, edgecolor='black')
    ax1.set_title('mAP Score Comparison', fontweight='bold')
    for bar in bars1: ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01, f'{bar.get_height():.4f}', ha='center', va='bottom', fontweight='bold')

    bars2 = ax2.bar(names, rank1s, color=colors, edgecolor='black')
    ax2.set_title('Rank-1 Accuracy (%)', fontweight='bold')
    for bar in bars2: ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1.0, f'{bar.get_height():.2f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, output_img_path), dpi=300)

    # --- Xuất báo cáo text ---
    with open(os.path.join(output_dir, output_txt_report), 'w', encoding='utf-8') as f:
        f.write("="*60 + "\nBÁO CÁO ĐÁNH GIÁ OGYEIv2 (STRICT SPLIT)\n" + "="*60 + "\n")
        f.write(f"{'Mô hình':<25} | {'Rank-1 (%)':<12} | {'mAP':<10}\n" + "-" * 55 + "\n")
        for r in results:
            f.write(f"{r['name']:<25} | {r['rank1']*100:>10.2f} % | {r['mAP']:>8.4f}\n")

def do_cross_check(gallery_split = 'train'):
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
        device_type = 'cuda'
    elif torch.backends.mps.is_available(): # Dành cho MacBook chip M1/M2/M3
        DEVICE = torch.device('mps')
        device_type = 'mps'
    else: # Dành cho MacBook chip Intel
        DEVICE = torch.device('cpu')
        device_type = 'cpu'
    OGYEI_ROOT = os.path.join('data/raw/OGYEIv2/ogyeiv2', 'ogyeiv2') 
    OUTPUT_DIR = os.path.join(os.getcwd(), 'reports', 'ogyei_eval')

    MODELS_CONFIG = [
        {"name": "Student Baseline", "backbone": "resnet18", "path": "weights/best_resnet18_gem_fold0.pth"},
        {"name": "Teacher Model", "backbone": "convnext_base", "path": "weights/best_teacher_convnext_base_fold0.pth"},
        {"name": "Student KD (Ours)", "backbone": "resnet18", "path": "weights/best_kd_resnet18_kd_typecosine_alpha10.0_fold0.pth"}
    ]

    transform = transforms.Compose([
        LetterboxResize(384),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    df = build_ogyei_df_strict_split(OGYEI_ROOT,gallery_split=gallery_split)
    loader = DataLoader(OGYEICropDataset(df, transform), batch_size=32, num_workers=4)

    results = []
    for cfg in MODELS_CONFIG:
        print(f"\n🚀 Đang chạy: {cfg['name']}...")
        model = PillRetrievalModel(num_classes=9804, backbone_type=cfg['backbone'], pooling_type='gem').to(DEVICE)
        model.load_state_dict(torch.load(cfg['path'], map_location=DEVICE, weights_only=True))
        r1, mAP = evaluate_model(model, loader, DEVICE)
        results.append({'name': cfg['name'], 'rank1': r1, 'mAP': mAP})
        del model
        torch.cuda.empty_cache()

    if results: 
        print(results)
        output_img_path=f'split_{gallery_split}_performance_comparison.png'
        output_txt_report=f'split_{gallery_split}_evaluation_report.txt'
        generate_report(results, OUTPUT_DIR, output_img_path=output_img_path, output_txt_report=output_txt_report)
if __name__ == "__main__":
    # gallery_split = 'train' if USE_TRAIN_AS_GALLERY else 'valid' 
    do_cross_check('train')
    do_cross_check('valid')
    