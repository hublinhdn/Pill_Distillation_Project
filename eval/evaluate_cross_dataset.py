import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

# Import local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Kéo các module từ project của bạn vào
from models.teacher_model import PillRetrievalModel
from utils.test_ogyei_pipeline import build_ogyei_df, OGYEICropDataset

# ==========================================
# 1. TIỀN XỬ LÝ ẢNH (GIỮ TỶ LỆ)
# ==========================================
class LetterboxResize:
    def __init__(self, size=384):
        self.size = size

    def __call__(self, img):
        img.thumbnail((self.size, self.size), Image.LANCZOS)
        delta_w = self.size - img.size[0]
        delta_h = self.size - img.size[1]
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        return ImageOps.expand(img, padding, fill=(127, 127, 127))

# ==========================================
# 2. HÀM ĐÁNH GIÁ (SINGLE-SIDE)
# ==========================================
def evaluate_model(model, dataloader, device):
    model.eval()
    all_feats, all_labels, all_is_ref = [], [], []

    with torch.no_grad():
        for imgs, _, labels, is_ref, _ in tqdm(dataloader, desc="Extracting Features", leave=False):
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
        q_lbl = q_labels[i].item()
        scores = S[i]
        sorted_indices = torch.argsort(scores, descending=True)
        sorted_g_labels = g_labels[sorted_indices]
        
        match_idx = (sorted_g_labels == q_lbl).nonzero(as_tuple=True)[0]
        ranks.append(match_idx[0].item() + 1 if len(match_idx) > 0 else float('inf'))

    ranks = np.array(ranks)
    valid_mask = ranks != float('inf')
    rank1 = (ranks[valid_mask] == 1).sum() / len(q_feats)
    mAP = (1.0 / ranks[valid_mask]).sum() / len(q_feats)
    
    return rank1, mAP

# ==========================================
# 3. HÀM VẼ BIỂU ĐỒ & XUẤT BÁO CÁO
# ==========================================
def generate_report_and_plots(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    names = [r['name'] for r in results]
    maps = [r['mAP'] for r in results]
    rank1s = [r['rank1'] * 100 for r in results]

    # --- 3.1. VẼ BIỂU ĐỒ (VISUALIZATION) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    colors = ['#ff9999', '#66b3ff', '#99ff99']

    # Biểu đồ mAP
    bars1 = ax1.bar(names, maps, color=colors, edgecolor='black')
    ax1.set_title('mAP Score Comparison (OGYEIv2)', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, max(maps) * 1.2)
    ax1.set_ylabel('mAP')
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.4f}', ha='center', va='bottom', fontweight='bold')

    # Biểu đồ Rank-1
    bars2 = ax2.bar(names, rank1s, color=colors, edgecolor='black')
    ax2.set_title('Rank-1 Accuracy Comparison (%)', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.set_ylabel('Rank-1 (%)')
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f'{yval:.2f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'performance_comparison.png')
    plt.savefig(plot_path, dpi=300)
    print(f"📊 Đã lưu biểu đồ tại: {plot_path}")

    # --- 3.2. XUẤT FILE TEXT BÁO CÁO ---
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("BÁO CÁO ĐÁNH GIÁ CHÉO (CROSS-DATASET) TRÊN OGYEIv2\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"{'Mô hình':<25} | {'Rank-1 (%)':<12} | {'mAP':<10}\n")
        f.write("-" * 55 + "\n")
        
        baseline_map = results[0]['mAP'] if 'Baseline' in results[0]['name'] else 0
        kd_map = 0
        
        for r in results:
            f.write(f"{r['name']:<25} | {r['rank1']*100:>10.2f} % | {r['mAP']:>8.4f}\n")
            if 'Baseline' in r['name']: baseline_map = r['mAP']
            if 'KD' in r['name']: kd_map = r['mAP']
            
        f.write("\n" + "="*60 + "\n")
        f.write("PHÂN TÍCH CHUYÊN SÂU (INSIGHTS):\n")
        if kd_map > 0 and baseline_map > 0:
            gain = kd_map - baseline_map
            gain_pct = (gain / baseline_map) * 100
            f.write(f"- Phương pháp KD giúp mô hình ResNet18 tăng tuyệt đối {gain:.4f} mAP so với Baseline.\n")
            f.write(f"- Tốc độ tăng trưởng tương đối (Relative Gain): +{gain_pct:.2f}% hiệu năng trên tập dữ liệu ngoại lai.\n")
            
            if gain > 0:
                f.write("- KẾT LUẬN: Mô hình Student KD đã học được đặc trưng tổng quát (Generalization) từ Teacher, vượt trội hoàn toàn so với việc tự học (Baseline) trên miền dữ liệu Unseen.\n")
            else:
                f.write("- KẾT LUẬN: Cần xem xét lại hệ số KD trên miền dữ liệu mới.\n")

    print(f"📝 Đã lưu file báo cáo tại: {report_path}")

# ==========================================
# 4. CHƯƠNG TRÌNH CHÍNH
# ==========================================
if __name__ == "__main__":
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root_path = 'data/raw/OGYEIv2/ogyeiv2'
    OGYEI_ROOT = os.path.join(root_path, 'ogyeiv2') # <-- SỬA ĐƯỜNG DẪN OGYEI
    OUTPUT_DIR = os.path.join(os.getcwd(), 'reports', 'ogyei_eval')

    STUDENT_BASE_WEIGHT = "weights/best_resnet18_gem_fold0.pth" # <-- SỬA PATH BASELINE
    TEACHER_WEIGHT = "weights/best_teacher_convnext_base_fold0.pth" # <-- SỬA PATH TEACHER
    STUDENT_KD_WEIGHT = "weights/best_kd_resnet18_kd_typecosine_alpha10.0_fold0.pth" # <-- SỬA PATH KD
    
    # Cấu hình danh sách 3 mô hình
    MODELS_CONFIG = [
        {
            "name": "Student Baseline", 
            "backbone": "resnet18", 
            "path": STUDENT_BASE_WEIGHT 
        },
        {
            "name": "Teacher Model", 
            "backbone": "convnext_base",               # <-- SỬA BACKBONE NẾU CẦN
            "path": TEACHER_WEIGHT         
        },
        {
            "name": "Student KD (Ours)", 
            "backbone": "resnet18", 
            "path": STUDENT_KD_WEIGHT
        }
    ]

    transform = transforms.Compose([
        LetterboxResize(384),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("🔍 Đang load OGYEIv2 Data...")
    df = build_ogyei_df(OGYEI_ROOT, use_all_train_as_gallery=True)
    dataset = OGYEICropDataset(df, transform=transform)
    loader = DataLoader(dataset, batch_size=32, num_workers=4)

    final_results = []

    for cfg in MODELS_CONFIG:
        print(f"\n🚀 Đang chạy đánh giá: {cfg['name']}...")
        try:
            # Lưu ý num_classes (9804) phải khớp với lúc cấu hình Model gốc
            model = PillRetrievalModel(num_classes=9804, backbone_type=cfg['backbone'], pooling_type='gem').to(DEVICE)
            model.load_state_dict(torch.load(cfg['path'], map_location=DEVICE))
            
            r1, mAP = evaluate_model(model, loader, DEVICE)
            final_results.append({
                'name': cfg['name'],
                'rank1': r1,
                'mAP': mAP
            })
            print(f"   => mAP: {mAP:.4f} | Rank-1: {r1*100:.2f}%")
            
            # Xóa model khỏi RAM/VRAM để tránh tràn bộ nhớ khi load model tiếp theo
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ Lỗi khi tải mô hình {cfg['name']}: {e}")

    if len(final_results) > 0:
        print("\n" + "="*50)
        print("ĐANG XUẤT BÁO CÁO VÀ BIỂU ĐỒ...")
        generate_report_and_plots(final_results, OUTPUT_DIR)
        print("✅ HOÀN TẤT QUÁ TRÌNH NGHIỆM THU!")
    else:
        print("⚠️ Không có mô hình nào chạy thành công để xuất báo cáo.")