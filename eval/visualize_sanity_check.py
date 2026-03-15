import os
import sys
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.teacher_model import PillRetrievalModel
from utils.dataset_ogyei import build_ogyei_df_strict_split, OGYEICropDataset, LetterboxResize

def main():
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
    MODEL_PATH = "weights/best_kd_resnet18_kd_typecosine_alpha10.0_fold0.pth" 
    OUTPUT_DIR = os.path.join(os.getcwd(), 'reports', 'ogyei_eval')

    df = build_ogyei_df_strict_split(OGYEI_ROOT)
    
    transform = transforms.Compose([
        LetterboxResize(384),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    dataset = OGYEICropDataset(df, transform=transform)
    # Shuffle=False để index của DataLoader khớp chính xác với index của DataFrame
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    print("\n🚀 Đang trích xuất đặc trưng với Student KD...")
    model = PillRetrievalModel(num_classes=9804, backbone_type='resnet18', pooling_type='gem').to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    all_feats, all_is_ref = [], []
    with torch.no_grad():
        for imgs, _, _, is_ref, _ in tqdm(loader, desc="Extracting"):
            feats = model(imgs.to(DEVICE))
            all_feats.append((feats[-1] if isinstance(feats, tuple) else feats).cpu())
            all_is_ref.append(is_ref.cpu())

    all_feats = F.normalize(torch.cat(all_feats, dim=0), p=2, dim=1)
    all_is_ref = torch.cat(all_is_ref, dim=0)

    # Tách Features
    g_feats = all_feats[all_is_ref == 1]
    q_feats = all_feats[all_is_ref == 0]
    
    # Lấy Index gốc từ DataFrame để phục hồi ảnh PIL
    g_indices = df.index[df['is_ref'] == 1].tolist()
    q_indices = df.index[df['is_ref'] == 0].tolist()

    S = torch.mm(q_feats, g_feats.T)

    random.seed(42)
    selected_q_idx_in_matrix = random.sample(range(len(q_feats)), 10)

    fig, axes = plt.subplots(5, 4, figsize=(16, 20))
    fig.suptitle("Visual Sanity Check: Query vs Top-1 Gallery (OGYEIv2)", fontsize=20, fontweight='bold', y=0.98)

    for i, q_mat_idx in enumerate(selected_q_idx_in_matrix):
        # Thông tin Query
        global_q_idx = q_indices[q_mat_idx]
        q_lbl_name = df.iloc[global_q_idx]['label_name']
        img_q = dataset.get_pil_image(global_q_idx)
        
        # Tìm Top-1 Gallery
        top1_g_mat_idx = torch.argmax(S[q_mat_idx]).item()
        global_g_idx = g_indices[top1_g_mat_idx]
        g_lbl_name = df.iloc[global_g_idx]['label_name']
        img_g = dataset.get_pil_image(global_g_idx)
        
        is_correct = (q_lbl_name == g_lbl_name)
        color = 'green' if is_correct else 'red'
        sim_score = S[q_mat_idx][top1_g_mat_idx].item()
        
        row, col_q = i // 2, (i % 2) * 2
        
        # Vẽ Query
        axes[row, col_q].imshow(img_q)
        axes[row, col_q].set_title(f"QUERY\nClass: {q_lbl_name}", fontweight='bold')
        axes[row, col_q].axis('off')
        
        # Vẽ Gallery
        ax_g = axes[row, col_q + 1]
        ax_g.imshow(img_g)
        ax_g.set_title(f"TOP-1\nClass: {g_lbl_name} | Sim: {sim_score:.2f}", color=color, fontweight='bold')
        for spine in ax_g.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(4)
        ax_g.set_xticks([]); ax_g.set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(OUTPUT_DIR, 'sanity_check_visual.png'), dpi=300, bbox_inches='tight')
    print("✅ Đã lưu ảnh Visual Sanity Check!")

if __name__ == "__main__":
    main()