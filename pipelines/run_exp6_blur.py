import torch
import pandas as pd
import os
import sys
import csv
import datetime
from torchvision import transforms
from torch.utils.data import DataLoader

# Import các module hiện có của bạn
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.pill_retrieval_model import PillRetrievalModel
from utils.dataset_loader import PillDataset
from utils.evaluator import evaluate_retrieval
from utils.data_utils import load_epill_full_data

# ==========================================
# ⚙️ CẤU HÌNH THỰC NGHIỆM 6: BLUR ROBUSTNESS
# ==========================================
MODELS_TO_TEST = [
    {
        "title": "Teacher (ResNeSt101e)",
        "model_role": "Teacher (ResNeSt101e)",
        "backbone": "resnest101e_timm",
        "weight_path": "weights/phase2/best_resnest101e_timm_gem_fold0.pth"
    },
    {
        "title": "Student Baseline (ResNet18)",
        "model_role": "Student Baseline (ResNet18)",
        "backbone": "resnet18_tv",
        "weight_path": "weights/phase2/best_resnet18_tv_gem_fold0.pth"
    },
    {
        "title": "KD Student (ResNet18)",
        "model_role": "KD Student (ResNet18)",
        "backbone": "resnet18_tv",
        "weight_path": "weights/kd_models/best_resnest101e_timm_kd_resnet18_tv_kd_typecosine_fold0.pth"
    }
]

# Các mức độ làm mờ (Kernel Size, Sigma)
BLUR_LEVELS = [
    {"name": "No Blur (Baseline)", "kernel": None, "sigma": None},
    {"name": "Mild Blur", "kernel": 5, "sigma": 1.0},
    {"name": "Moderate Blur", "kernel": 9, "sigma": 2.0},
    {"name": "Severe Blur", "kernel": 15, "sigma": 3.0}
]

IMAGE_SIZE = 384
BATCH_SIZE = 128

def main():
    print("="*60)
    print(f"🚀 BẮT ĐẦU THỰC NGHIỆM 6: ROBUSTNESS TO GAUSSIAN BLUR")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df_all = load_epill_full_data()
    num_classes = df_all['sub_label_idx'].nunique()
    
    # Gom dữ liệu Validation và Gallery
    fold = 0
    df_ref_gallery = df_all[df_all['is_ref'] == 1].reset_index(drop=True)
    df_val = df_all[(df_all['fold'] == fold) & (df_all['is_ref'] == 0)].reset_index(drop=True)
    df_eval_combined = pd.concat([df_val, df_ref_gallery]).reset_index(drop=True)

    results = []

    # 1. Vòng lặp thay đổi mức độ Blur
    for blur_cfg in BLUR_LEVELS:
        b_name = blur_cfg["name"]
        k = blur_cfg["kernel"]
        s = blur_cfg["sigma"]
        
        print(f"\n[🌫️ MỨC ĐỘ: {b_name}] Đang chuẩn bị DataLoader...")
        
        # 🛡️ Cấu hình Transform: Thêm GaussianBlur nếu có
        transform_list = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
        
        if k is not None and s is not None:
            transform_list.append(transforms.GaussianBlur(kernel_size=k, sigma=s))
            
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        val_transform = transforms.Compose(transform_list)

        val_loader = DataLoader(
            PillDataset(df_eval_combined, transform=val_transform), 
            batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
        )

        # 2. Vòng lặp đánh giá từng mô hình
        for model_cfg in MODELS_TO_TEST:
            role = model_cfg["model_role"]
            backbone = model_cfg["backbone"]
            w_path = model_cfg["weight_path"]

            print(f"⏳ Đang đánh giá {role}...")
            
            if not os.path.exists(w_path):
                print(f"❌ KHÔNG TÌM THẤY WEIGHT: {w_path} -> Bỏ qua!")
                continue

            model = PillRetrievalModel(num_classes=num_classes, backbone_type=backbone, pooling_type='gem').to(device)
            model.load_state_dict(torch.load(w_path, map_location=device))
            model.eval()

            with torch.no_grad():
                with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                    val_metrics = evaluate_retrieval(model, val_loader, device, flag_both_side=True)
            
            map_val = val_metrics['mAP']
            r1_val = val_metrics['Rank-1']
            
            results.append({
                "Blur_Level": b_name,
                "Model_Role": role,
                "mAP": map_val,
                "Rank-1": r1_val
            })
            
            print(f"✅ {role} @ {b_name} -> mAP: {map_val:.4f} | Rank-1: {r1_val:.4f}")
            
            del model
            torch.cuda.empty_cache()

    # ==========================================
    # 💾 3. XUẤT KẾT QUẢ RA CSV
    # ==========================================
    os.makedirs("reports_kd", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"reports_kd/Exp6_Blur_Robustness_{timestamp}.csv"

    if results:
        with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["Blur_Level", "Model_Role", "mAP", "Rank-1"])
            writer.writeheader()
            writer.writerows(results)
            
        print("\n" + "="*60)
        print("🎉 THỰC NGHIỆM 6 ĐÃ HOÀN TẤT!")
        print(f"📊 Kết quả lưu tại: {csv_file}")
        print("="*60)
    else:
        print("\n⚠️ Không có kết quả nào được ghi nhận.")

if __name__ == "__main__":
    main()