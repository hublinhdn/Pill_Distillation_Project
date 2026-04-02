import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# Thêm root path để import được các module local (models, utils)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.pill_retrieval_model import PillRetrievalModel
from utils.dataset_loader import PillDataset
from utils.data_utils import load_epill_full_data

# ==========================================
# 1. CÁC HÀM TÍNH TOÁN CKA SIMILARITY
# ==========================================
def center_gram(gram):
    n = gram.shape[0]
    H = torch.eye(n, device=gram.device) - torch.ones((n, n), device=gram.device) / n
    return H @ gram @ H

def linear_cka(features_x, features_y):
    gram_x = features_x @ features_x.T
    gram_y = features_y @ features_y.T
    
    gram_x_c = center_gram(gram_x)
    gram_y_c = center_gram(gram_y)
    
    scaled_hsic = torch.sum(gram_x_c * gram_y_c)
    norm_x = torch.sqrt(torch.sum(gram_x_c * gram_x_c))
    norm_y = torch.sqrt(torch.sum(gram_y_c * gram_y_c))
    
    return (scaled_hsic / (norm_x * norm_y)).item()

# Hàm tạo tên file trọng số chính xác theo train_teacher_cv.py
def get_weight_path(backbone, pooling='gem', fold=0, w_sce=1.0, w_csce=0.2, w_triplet=1.0, w_cont=1.0):
    folder_weight = 'weights/phase3'
    return f"{folder_weight}/best_{backbone}_{pooling}_fold{fold}_{w_sce}_{w_csce}_{w_triplet}_{w_cont}.pth"

# ==========================================
# 2. HÀM MAIN THỰC THI
# ==========================================
def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Đang chạy trên thiết bị: {device}")

    # Danh sách cấu hình (Đảm bảo giống hệt lúc train baseline)
    students = ['resnet18_tv', 'mobilenetv3_large_100_tv', 'ghostnet_100_timm', 'efficientnet_b1_timm']
    teachers = [
        'efficientnet_b5_timm', 'convnextv2_base.fcmae_ft_in22k_in1k_384_timm',
        'convnext_base_timm', 'resnest101e_timm', 'convnext_large_timm', 
        'seresnext101_32x4d_timm', 'densenet161_tv', 'tresnet_l_timm', 
        'resnet101_tv', 'tf_efficientnetv2_l.in21k_ft_in1k', 'maxvit_base_tf_384_timm'
    ]

    # Load dữ liệu để lấy total_sub_classes
    print("⏳ Đang chuẩn bị dữ liệu Validation...")
    df_all = load_epill_full_data()
    total_sub_classes = df_all['sub_label_idx'].nunique()
    
    # Chỉ lấy tập Query Consumer của Fold 0 làm tập tính CKA
    df_val = df_all[(df_all['fold'] == 0) & (df_all['is_ref'] == 0)].reset_index(drop=True)

    val_transform = transforms.Compose([
        transforms.Resize((384, 384)), # Dùng chung 384 vì đã gạch bỏ ViT Patch14
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Shuffle=True để lấy một lượng ảnh ngẫu nhiên đa dạng tính CKA
    val_loader = DataLoader(
        PillDataset(df_val, transform=val_transform),
        batch_size=64, shuffle=True, num_workers=4, pin_memory=True
    )

    # Khởi tạo ma trận rỗng
    cka_results = pd.DataFrame(index=students, columns=teachers)

    print(f"🚀 BẮT ĐẦU TÍNH MA TRẬN CKA (4 Students x 11 Teachers)")
    
    for s_name in students:
        print(f"\n🟢 Đang load Student: {s_name}")
        student_weight = get_weight_path(s_name)
        
        if not os.path.exists(student_weight):
            print(f"   ❌ Không tìm thấy trọng số: {student_weight} -> BỎ QUA!")
            continue

        student = PillRetrievalModel(num_classes=total_sub_classes, backbone_type=s_name, pooling_type='gem').to(device)
        student.load_state_dict(torch.load(student_weight, map_location=device))
        student.eval()

        for t_name in teachers:
            print(f"   🔵 Đối chiếu với Teacher: {t_name}")
            teacher_weight = get_weight_path(t_name)
            
            if not os.path.exists(teacher_weight):
                print(f"      ❌ Không tìm thấy: {teacher_weight} -> Đánh dấu NaN")
                cka_results.loc[s_name, t_name] = np.nan
                continue

            teacher = PillRetrievalModel(num_classes=total_sub_classes, backbone_type=t_name, pooling_type='gem').to(device)
            teacher.load_state_dict(torch.load(teacher_weight, map_location=device))
            teacher.eval()

            total_cka = 0.0
            num_batches = 0
            
            with torch.no_grad():
                # Lấy 10 batch (~640 ảnh) là đủ độ tin cậy thống kê cho ma trận Gram
                for i, (imgs, _, _, _) in enumerate(val_loader):
                    if i >= 10: break 
                    imgs = imgs.to(device)
                    
                    # Model trả về norm_embedding khi không có labels truyền vào
                    s_embed = student(imgs) 
                    t_embed = teacher(imgs)
                    
                    total_cka += linear_cka(t_embed, s_embed)
                    num_batches += 1
                    
            avg_cka = total_cka / num_batches
            cka_results.loc[s_name, t_name] = round(avg_cka, 4)
            print(f"      => CKA Score: {avg_cka:.4f}")
            
            del teacher
            torch.cuda.empty_cache()
            
        del student
        torch.cuda.empty_cache()

    # Xuất báo cáo
    os.makedirs("reports", exist_ok=True)
    csv_path = "reports/CKA_Similarity_Matrix.csv"
    cka_results.to_csv(csv_path)
    print("\n" + "="*50)
    print(f"✅ HOÀN TẤT! Ma trận CKA đã được lưu tại: {csv_path}")
    print("="*50)

if __name__ == '__main__':
    main()