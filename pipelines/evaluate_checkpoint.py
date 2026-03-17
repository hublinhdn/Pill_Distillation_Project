import argparse
import torch
import pandas as pd
import os
import sys
from torch.utils.data import DataLoader
from torchvision import transforms

# Import local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.pill_retrieval_model import PillRetrievalModel
from utils.dataset_loader import PillDataset
from utils.evaluator import evaluate_retrieval
from utils.data_utils import load_epill_full_data

def main():
    parser = argparse.ArgumentParser(description="Đánh giá mô hình đã huấn luyện (Standalone Evaluation)")
    parser.add_argument('--weight_path', type=str, required=True, help='Đường dẫn đến file .pth')
    parser.add_argument('--backbone', type=str, required=True, help='Tên backbone (VD: convnext_base, resnet50)')
    parser.add_argument('--pooling', type=str, default='gem', choices=['gem', 'mpncov'], help='Loại pooling dùng khi train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size cho quá trình trích xuất đặc trưng')
    parser.add_argument('--fold', type=int, default=0, help='Fold để đánh giá (mặc định: 0)')
    parser.add_argument('--single_side', action='store_true', help='Thêm cờ này nếu CHỈ muốn đánh giá Single Side (Bỏ qua Late Fusion)')
    args = parser.parse_args()

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_type = 'cuda'
    elif torch.backends.mps.is_available(): # Dành cho MacBook chip M1/M2/M3
        device = torch.device('mps')
        device_type = 'mps'
    else: # Dành cho MacBook chip Intel
        device = torch.device('cpu')
        device_type = 'cpu'
    print(f"🚀 Khởi động Evaluation Script trên thiết bị: {device}")
    print(f"📦 Đang nạp trọng số từ: {args.weight_path}")

    # --- 1. CẤU HÌNH ĐỘNG KÍCH THƯỚC ẢNH (Phải khớp với lúc Train) ---
    if 'convnext' in args.backbone:
        img_size = 384
    elif 'resnet101' in args.backbone:
        img_size = 448
    else: 
        img_size = 448
        
    print(f"⚙️ Config {args.backbone}: Size={img_size}x{img_size}")

    # --- 2. CHUẨN BỊ DỮ LIỆU ---
    df_all = load_epill_full_data()
    total_sub_classes = df_all['sub_label_idx'].nunique()

    # Lấy Gallery (is_ref = 1) và Query của fold hiện tại (is_ref = 0)
    df_ref_gallery = df_all[df_all['is_ref'] == 1].reset_index(drop=True)
    df_val = df_all[(df_all['fold'] == args.fold) & (df_all['is_ref'] == 0)].reset_index(drop=True)
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_loader = DataLoader(
        PillDataset(pd.concat([df_val, df_ref_gallery]).reset_index(drop=True), transform=val_transform), 
        batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    # --- 3. KHỞI TẠO MÔ HÌNH VÀ NẠP TRỌNG SỐ ---
    model = PillRetrievalModel(num_classes=total_sub_classes, backbone_type=args.backbone, pooling_type=args.pooling).to(device)
    
    # Load weights (Bỏ qua lỗi strict nếu có các tham số phụ không khớp)
    if os.path.exists(args.weight_path):
        model.load_state_dict(torch.load(args.weight_path, map_location=device))
        print("✅ Nạp trọng số thành công!")
    else:
        print(f"❌ Lỗi: Không tìm thấy file trọng số tại {args.weight_path}")
        return

    # --- 4. TIẾN HÀNH ĐÁNH GIÁ ---
    # Cờ flag_both_side sẽ là True nếu người dùng KHÔNG truyền --single_side
    is_both_side = not args.single_side 
    mode_name = "BOTH SIDES (Late Fusion)" if is_both_side else "SINGLE SIDE"
    
    print(f"\n{'='*50}")
    print(f"🔍 ĐANG ĐÁNH GIÁ CHẾ ĐỘ: {mode_name}")
    print(f"{'='*50}")
    
    val_metrics = evaluate_retrieval(model, val_loader, device, flag_both_side=is_both_side)
    
    print("\n" + "="*50)
    print("🏆 KẾT QUẢ ĐÁNH GIÁ")
    print("="*50)
    print(f" - Backbone: {args.backbone}")
    print(f" - Chế độ:   {mode_name}")
    print(f" - Rank-1:   {val_metrics['Rank-1']:.4f}")
    print(f" - mAP:      {val_metrics['mAP']:.4f}")
    print("="*50 + "\n")

if __name__ == '__main__':
    main()