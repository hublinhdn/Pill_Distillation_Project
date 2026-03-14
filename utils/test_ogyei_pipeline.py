import os
import glob
import random
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# ==========================================
# 1. HÀM TẠO DATAFRAME (Đã sửa theo chuẩn path mới)
# ==========================================
def build_ogyei_df(ogyei_root, use_all_train_as_gallery=True, n_refs=2):
    """
    Hàm tạo DataFrame cho bộ OGYEIv2 (Format YOLO phân rã phẳng).
    - use_all_train_as_gallery = True: Dùng toàn bộ thư mục 'train' làm Gallery.
    - use_all_train_as_gallery = False: Bốc ngẫu nhiên n_refs ảnh từ 'train' làm Gallery.
    """
    data = []
    splits = ['train', 'valid', 'test'] 
    
    # Dùng dictionary để gom nhóm các ảnh trong tập train theo Class ID
    train_class_dict = {}

    for split in splits:
        img_dir = os.path.join(ogyei_root, split, 'images')
        lbl_dir = os.path.join(ogyei_root, split, 'labels')
        
        if not os.path.exists(img_dir):
            continue
            
        # Quét toàn bộ ảnh trong thư mục images (không có thư mục con)
        img_paths = glob.glob(os.path.join(img_dir, '*.*'))
        
        for img_path in img_paths:
            img_name = os.path.basename(img_path)
            txt_name = os.path.splitext(img_name)[0] + '.txt'
            txt_path = os.path.join(lbl_dir, txt_name)
            
            class_id = -1
            # ĐỌC NHÃN TỪ FILE TXT
            if os.path.exists(txt_path):
                with open(txt_path, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        # Lấy con số đầu tiên trong file làm class_id
                        class_id = int(lines[0].strip().split()[0])
            
            if class_id == -1:
                continue # Bỏ qua nếu ảnh không có file nhãn hợp lệ
                
            if split == 'train':
                # Gom vào từ điển để lát nữa chia Gallery/Query
                if class_id not in train_class_dict:
                    train_class_dict[class_id] = []
                train_class_dict[class_id].append({
                    'image_path': img_path, 
                    'txt_path': txt_path, 
                    'label_idx': class_id,
                    'label_name': f"Class_{class_id}"
                })
            else:
                # Tập Valid/Test mặc định 100% là Query (is_ref = 0)
                data.append({
                    'image_path': img_path,
                    'txt_path': txt_path,
                    'label_idx': class_id,
                    'is_ref': 0,
                    'label_name': f"Class_{class_id}"
                })

    # Xử lý tập Train: Chia Gallery / Query
    for class_id, items in train_class_dict.items():
        if use_all_train_as_gallery:
            # Lấy TOÀN BỘ tập Train làm Gallery
            gallery_items = items
            query_items = []
        else:
            # Chỉ lấy n_refs ảnh làm Gallery, còn dư đẩy xuống Query
            random.shuffle(items)
            gallery_items = items[:n_refs]
            query_items = items[n_refs:]
        
        for item in gallery_items:
            item['is_ref'] = 1
            data.append(item)
            
        for item in query_items:
            item['is_ref'] = 0
            data.append(item)

    df = pd.DataFrame(data)
    
    if len(df) == 0:
        print("❌ Lỗi: DataFrame rỗng! Hãy kiểm tra lại đường dẫn thư mục images/labels.")
        return df
        
    df['sub_label_idx'] = df['label_idx'] # Fake sub_label cho tương thích Pipeline cũ
    
    print(f"📦 Đã load OGYEIv2 (YOLO Format) | Tổng: {len(df)} ảnh | Gallery: {len(df[df['is_ref']==1])} | Query: {len(df[df['is_ref']==0])}")
    return df
# ==========================================
# 2. DATASET THỰC HIỆN CROP ON-THE-FLY
# ==========================================
class OGYEICropDataset(Dataset):
    def __init__(self, df, transform=None, margin_ratio=0.15):
        self.df = df
        self.transform = transform
        self.margin_ratio = margin_ratio

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        txt_path = row['txt_path']
        
        img = Image.open(img_path).convert('RGB')
        img_w, img_h = img.size
        cropped_img = img 
        
        if pd.notna(txt_path) and os.path.exists(txt_path):
            try:
                with open(txt_path, 'r') as f:
                    lines = f.readlines()
                
                if lines:
                    parts = lines[0].strip().split()
                    if len(parts) >= 5: # Chắc chắn là format Polygon
                        coords = [float(p) for p in parts[1:]]
                        x_coords = coords[0::2]
                        y_coords = coords[1::2]

                        min_x, max_x = min(x_coords), max(x_coords)
                        min_y, max_y = min(y_coords), max(y_coords)

                        abs_min_x = min_x * img_w
                        abs_max_x = max_x * img_w
                        abs_min_y = min_y * img_h
                        abs_max_y = max_y * img_h

                        box_w = abs_max_x - abs_min_x
                        box_h = abs_max_y - abs_min_y

                        pad_x = box_w * self.margin_ratio
                        pad_y = box_h * self.margin_ratio

                        crop_x1 = max(0, abs_min_x - pad_x)
                        crop_y1 = max(0, abs_min_y - pad_y)
                        crop_x2 = min(img_w, abs_max_x + pad_x)
                        crop_y2 = min(img_h, abs_max_y + pad_y)

                        cropped_img = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            except Exception as e:
                print(f"⚠️ Lỗi crop ảnh {img_path}: {e}")

        if self.transform:
            tensor_img = self.transform(cropped_img)
        else:
            tensor_img = cropped_img # Trả về PIL Image để dễ visualize

        return tensor_img, row['sub_label_idx'], row['label_idx'], row['is_ref'], row['label_name']

# ==========================================
# 3. KHU VỰC TEST TRÊN MÁY CÁ NHÂN
# ==========================================
if __name__ == '__main__':
    # THAY ĐỔI ĐƯỜNG DẪN NÀY THÀNH FOLDER OGYEIv2 TRÊN MÁY BẠN
    root_path = 'data/raw/OGYEIv2/ogyeiv2'
    OGYEI_ROOT = os.path.join(root_path, 'ogyeiv2')

    # OGYEI_ROOT = r"C:\path\to\your\OGYEIv2" 
    
    # 1. Khởi tạo DataFrame
    df = build_ogyei_df(OGYEI_ROOT, n_refs=2)
    
    # 2. Khởi tạo Dataset (Không dùng transform để trả về ảnh gốc dễ in ra matplotlib)
    dataset = OGYEICropDataset(df, transform=None, margin_ratio=0.15)
    
    if len(dataset) == 0:
        print("❌ Không tìm thấy dữ liệu. Hãy kiểm tra lại cấu trúc thư mục!")
        exit()

    # 3. Lấy ngẫu nhiên 6 mẫu để vẽ lên màn hình
    indices = random.sample(range(len(dataset)), min(6, len(dataset)))
    
    plt.figure(figsize=(15, 8))
    plt.suptitle("KIỂM THỬ OGYEIv2 - CROP ON-THE-FLY", fontsize=16, fontweight='bold')
    
    for i, idx in enumerate(indices):
        img, sub_lbl, lbl, is_ref, class_name = dataset[idx]
        
        role = "GALLERY (Từ điển)" if is_ref == 1 else "QUERY (Cần tìm)"
        color = 'green' if is_ref == 1 else 'blue'
        
        plt.subplot(2, 3, i + 1)
        plt.imshow(img)
        plt.title(f"Class: {class_name}\nRole: {role}", color=color, fontsize=10)
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()
    
    print("\n✅ Nếu bạn thấy các ảnh viên thuốc được CẮT GỌN GÀNG, không dư quá nhiều nền gỗ/bàn...")
    print("🚀 NGHĨA LÀ DATA PIPELINE ĐÃ SẴN SÀNG 100% CHO KIỂM THỬ MÔ HÌNH!")