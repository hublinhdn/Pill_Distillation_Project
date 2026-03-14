import os
import glob
import pandas as pd
from PIL import Image, ImageOps
from torch.utils.data import Dataset

# --- 1. TRANSFORMS DÙNG CHUNG ---
class LetterboxResize:
    """Giữ nguyên tỷ lệ ảnh, chèn viền xám để tạo thành hình vuông"""
    def __init__(self, size=384):
        self.size = size

    def __call__(self, img):
        img.thumbnail((self.size, self.size), Image.LANCZOS)
        delta_w = self.size - img.size[0]
        delta_h = self.size - img.size[1]
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        return ImageOps.expand(img, padding, fill=(127, 127, 127))

# --- 2. HÀM BUILD DATAFRAME CHUẨN (STRICT SPLIT) ---
def build_ogyei_df_strict_split(ogyei_root):
    """
    Valid -> Gallery (is_ref = 1)
    Test -> Query (is_ref = 0)
    Bỏ qua Train để đánh giá khắt khe nhất.
    """
    data = []
    splits = ['valid', 'test'] 
    
    for split in splits:
        img_dir = os.path.join(ogyei_root, split, 'images')
        lbl_dir = os.path.join(ogyei_root, split, 'labels')
        
        if not os.path.exists(img_dir):
            continue
            
        img_paths = glob.glob(os.path.join(img_dir, '*.*'))
        
        for img_path in img_paths:
            img_name = os.path.basename(img_path)
            txt_path = os.path.join(lbl_dir, os.path.splitext(img_name)[0] + '.txt')
            
            class_id = -1
            if os.path.exists(txt_path):
                with open(txt_path, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        class_id = int(lines[0].strip().split()[0])
            
            if class_id == -1:
                continue 
                
            is_ref_val = 1 if split == 'valid' else 0
            
            data.append({
                'image_path': img_path,
                'txt_path': txt_path,
                'label_idx': class_id,
                'is_ref': is_ref_val,
                'label_name': f"Class_{class_id}"
            })

    df = pd.DataFrame(data)
    if len(df) > 0:
        df['sub_label_idx'] = df['label_idx']
        print(f"📦 Load OGYEIv2 (Strict) | Gallery (Valid): {len(df[df['is_ref']==1])} | Query (Test): {len(df[df['is_ref']==0])}")
    else:
        print("❌ Lỗi: DataFrame rỗng!")
    return df

# --- 3. DATASET CROP ON-THE-FLY ---
class OGYEICropDataset(Dataset):
    def __init__(self, df, transform=None, margin_ratio=0.15):
        self.df = df
        self.transform = transform
        self.margin_ratio = margin_ratio

    def __len__(self):
        return len(self.df)

    def _crop_image(self, img_path, txt_path):
        """Hàm nội bộ thực hiện vật lý việc cắt ảnh"""
        img = Image.open(img_path).convert('RGB')
        if pd.isna(txt_path) or not os.path.exists(txt_path):
            return img
            
        try:
            with open(txt_path, 'r') as f:
                lines = f.readlines()
            if not lines: return img
            
            parts = lines[0].strip().split()
            if len(parts) >= 5:
                img_w, img_h = img.size
                coords = [float(p) for p in parts[1:]]
                x_coords, y_coords = coords[0::2], coords[1::2]

                min_x, max_x = min(x_coords), max(x_coords)
                min_y, max_y = min(y_coords), max(y_coords)

                box_w, box_h = (max_x - min_x) * img_w, (max_y - min_y) * img_h
                pad_x, pad_y = box_w * self.margin_ratio, box_h * self.margin_ratio

                crop_x1 = max(0, min_x * img_w - pad_x)
                crop_y1 = max(0, min_y * img_h - pad_y)
                crop_x2 = min(img_w, max_x * img_w + pad_x)
                crop_y2 = min(img_h, max_y * img_h + pad_y)

                return img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        except Exception:
            pass
        return img

    def get_pil_image(self, idx):
        """Hỗ trợ lấy ảnh PIL gốc đã crop (để phục vụ script Visualize)"""
        row = self.df.iloc[idx]
        cropped_img = self._crop_image(row['image_path'], row['txt_path'])
        # Ép qua Letterbox luôn để vẽ cho đẹp
        return LetterboxResize(384)(cropped_img)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cropped_img = self._crop_image(row['image_path'], row['txt_path'])
        
        tensor_img = self.transform(cropped_img) if self.transform else cropped_img
        return tensor_img, row['sub_label_idx'], row['label_idx'], row['is_ref'], row['label_name']