import os
import pandas as pd
import numpy as np
from tqdm import tqdm

def get_info_from_yolo_seg(label_path):
    """Trích xuất BBox từ chuỗi Polygon YOLO Seg."""
    if not os.path.exists(label_path):
        return None, None
    
    with open(label_path, 'r') as f:
        line = f.readline().strip()
    
    if not line:
        return None, None

    parts = line.split()
    # format: class_id x1 y1 x2 y2 ...
    coords = list(map(float, parts[1:]))
    
    xs = coords[0::2]
    ys = coords[1::2]
    
    bbox = [min(xs), min(ys), max(xs), max(ys)] # [xmin, ymin, xmax, ymax]
    return bbox, coords

def generate_ogyei_manifest(root_dir, output_csv):
    data = []
    # Cấu trúc: data/raw/OGYEIv2/ogyeiv2/ogyeiv2/[train, valid, test]
    splits = ['train', 'valid', 'test']
    
    for split in splits:
        split_path = os.path.join(root_dir, split)
        img_dir = os.path.join(split_path, 'images')
        label_dir = os.path.join(split_path, 'labels')
        
        if not os.path.exists(img_dir):
            print(f"⚠️ Không tìm thấy: {img_dir}")
            continue

        files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_file in tqdm(files, desc=f"Processing {split}"):
            # Parse tên file: {pill_name}_{s/u}_{number}.jpg
            # Ví dụ: "Paracetamol_s_1.jpg" -> pill_name = "Paracetamol"
            pill_name = "_".join(img_file.split('_')[:-2]) 
            
            img_path = os.path.join(img_dir, img_file)
            # Nhãn tương ứng: labels/{img_name}.txt
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(label_dir, label_file)
            
            bbox, polygon = get_info_from_yolo_seg(label_path)
            
            data.append({
                'pillId': f"OGYEI_{pill_name}", # Định danh class
                'image_path': os.path.abspath(img_path),
                'split': split,
                'source': 'OGYEIv2',
                'bbox_norm': str(bbox) if bbox else None,
                'polygon_norm': str(polygon) if polygon else None
            })

    df = pd.DataFrame(data)
    # Tạo thư mục lưu CSV nếu chưa có
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Đã tạo xong Manifest: {output_csv}")
    print(f"📊 Thống kê: {df['pillId'].nunique()} classes, {len(df)} images.")

if __name__ == "__main__":
    # Sử dụng đường dẫn tuyệt đối dựa trên vị trí script để tránh nhầm lẫn
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Khớp chính xác với cấu trúc folder bạn mô tả
    RAW_DATA = os.path.join(project_root, 'data', 'raw', 'OGYEIv2', 'ogyeiv2', 'ogyeiv2')
    OUTPUT_CSV = os.path.join(project_root, 'data', 'processed', 'ogyei_manifest.csv')
    
    generate_ogyei_manifest(RAW_DATA, OUTPUT_CSV)