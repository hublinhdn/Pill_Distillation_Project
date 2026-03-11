import pandas as pd
import os

def load_epill_full_data():
    """
    Load dữ liệu ePillID và gán nhãn cố định. 
    Đảm bảo 4902 nhãn đầu tiên khớp hoàn toàn với bài báo.
    """
    root_path = 'data/raw/ePillID'
    folds_dir = os.path.join(root_path, 'folds/pilltypeid_nih_sidelbls0.01_metric_5folds/base')
    label_path = os.path.join(folds_dir, 'pill_classes.txt')
    
    # 1. Load file MASTER
    all_csv = os.path.join(folds_dir, 'pilltypeid_nih_sidelbls0.01_metric_5folds_all.csv')
    if not os.path.exists(all_csv):
        raise FileNotFoundError(f"Không tìm thấy file: {all_csv}")
    
    # Ép kiểu label_code_id thành string ngay từ khi load
    df_all = pd.read_csv(all_csv, dtype={'label_code_id': str})
    
    # 2. XỬ LÝ NHÃN (LABEL ENCODING)
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            # Loại bỏ khoảng trắng hoặc ký tự xuống dòng thừa
            classes = [line.strip() for line in f.readlines() if line.strip()]
        
        # Lấy danh sách nhãn thực tế từ CSV
        actual_labels = df_all['label_code_id'].unique().tolist()
        
        # Tìm các nhãn có trong CSV nhưng thiếu trong file .txt
        extra_labels = [l for l in actual_labels if l not in classes]
        
        # Gộp lại: Nhãn chuẩn của bài báo đứng trước, nhãn mới đứng sau
        full_classes = classes + sorted(extra_labels)
        
        # Gán label_idx
        df_all['label_idx'] = pd.Categorical(df_all['label_code_id'], categories=full_classes).codes
        
        print(f"📊 Đã khớp {len(classes)} nhãn chuẩn bài báo.")
        if len(extra_labels) > 0:
            print(f"➕ Đã bổ sung {len(extra_labels)} nhãn mới phát hiện trong CSV.")
    else:
        print(f"⚠️ Không tìm thấy {label_path}. Đang tạo nhãn bằng factorize.")
        df_all['label_idx'], _ = pd.factorize(df_all['label_code_id'])

    # 3. GÁN FOLD (Giữ nguyên logic chuẩn của bài báo)
    df_all['fold'] = -1 
    for i in range(5):
        fold_csv_path = os.path.join(folds_dir, f'pilltypeid_nih_sidelbls0.01_metric_5folds_{i}.csv')
        if os.path.exists(fold_csv_path):
            df_f = pd.read_csv(fold_csv_path, dtype={'image_path': str})
            # Gán fold dựa trên danh sách ảnh trong từng file fold chuẩn
            df_all.loc[df_all['image_path'].isin(df_f['image_path']), 'fold'] = i
            
    # 4. CHUẨN HÓA is_ref
    df_all['is_ref'] = df_all['is_ref'].map({True: 1, False: 0, 1: 1, 0: 0, 'True': 1, 'False': 0}).astype(int)
    df_all['is_front'] = df_all['is_front'].map({True: 1, False: 0, 1: 1, 0: 0, 'True': 1, 'False': 0}).astype(int)

       # THÊM DÒNG NÀY: Tạo nhãn phụ phân biệt mặt trước/sau
    # Nếu label_idx là 10: 
    #   - Mặt trước (is_front=1) -> sub_label = 21
    #   - Mặt sau (is_front=0)   -> sub_label = 20
    df_all['sub_label_idx'] = df_all['label_idx'] * 2 + df_all['is_front'].astype(int)
    
    # Kiểm tra kết quả cuối cùng
    total_samples = len(df_all)
    ref_count = (df_all['is_ref'] == 1).sum()
    cons_count = (df_all['is_ref'] == 0).sum()
    
    print(f"✅ Load thành công: Total {total_samples} | Ref: {ref_count} | Cons: {cons_count}")
    return df_all

if __name__ == "__main__":
    df = load_epill_full_data()
    if len(df) > 0:
        print(df[['image_path', 'label_code_id', 'label_idx', 'is_ref', 'fold']].head())
    else:
        print("❌ Lỗi: DataFrame vẫn trống!")