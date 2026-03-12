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
    
    # Ép kiểu pilltype_id thành string ngay từ khi load
    df_all = pd.read_csv(all_csv, dtype={'pilltype_id': str})
    # Dọn dẹp chuỗi (đề phòng pandas tự thêm đuôi .0 vào cuối mã ID)
    df_all['pilltype_id'] = df_all['pilltype_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
    
    # 2. XỬ LÝ NHÃN (LABEL ENCODING)
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            # Loại bỏ khoảng trắng hoặc ký tự xuống dòng thừa
            classes = [line.strip() for line in f.readlines() if line.strip()]
        
        # Lấy danh sách nhãn thực tế từ CSV
        actual_labels = df_all['pilltype_id'].unique().tolist()

        # Lọc ra danh sách khớp và danh sách thừa
        matched_labels = [l for l in actual_labels if l in classes]
        extra_labels = [l for l in actual_labels if l not in classes]

        full_classes = classes + sorted(extra_labels)
        df_all['label_idx'] = pd.Categorical(df_all['pilltype_id'], categories=full_classes).codes

        # IN LOG CHUẨN XÁC: Cho biết có bao nhiêu nhãn thực sự khớp
        print(f"📊 Đã khớp {len(matched_labels)}/{len(classes)} nhãn chuẩn bài báo.")
        if len(extra_labels) > 0:
            print(f"➕ Đã bổ sung {len(extra_labels)} nhãn mới phát hiện trong CSV.")
        
        # # Tìm các nhãn có trong CSV nhưng thiếu trong file .txt
        # extra_labels = [l for l in actual_labels if l not in classes]
        
        # # Gộp lại: Nhãn chuẩn của bài báo đứng trước, nhãn mới đứng sau
        # full_classes = classes + sorted(extra_labels)
        
        # Gán label_idx
        # df_all['label_idx'] = pd.Categorical(df_all['label_code_id'], categories=full_classes).codes
        
        # print(f"📊 Đã khớp {len(classes)} nhãn chuẩn bài báo.")
        # if len(extra_labels) > 0:
        #     print(f"➕ Đã bổ sung {len(extra_labels)} nhãn mới phát hiện trong CSV.")
    else:
        print(f"⚠️ Không tìm thấy {label_path}. Đang tạo nhãn bằng factorize.")
        df_all['label_idx'], _ = pd.factorize(df_all['pilltype_id'])

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

    # 1. Tạo chuỗi kết hợp ID thuốc và mặt trước/sau (ví dụ: "105_1", "105_0")
    df_all['sub_label_raw'] = df_all['label_idx'].astype(str) + "_" + df_all['is_front'].astype(str)
    # 2. pd.factorize tự động đánh số lại chuỗi trên thành các số nguyên liên tục: 0, 1, 2, ..., N-1
    df_all['sub_label_idx'] = pd.factorize(df_all['sub_label_raw'])[0]
    
    # Kiểm tra kết quả cuối cùng
    total_samples = len(df_all)
    ref_count = (df_all['is_ref'] == 1).sum()
    cons_count = (df_all['is_ref'] == 0).sum()

    total_classes = df_all['label_idx'].nunique()
    total_sub_classes = df_all['sub_label_idx'].nunique()
    
    print(f"✅ Load thành công: Total {total_samples} | Ref: {ref_count} | Cons: {cons_count} | total_classes: {total_classes} | total_sub_classes:{total_sub_classes}")
    return df_all

if __name__ == "__main__":
    df = load_epill_full_data()
    if len(df) > 0:
        print(df[['image_path', 'pilltype_id', 'label_idx', 'is_ref', 'fold']].head())
    else:
        print("❌ Lỗi: DataFrame vẫn trống!")