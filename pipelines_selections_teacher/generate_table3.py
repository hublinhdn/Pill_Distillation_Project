import pandas as pd
import os

def main():
    # Đường dẫn file log KD đầu vào và file CSV đầu ra
    kd_log_path = 'reports_kd/KD_Matrix_Experiment_Summary.txt'
    output_csv = 'reports_kd/Table3_KD_Results.csv'

    print("🚀 BẮT ĐẦU TRÍCH XUẤT VÀ TỔNG HỢP BẢNG 3 (KD RESULTS)")
    
    if not os.path.exists(kd_log_path):
        print(f"❌ Lỗi: Không tìm thấy file {kd_log_path}. Vui lòng đảm bảo script KD đã chạy.")
        return

    # Tích hợp sẵn Bảng 1 (Baseline) để đối chiếu
    baselines = {
        'resnet18_tv': {'map_base': 0.7332, 'delta_base': 0.2668},
        'mobilenetv3_large_100_tv': {'map_base': 0.7202, 'delta_base': 0.2798},
        'efficientnet_b1_timm': {'map_base': 0.7813, 'delta_base': 0.2187},
        'ghostnet_100_timm': {'map_base': 0.4511, 'delta_base': 0.5489}
    }

    records = []
    
    # Đọc và parse file log
    with open(kd_log_path, 'r') as f:
        for line in f:
            # Bỏ qua các dòng gạch ngang hoặc dòng header
            if '|' not in line or 'mAP(Cons)' in line:
                continue

            # Cắt chuỗi theo dấu '|' và xóa khoảng trắng thừa
            parts = [p.strip() for p in line.split('|')]
            
            # Cấu trúc của train_student_kd.py: 
            # Teacher [0] | Student [1] | Alpha [2] | Temp [3] | Type [4] | Loss [5] | mAP(Cons) [6] | mAP(Ref) [7] | Delta [8]
            if len(parts) >= 9:
                teacher = parts[0]
                student = parts[1]
                
                try:
                    map_kd = float(parts[6])
                    delta_kd = float(parts[8])
                except ValueError:
                    continue # Bỏ qua nếu không parse được số

                if student in baselines:
                    map_base = baselines[student]['map_base']
                    delta_base = baselines[student]['delta_base']
                    
                    # Tính toán Gain (Sự cải thiện)
                    gain = map_kd - map_base

                    records.append({
                        'Student': student,
                        'Teacher': teacher,
                        'mAP(Cons) Baseline': map_base,
                        'mAP(Cons) KD': map_kd,
                        'Gain (mAP)': round(gain, 4),
                        'Delta mAP (Baseline)': delta_base,
                        'Delta mAP (KD)': delta_kd
                    })

    if not records:
        print("⚠️ Không tìm thấy dữ liệu hợp lệ trong file log.")
        return

    # Chuyển thành DataFrame
    df = pd.DataFrame(records)
    
    # 🌟 Sắp xếp thông minh: Gom theo từng Học sinh, rồi xếp Giáo viên từ dạy giỏi nhất xuống kém nhất
    df = df.sort_values(by=['Student', 'Gain (mAP)'], ascending=[True, False]).reset_index(drop=True)
    
    # Lưu ra file CSV
    os.makedirs('reports_kd', exist_ok=True)
    df.to_csv(output_csv, index=False)
    
    print("\n" + "="*80)
    print("🏆 BẢNG 3: KẾT QUẢ KNOWLEDGE DISTILLATION (Preview Top 10)")
    print("="*80)
    print(df.head(10).to_string(index=False))
    print("="*80)
    print(f"✅ Đã trích xuất thành công {len(df)} tổ hợp KD.")
    print(f"📁 Dữ liệu đầy đủ (CSV) đã được lưu tại: {output_csv}")

if __name__ == '__main__':
    main()