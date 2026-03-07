import pandas as pd
import os
from datetime import datetime

class ExperimentLogger:
    def __init__(self, log_dir="reports", exp_name="Teacher_Training"):
        self.log_dir = log_dir
        self.exp_name = exp_name
        os.makedirs(log_dir, exist_ok=True)
        
        # Tạo file name theo thời gian để không ghi đè các lần chạy trước
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(log_dir, f"{exp_name}_{timestamp}.csv")
        self.summary_path = os.path.join(log_dir, f"{exp_name}_summary.md")
        
        self.results = []

    def log_fold(self, fold_idx, metrics):
        """Lưu kết quả của từng Fold"""
        log_data = {"fold": fold_idx, **metrics, "timestamp": datetime.now()}
        self.results.append(log_data)
        
        # Cập nhật file CSV sau mỗi fold (đề phòng crash máy)
        df = pd.DataFrame(self.results)
        df.to_csv(self.csv_path, index=False)
        print(f"📊 Đã ghi log Fold {fold_idx} vào {self.csv_path}")

    def save_final_summary(self):
        """Tính toán trung bình cộng và ghi vào file Markdown"""
        df = pd.DataFrame(self.results)
        # Chỉ lấy các cột số để tính toán
        numeric_cols = df.select_dtypes(include=['number']).columns
        mean_results = df[numeric_cols].mean()
        std_results = df[numeric_cols].std()

        with open(self.summary_path, "a") as f:
            f.write(f"\n## Experiment: {self.exp_name} ({datetime.now()})\n")
            f.write("| Metric | Mean | Std |\n")
            f.write("| :--- | :---: | :---: |\n")
            for col in numeric_cols:
                if col != 'fold':
                    f.write(f"| {col} | {mean_results[col]:.2f}% | {std_results[col]:.2f}% |\n")
            f.write("\n" + "="*40 + "\n")
        
        print(f"🏆 Đã tạo bảng tổng hợp tại: {self.summary_path}")