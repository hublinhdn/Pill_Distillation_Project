import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_comparative_report(exp_names):
    """
    exp_names: list các tên experiment (ví dụ: ['Teacher_Mixed_CV', 'Student_KD_Mixed_CV'])
    """
    summary_data = []

    for name in exp_names:
        log_path = f'logs/{name}/final_summary.csv' # Giả định logger lưu ở đây
        if os.path.exists(log_path):
            df = pd.read_csv(log_path)
            # Lấy giá trị trung bình của 5 folds
            mean_metrics = {
                'Experiment': name,
                'mAP': df['mAP'].mean(),
                'Rank-1': df['Rank-1'].mean(),
                'Rank-5': df['Rank-5'].mean()
            }
            summary_data.append(mean_metrics)
    
    report_df = pd.DataFrame(summary_data)
    
    # Xuất bảng Markdown để bạn copy thẳng vào Paper
    print("\n📝 TABLE FOR PAPER (Markdown format):")
    print(report_df.to_markdown(index=False))
    
    # Vẽ biểu đồ so sánh
    report_df.plot(kind='bar', x='Experiment', y=['mAP', 'Rank-1'], figsize=(10, 6))
    plt.title("Comparison: Teacher vs Baseline vs KD")
    plt.ylabel("Score")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('results/performance_comparison.png')
    print("\n📊 Chart saved to results/performance_comparison.png")

if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    # Sau khi bạn chạy xong cả 3 script CV, hãy chạy lệnh này:
    generate_comparative_report(['Teacher_Mixed_CV', 'Student_Baseline_Mixed_CV', 'Student_KD_Mixed_CV'])