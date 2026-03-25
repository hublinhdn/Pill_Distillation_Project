import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def load_data(json_file='kd_results.json'):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def plot_grouped_bar_chart(data, output_dir):
    """Vẽ biểu đồ Bar Chart so sánh Baseline vs KD mAP, có đường nét đứt của Teacher"""
    pair_names = [item['pair_name'] for item in data]
    baseline_maps = [item['student']['baseline_map'] for item in data]
    kd_maps = [item['student']['kd_map'] for item in data]
    teacher_maps = [item['teacher']['map'] for item in data]

    x = np.arange(len(pair_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 7))
    rects1 = ax.bar(x - width/2, baseline_maps, width, label='Student Baseline', color='#B0BEC5', edgecolor='black')
    rects2 = ax.bar(x + width/2, kd_maps, width, label='Student w/ KD', color='#4CAF50', edgecolor='black')

    # Vẽ đường ngang biểu thị Teacher mAP cho từng cặp
    for i in range(len(x)):
        ax.hlines(y=teacher_maps[i], xmin=x[i]-0.4, xmax=x[i]+0.4, color='#F44336', linestyles='dashed', linewidth=2)
        if i == 0: # Thêm label 1 lần để hiện trong chú thích (legend)
            ax.hlines(y=teacher_maps[i], xmin=x[i]-0.4, xmax=x[i]+0.4, color='#F44336', linestyles='dashed', linewidth=2, label='Teacher mAP')

    # Định dạng trục
    ax.set_ylabel('mAP Score', fontsize=12, fontweight='bold')
    ax.set_title('Knowledge Distillation Performance Gain', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(pair_names, rotation=25, ha="right", fontsize=10)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(0.2, 1.0) # Tùy chỉnh giới hạn trục Y cho đẹp

    # Gắn nhãn số lên đỉnh cột
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, rotation=0)

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'kd_barchart_comparison.png'), dpi=300)
    print("✅ Đã xuất biểu đồ Bar Chart!")
    plt.close()

def plot_pareto_arrows(data, output_dir):
    """Vẽ biểu đồ Pareto (Tham số vs mAP) với mũi tên tiến hóa từ Baseline lên KD"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Lấy dữ liệu
    for item in data:
        t_params = item['teacher']['params_m']
        t_map = item['teacher']['map']
        
        s_params = item['student']['params_m']
        s_base_map = item['student']['baseline_map']
        s_kd_map = item['student']['kd_map']
        name = item['student']['name']
        
        # Vẽ Teacher
        ax.scatter(t_params, t_map, color='#F44336', s=200, marker='*', zorder=5)
        
        # Vẽ Student Baseline
        ax.scatter(s_params, s_base_map, color='#9E9E9E', s=80, alpha=0.7)
        
        # Vẽ Student KD
        color_kd = '#4CAF50' if s_kd_map >= s_base_map else '#FF9800' # Xanh nếu tăng, Cam nếu giảm (Negative Transfer)
        ax.scatter(s_params, s_kd_map, color=color_kd, s=120, edgecolors='black', zorder=4)
        
        # Vẽ MŨI TÊN (Từ Baseline lên KD)
        ax.annotate('', xy=(s_params, s_kd_map), xytext=(s_params, s_base_map),
                    arrowprops=dict(facecolor=color_kd, shrink=0.05, width=1.5, headwidth=8), zorder=3)
        
        # Gắn tên cho điểm KD
        ax.text(s_params + 1, s_kd_map, name, fontsize=9, va='center')

    # Trick để hiển thị Legend đúng
    ax.scatter([], [], color='#F44336', s=150, marker='*', label='Teacher Models')
    ax.scatter([], [], color='#9E9E9E', s=80, label='Student Baseline')
    ax.scatter([], [], color='#4CAF50', s=100, edgecolors='black', label='Student After KD')

    # Định dạng
    ax.set_xscale('log') # RẤT QUAN TRỌNG: Scale logarit cho số tham số
    ax.set_xlabel('Number of Parameters (Millions - Log Scale)', fontsize=12, fontweight='bold')
    ax.set_ylabel('mAP Score', fontsize=12, fontweight='bold')
    ax.set_title('Knowledge Distillation: Parameter vs. Accuracy Evolution', fontsize=16, fontweight='bold')
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'kd_pareto_evolution.png'), dpi=300)
    print("✅ Đã xuất biểu đồ Pareto Evolution!")
    plt.close()

if __name__ == "__main__":
    output_directory = 'reports/final_charts'
    os.makedirs(output_directory, exist_ok=True)
    
    # Đọc dữ liệu từ file JSON
    dataset = load_data('results/kd_results.json')
    
    # Vẽ biểu đồ
    plot_grouped_bar_chart(dataset, output_directory)
    plot_pareto_arrows(dataset, output_directory)