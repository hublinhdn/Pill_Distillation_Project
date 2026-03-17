import matplotlib.pyplot as plt
import pandas as pd
import os

# 1. Chuẩn bị dữ liệu (Bao gồm mAP và Parameters)
# Số liệu Parameters (ước tính) dựa trên các backbone chuẩn trong PyTorch
data = [
    {'Backbone': 'convnext_base',     'mAP': 0.8015, 'Params_M': 88.6}, # Triệu
    {'Backbone': 'convnext_large',    'mAP': 0.7873, 'Params_M': 197.8}, # Triệu
    {'Backbone': 'resnet101',        'mAP': 0.7447, 'Params_M': 44.5}, # Triệu
    {'Backbone': 'resnet50',         'mAP': 0.7089, 'Params_M': 25.6}, # Triệu
    {'Backbone': 'resnet18',         'mAP': 0.6909, 'Params_M': 11.7}, # Triệu
    {'Backbone': 'efficientnet_b0',   'mAP': 0.6839, 'Params_M': 5.3}, # Triệu
    {'Backbone': 'mobilenet_v3_large','mAP': 0.6799, 'Params_M': 5.5} # Triệu
]

# Chuyển dữ liệu thành DataFrame để dễ xử lý
df = pd.DataFrame(data)

# Sắp xếp theo mAP giảm dần
df = df.sort_values(by='mAP', ascending=False)

# 2. Khởi tạo biểu đồ
fig, ax = plt.subplots(figsize=(12, 7))

# Vẽ biểu đồ cột mAP
colors = ['#1f77b4' if 'convnext' in b else '#ff7f0e' if 'resnet' in b else '#2ca02c' for b in df['Backbone']]
bars = ax.bar(df['Backbone'], df['mAP'], color=colors, alpha=0.8, edgecolor='black', width=0.7)

# 3. Thêm chú thích cho từng cột (mAP và Params)
for bar, param_m in zip(bars, df['Params_M']):
    yval = bar.get_height()
    
    # Text mAP (in đậm, to hơn)
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.005, 
            f'{yval:.4f}', 
            ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')
    
    # Text Parameters (in thường, nhỏ hơn, bên dưới mAP)
    ax.text(bar.get_x() + bar.get_width()/2, yval - 0.02, # Đặt text vào bên trong cột
            f'({param_m}M Params)', 
            ha='center', va='top', fontsize=9, color='white', fontweight='medium')

# 4. Tùy chỉnh giao diện (Styling)
ax.set_title('Hiệu năng (mAP) và Độ phức tạp (Params) của các Backbone', fontsize=16, pad=25)
ax.set_ylabel('Best mAP', fontsize=13)
ax.set_xlabel('Backbone Architectures', fontsize=13)
plt.xticks(rotation=30, ha='right')

# Tùy chỉnh trục Y để thấy sự khác biệt
ax.set_ylim(0.6, 0.85) 
ax.grid(axis='y', linestyle='--', alpha=0.5)

# 5. Hiển thị
plt.tight_layout()
# plt.show()

output_dir = 'reports'
output_img_path = 'benchmark_teacher.png'
plt.savefig(os.path.join(output_dir, output_img_path), dpi=300)
print('DONE')