import matplotlib.pyplot as plt
import pandas as pd
import os

# 1. Chuẩn bị dữ liệu (Bao gồm mAP và Parameters)
# Số liệu Parameters (ước tính) dựa trên các backbone chuẩn trong PyTorch
data = [
    {'Backbone': 'efficientnet_b5',     'mAP': 0.8618, 'Params_M': 30}, # Triệu
    {'Backbone': 'convnextv2_base.fcmae_ft_in22k_in1k_384_timm',     'mAP': 0.8251, 'Params_M': 89},
    {'Backbone': 'convnext_base_timm',     'mAP': 0.8207, 'Params_M': 89},
    {'Backbone': 'resnest101e_timm',     'mAP': 0.8157, 'Params_M': 48},
    {'Backbone': 'seresnext101_32x4d_timm',     'mAP': 0.7986, 'Params_M': 49},
    {'Backbone': 'convnext_large_timm',     'mAP': 0.7983, 'Params_M': 198},
    {'Backbone': 'resnet34_tv',     'mAP': 0.7978, 'Params_M': 21},
    {'Backbone': 'densenet161_tv',     'mAP': 0.7924, 'Params_M': 28},
    {'Backbone': 'repvgg_a0_timm',     'mAP': 0.7913, 'Params_M': 8.3},
    {'Backbone': 'edgenext_small_timm',     'mAP': 0.7913, 'Params_M': 5.6},
    {'Backbone': 'convnext_femto_timm',     'mAP': 0.7899, 'Params_M': 5.2},
    {'Backbone': 'tresnet_l_timm',     'mAP': 0.7871, 'Params_M': 55},
    {'Backbone': 'resnet101_tv',     'mAP': 0.7853, 'Params_M': 44},
    {'Backbone': 'convnext_atto_timm',     'mAP': 0.7804, 'Params_M': 3.7},
    {'Backbone': 'convnextv2_atto_timm',     'mAP': 0.7800, 'Params_M': 3.7},
    {'Backbone': 'efficientnet_b2_timm',     'mAP': 0.7612, 'Params_M': 9.1},
    {'Backbone': 'tf_efficientnetv2_l.in21k_ft_in1k',     'mAP': 0.7575, 'Params_M': 118},
    {'Backbone': 'mobilenetv3_large_100_tv',     'mAP': 0.7254, 'Params_M': 5.4},
    {'Backbone': 'maxvit_base_tf_384_timm',     'mAP': 0.7220, 'Params_M': 119},
    {'Backbone': 'resnet18_tv',     'mAP': 0.7168, 'Params_M': 11},
    {'Backbone': 'efficientnet_b1_timm',     'mAP': 0.7051, 'Params_M': 7.8},
    {'Backbone': 'mobilenetv3_large_100_timm',     'mAP': 0.6873, 'Params_M': 5.4},
    {'Backbone': 'mobilenetv2_100_timm',     'mAP': 0.6826, 'Params_M': 3.5},
    {'Backbone': 'efficientnet_b0_tv',     'mAP': 0.6820, 'Params_M': 5.3},
    {'Backbone': 'resnet101.a1h_in1k_timm',     'mAP': 0.6328, 'Params_M': 44},
    {'Backbone': 'mobilevit_s_timm',     'mAP': 0.5895, 'Params_M': 5.6},
    {'Backbone': 'squeezenet1_1',     'mAP': 0.588, 'Params_M': 1.2},
    {'Backbone': 'mobilenetv2_100_tv',     'mAP': 0.5731, 'Params_M': 3.5},
    {'Backbone': 'ghostnet_100_timm',     'mAP': 0.5237, 'Params_M': 5.2},
    {'Backbone': 'shufflenet_v2_x1_0_tv',     'mAP': 0.4964, 'Params_M': 2.3},
]

def draw(df, title='Hiệu năng (mAP) và Độ phức tạp (Params) của các Backbone', output_img='benchmark_teacher.png', output_dir='reports'):
    # 2. Khởi tạo biểu đồ
    fig, ax = plt.subplots(figsize=(12, 7))

    # Vẽ biểu đồ cột mAP
    colors = ['#2ca02c' if mAP > 0.8 else '#ff7f0e' if mAP > 0.65 else '#1f77b4' for mAP in df['mAP']]
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
                f'({param_m}M)', 
                ha='center', va='top', fontsize=9, color='white', fontweight='medium')

    # 4. Tùy chỉnh giao diện (Styling)
    ax.set_title(title, fontsize=16, pad=25)
    ax.set_ylabel('Best mAP', fontsize=13)
    ax.set_xlabel('Backbone Architectures', fontsize=13)
    plt.xticks(rotation=30, ha='right')

    # Tùy chỉnh trục Y để thấy sự khác biệt
    ax.set_ylim(0.6, 0.85) 
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # 5. Hiển thị
    plt.tight_layout()
    # plt.show()

    plt.savefig(os.path.join(output_dir, output_img), dpi=300)
    print(f'DONE for {output_img}')

# Chuyển dữ liệu thành DataFrame để dễ xử lý
df = pd.DataFrame(data)

# Sắp xếp theo mAP giảm dần
df = df.sort_values(by='mAP', ascending=False)
df_teacher = df[df['Params_M'] >= 20]
df_student = df[df['Params_M'] < 20]
draw(df_teacher, title='Hiệu năng (mAP) và Độ phức tạp (Params) của các Larger Backbone', output_img='benchmark_teacher.png')
draw(df_student, title='Hiệu năng (mAP) và Độ phức tạp (Params) của các Small Backbone', output_img='benchmark_student.png')
print('DONE')