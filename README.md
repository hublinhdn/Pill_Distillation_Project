# Pill Identification via Knowledge Distillation (Pill-KD)

Dự án nghiên cứu về nhận diện thuốc qua ảnh chụp thực tế bằng kỹ thuật **Knowledge Distillation (Chưng cất tri thức)**. Mục tiêu là huấn luyện một mô hình nhỏ (**MobileNetV3**) đạt hiệu năng tiệm cận mô hình lớn (**EfficientNet-B3**) nhưng tối ưu cho thiết bị di động.

## 📂 Cấu trúc dự án
```text
.
├── models/             # Định nghĩa kiến trúc Teacher & Student
├── pipelines/          # Các kịch bản huấn luyện và đánh giá
├── utils/              # Tiện ích xử lý dữ liệu, dataset và đánh giá
├── data/               # [Ignore] Nơi chứa dataset ePillID
├── weights/            # [Ignore] Nơi lưu trữ trọng số (.pth)
├── results/            # Nơi lưu trữ file CSV và báo cáo đánh giá
├── README.md           # Hướng dẫn dự án
└── .gitignore          # Cấu hình bỏ qua file rác và dữ liệu nặng
```
## benchmark simple
L_SCE = 1.0        # Trọng số Softmax Cross Entropy
L_CSCE = 0.2       # Trọng số ArcFace (Benchmark dùng 0.1)
L_TRIPLET = 1.0    # Trọng số Triplet Loss
L_CONTRASTIVE = 1.0 # Trọng số Contrastive Loss

resnet50 (1, 0.1, 1, 1): 0.61
resnet50(1,1, 0.2, 1) + GemPooling + m=0.35 (arcface) + CosineAnnealingLR 
+ size 384 + correct CE embedding + correct class number count: .6763
resnet50(1,1, 1, 1) + GemPooling + m=0.35 (arcface) + CosineAnnealingLR 
+ size 384 + correct CE embedding + 100 epoch: 0.65 ==> rollback

resnet50(1,1, 1, 1) + GemPooling + SubCenterArcFace + 60 epoch: 0.53

convnext_base(1,1, 1, 1) + GemPooling + m=0.35 (arcface) + CosineAnnealingLR 
+ size 384 + correct CE embedding + 100 epoch: 0.762

GemPooling: Single Eval (1,1, 0.2, 1)
 - resnet101           : Best mAP = 0.7764
 - convnext_base       : Best mAP = 0.7865
 - convnext_large      : Best mAP = 0.7818
 - resnet50            : Best mAP = 0.7608
 - mobilenet_v3_large  : Best mAP = 0.6680
 - resnet18            : Best mAP = 0.7131
 - efficientnet_b0     : Best mAP = 0.6788


convnext_base, convnext_large: Khong load nổi với size 448x448

Next step: two side eval with gem / mpncov

## Run train_teacher_cv.py

tmux new -s train_gem -d "bash -lc '
python pipelines/train_teacher_cv.py \
--backbone resnet101,convnext_base,convnext_large,resnet50,mobilenet_v3_large,resnet18,efficientnet_b0 \
--pooling gem \
|& tee -a logs/train_gem_$(date +%F_%H%M%S).log
'"

## Run eval checkpoint

tmux new -s eval_teacher -d "bash -lc '
python pipelines/evaluate_checkpoint.py \
--weight_path weights/best_teacher_convnext_base_fold0.pth \
--backbone convnext_base \
--pooling gem \
|& tee -a logs/eval_teacher_$(date +%F_%H%M%S).log
'"