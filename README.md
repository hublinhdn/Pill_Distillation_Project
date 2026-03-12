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
resnet50(1,0.2, 1, 1) + mpn cov + m=0.5 (arcface): 0.65
resnet50(1,1, 1, 1) + GemPooling + m=0.5 (arcface): 0.707
resnet50(1,1, 1, 1) + GemPooling + m=0.35 (arcface) + OneCycleLR: 

## Run train_teacher_cv.py

tmux new -s train_teacher_cv -d "bash -lc '
python pipelines/train_teacher_cv.py \
|& tee -a logs/train_teacher_cv_$(date +%F_%H%M%S).log
'"


## Run train_student_baseline.py

tmux new -s train_student_baseline -d "bash -lc '
python pipelines/train_student_baseline.py \
|& tee -a logs/train_student_baseline_$(date +%F_%H%M%S).log
'"

## Run train_student_kd.py

tmux new -s train_student_kd -d "bash -lc '
python pipelines/train_student_kd.py \
|& tee -a logs/train_student_kd_$(date +%F_%H%M%S).log
'"

## Run eval_teacher.py

tmux new -s eval_teacher -d "bash -lc '
python pipelines/eval_teacher.py \
|& tee -a logs/eval_teacher_$(date +%F_%H%M%S).log
'"