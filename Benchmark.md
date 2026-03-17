## Tổng quan Hệ thống (System Architecture): 
- Sơ đồ luồng đi của dữ liệu: 
từ ảnh gốc => Crop => Backbone => GeM Pooling => L2 Normalization => Matching.
## benchmark simple: Try to find who is best teacher, who is student
L_SCE = 1.0        # Trọng số Softmax Cross Entropy
L_CSCE = 0.2       # Trọng số ArcFace (Benchmark dùng 0.1)
L_TRIPLET = 1.0    # Trọng số Triplet Loss
L_CONTRASTIVE = 1.0 # Trọng số Contrastive Loss

GemPooling: Single Eval (1,1, 0.2, 1)
 - resnet101           : Best mAP = 0.7764
 - convnext_base       : Best mAP = 0.7865
 - convnext_large      : Best mAP = 0.7818
 - resnet50            : Best mAP = 0.7608
 - mobilenet_v3_large  : Best mAP = 0.6680
 - resnet18            : Best mAP = 0.7131
 - efficientnet_b0     : Best mAP = 0.6788

 GemPooling: 2 side Eval (1,1, 0.2, 1)
 - resnet101           : Best mAP = 0.8099
 - convnext_base       : Best mAP = 0.8223
 - convnext_large      : Best mAP = 0.8201
 - resnet50            : Best mAP = 0.8017

 - mobilenet_v3_large  : Best mAP = 0.7106
 - resnet18            : Best mAP = 0.7512
 - efficientnet_b0     : Best mAP = 0.7249

==> Thử thêm: mô hình lớn (15 cái khác họ)
==> Thử thêm: 15 cái nhỏ: mobilenet, dennet, squeezeNet,efficientnet, channel, CBAM 

 MPN COV: 2 side Eval (1,1, 0.2, 1)
 - resnet101           : Best mAP = 0.7966
 - convnext_base       : Best mAP = 0.8486
 - convnext_large      : Best mAP = 0.8169
 - resnet50            : Best mAP = 0.8006
 - mobilenet_v3_large  : Best mAP = 0.7467
 - resnet18            : Best mAP = 0.7846
 - efficientnet_b0     : Best mAP = 0.7466

convnext_base, convnext_large: Khong load nổi với size 448x448
==> teacher - student: convnext_base - resnet18

## NEXT: teacher - student: convnext_base - resnet18
Total Loss = Loss_Student + alpha * MSE(Emb_Student, Emb_Teacher)
==> 5 teacher KD to 5 student (5x5)

1. train again student baseline with size 384
 - resnet18(1,1, 0.2, 1) + gem + 384           : Best mAP = 0.7583

2. train KD student
Thứ tự	Loại hình huấn luyện	Cấu hình chi tiết	            Best mAP	  Trạng thái
1	      Baseline	            -                             	0.7583	    Điểm sàn
2	      Phase 1 - MSE	        Alpha 50.0	                    0.7974	    Khá
3	      Phase 1 - KL Div	    Alpha 1.0	                      0.81	      Tốt
4	      Phase 2 - Cosine	    Alpha 100.0	                    0.8339	    Rất tốt
5	      Phase 3 - Hybrid	    Cosine (Alpha 30) + KL	        0.8249	    Không hiệu quả bằng
6	      Phase 2 - Cosine	    Alpha 10.0	                    0.8409	    QUÁN QUÂN

## Cross dataset domain: check on OGYEIv2 ( use validation as gallery, test as query) as unseen dataset
- dataset of OGYEIv2 is 28/6/6 as train/test/valid
- Eval cross dataset (eval/evalue_cross_dataset.py)

### 1.OGYEI valid(6) => gallery, test => query, ignore train

Baseline                  |      Map     |   R1
Student Baseline          |      78.57 % |   0.8481
Teacher Model             |      87.65 % |   0.9116
Student KD (Ours)         |      83.63 % |   0.8823

### 2.OGYEI train(28) => gallery, (test, valid) => query
Student Baseline          |      93.75 % |   0.9613
Teacher Model             |      95.83 % |   0.9744
Student KD (Ours)         |      95.61 % |   0.9707

### 3. Eval
- Eval cross dataset (eval/evalue_cross_dataset.py)
============================================================
BÁO CÁO LỖI OGYEIv2 
============================================================
Tên                       | Mô hình                   | Số lỗi    
------------------------------------------------------------
student_kd_resnet18       | resnet18                  | 27
student_baseline_resnet18 | resnet18                  | 39
teacher_convnext_base     | convnext_base             | 30





## Run train_teacher_cv.py

tmux new -s train_phase2 -d "bash -lc '
python pipelines/train_teacher_cv.py \
--backbone efficientnet_b0,resnet50,resnet101,convnext_base,convnext_large,resnet18,mobilenet_v3_large \
--pooling gem \
|& tee -a logs/train_with_shape_aware_$(date +%F_%H%M%S).log
'"


## Run train KD
Total Loss = Loss_Student + alpha * MSE(Emb_Student, Emb_Teacher)

### 🧪 Giai đoạn 1: Tìm ra Tâm Pháp Tốt Nhất (So sánh các trường phái KD)
Chạy 3 lệnh sau (sử dụng alpha tiêu chuẩn cho từng loại) để xem ResNet18 "hấp thụ" cách dạy nào tốt nhất.

1. Lệnh 1: Dạy bằng Tọa Độ (MSE) - Ép học vẹt từng con số.

python pipelines/train_student_kd.py --teacher convnext_base --teacher_weight weights/best_teacher_convnext_base_fold0.pth --student resnet18 --kd_type mse --alpha 50.0

2. Lệnh 2: Dạy bằng Hướng (Cosine) - Ép học cấu trúc không gian (Rất tốt cho Retrieval).
python pipelines/train_student_kd.py --teacher convnext_base --teacher_weight weights/best_teacher_convnext_base_fold0.pth --student resnet18 --kd_type cosine --alpha 50.0

3. Lệnh 3: Dạy bằng Logits (KL-Divergence) - Truyền sự phân vân, dark knowledge.
python pipelines/train_student_kd.py --teacher convnext_base --teacher_weight weights/best_teacher_convnext_base_fold0.pth --student resnet18 --kd_type kl --alpha 1.0 --temperature 4.0

🔥 Chấm điểm Giai đoạn 1: Kết thúc 3 lệnh này, bạn hãy so sánh best_mAP của chúng. Phương pháp nào cho mAP cao nhất sẽ là Tâm pháp chính của bạn. Giả sử cosine chiến thắng.

### 🧪 Giai đoạn 2: Dò tìm Hệ số Alpha "Điểm Ngọt" (Sweet Spot)
Chạy quét hệ số --alpha:
- Thử Alpha thấp (Cho phép tự do suy nghĩ): --alpha 10.0
- Thử Alpha trung bình (Cân bằng): --alpha 30.0
- Thử Alpha cực đoan (Ép nghe lời tuyệt đối): --alpha 100.0
### 🧪 Giai đoạn 3: Cuộc Tấn Công Cuối Cùng (The Hybrid Cocktail)
Khi đã có Alpha vàng từ Giai đoạn 2 (ví dụ alpha=30.0 là tốt nhất), hãy chạy một kịch bản Hybrid (pha trộn cả 3 Loss: MSE, Cosine và KL) để vét sạch mọi tinh hoa của Teacher.

python pipelines/train_student_kd.py \
  --teacher convnext_base \
  --teacher_weight weights/best_teacher_convnext_base_fold0.pth \
  --student resnet18 \
  --kd_type hybrid \
  --alpha 30.0 \
  --temperature 4.0

### 1 script duy nhat
tmux new -s benchmark_student_kd -d "bash -lc '
python run_kd_benchmark.py \
|& tee -a logs/benchmark_student_kd_$(date +%F_%H%M%S).log
'"

## Run eval checkpoint

tmux new -s eval_teacher -d "bash -lc '
python pipelines/evaluate_checkpoint.py \
--weight_path weights/best_teacher_efficientnet_b0_fold0.pth \
--backbone efficientnet_b0 \
--pooling gem \
|& tee -a logs/eval_teacher_$(date +%F_%H%M%S).log
'"