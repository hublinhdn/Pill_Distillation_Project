# Giai đoạn 1: 
- Huấn luyện Teacher (Expert Model)Dữ liệu: 4 Folds của ePillID (tập trung vào ảnh Lab-shot và Consumer-shot chất lượng cao).
- Mục tiêu: Tạo ra một "không gian đặc trưng" chuẩn mực bằng ArcFace.Metric: mAP trên Fold thứ 5 của ePillID.

# Giai đoạn 2: 
- Chưng cất tri thức (Cross-Domain KD)Cơ chế: Teacher (đã đóng băng) sẽ hướng dẫn Student (MobileNetV3).
- Điểm mới: Student học cách trích xuất đặc trưng từ ảnh Consumer sao cho vector đầu ra gần khớp với vector ảnh Lab của Teacher.
- Loss: $L_{KD} = L_{Feature\_Mimicking} + L_{Metric\_ArcFace}$.
# Giai đoạn 3: 
- Kiểm tra nội bộ (Internal Validation)Dùng Student đã chưng cất chạy trên Fold thứ 5 của ePillID.
- So sánh kết quả với bài báo gốc (Baseline). Đây là lúc bạn chứng minh: "Nhờ KD, mô hình nhẹ của tôi đạt kết quả tương đương hoặc hơn mô hình nặng của bài gốc".
# Giai đoạn 4: 
- Kiểm tra xuyên miền (External Validation - OGYEIv2)Đây là phần "ăn tiền" nhất của Paper.
- Gallery: Toàn bộ folder train (40 ảnh/class) của OGYEIv2.
- Query: Folder test (6 ảnh/class) của OGYEIv2.
- Tính Rank-1 và mAP. Nếu kết quả cao, bạn chứng minh được mô hình học từ ePillID (Mỹ) có tính tổng quát hóa cực tốt sang OGYEIv2 (Châu Âu).


---
# 1. Thiết kế Dataset Wrapper (Xử lý đa nguồn)
Vì bạn có 2 bộ data khác nhau về cấu trúc (ePillID có fold, OGYEIv2 có folder), chúng ta cần một lớp PillDataLoader thông minh:

Đối với ePillID (Giai đoạn Train/Val): * Sử dụng cột fold để phân chia.
-- Teacher: Nhận diện theo cặp (Lab-shot + Consumer-shot) để học tính bất biến của dữ liệu.
-- Student: Chỉ nhận diện Consumer-shot (mô phỏng thực tế người dùng).

Đối với OGYEIv2 (Giai đoạn Test mù):
-- Tự động quét folder train làm Gallery và test làm Query.