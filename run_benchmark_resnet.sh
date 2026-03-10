#!/bin/bash

# --- CẤU HÌNH DANH SÁCH BACKBONE ---
# BACKBONES=("resnet50" "convnext_base" "efficientnet")

DATE_STR=$(date +%d%m_%H%M)
LOG_BASE_DIR="logs/comparison_${DATE_STR}"
SESSION_NAME="benchmark_resnet"

mkdir -p "$LOG_BASE_DIR"

echo "📂 Log sẽ lưu tại: $LOG_BASE_DIR"

# Kiểm tra và xóa session cũ nếu trùng tên (để tránh chồng chéo lệnh)
tmux kill-session -t "$SESSION_NAME" 2>/dev/null

# Khởi tạo tmux session mới ở chế độ nền
tmux new-session -d -s "$SESSION_NAME"

echo "🟡 Đang nạp lệnh ..."

# Định nghĩa lệnh chạy
CMD="python pipelines/train_teacher_cv.py  --fold 0 --epochs 100 2>&1 | tee $LOG_BASE_DIR/fold0.txt"

# Gửi từng phần để tránh lỗi command
tmux send-keys -t "$SESSION_NAME" "echo '----------------------------------------'" Enter
tmux send-keys -t "$SESSION_NAME" "$CMD" Enter
tmux send-keys -t "$SESSION_NAME" "echo '✅ HOÀN THÀNH'" Enter

echo "------------------------------------------------"
echo "✅ Đã kích hoạt xong. Hãy kiểm tra bằng lệnh:"
echo "tmux attach -t $SESSION_NAME"