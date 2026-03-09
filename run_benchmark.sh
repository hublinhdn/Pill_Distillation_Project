#!/bin/bash

# Kiểm tra nếu thiếu tham số truyền vào
if [ -z "$1" ]; then
    echo "Sử dụng: bash run_benchmark.sh [backbone_name]"
    echo "Ví dụ: bash run_benchmark.sh convnext_base"
    exit 1
fi

BACKBONE=$1
DATE_STR=$(date +%d%m_%H%M)
LOG_DIR="logs/${BACKBONE}_${DATE_STR}"
SESSION_NAME="train_${BACKBONE}"

# Tạo thư mục log
mkdir -p "$LOG_DIR"

echo "🚀 Bắt đầu tiến trình Benchmark cho: $BACKBONE"
echo "📂 Log sẽ được lưu tại: $LOG_DIR"
echo "🖥️  Tmux session: $SESSION_NAME"

# Khởi tạo tmux session mới ở chế độ nền (detached)
tmux new-session -d -s "$SESSION_NAME"

# Gửi lệnh chạy lần lượt 4 Fold (0, 1, 2, 3) - Fold 4 để dành làm Hold-out
tmux send-keys -t "$SESSION_NAME" "for f in 0 1 2 3; do \
    echo \"--- Đang chạy Fold \$f ---\"; \
    python pipelines/train_teacher_cv.py --backbone $BACKBONE --fold \$f --epochs 100 2>&1 | tee $LOG_DIR/fold_\$f.txt; \
done" C-m

echo "✅ Đã kích hoạt tmux. Để theo dõi quá trình, hãy dùng lệnh:"
echo "   tmux attach -t $SESSION_NAME"