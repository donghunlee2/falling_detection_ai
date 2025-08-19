#!/usr/bin/env bash
set -euo pipefail

# ===== 경로 설정 =====
DET_CFG="/root/dhyee/mmpose/projects/rtmpose3d/demo/rtmdet_m_640-8xb32_coco-person.py"
DET_CKPT="/root/dhyee/mmpose/projects/rtmpose3d/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
POSE_CFG="/root/dhyee/mmpose/projects/rtmpose3d/configs/rtmw3d-l_8xb64_cocktail14-384x288.py"
POSE_CKPT="/root/dhyee/mmpose/projects/rtmpose3d/rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth"

DEMO="/root/dhyee/mmpose/projects/rtmpose3d/demo/demo_no_vis.py"
INPUT_DIR="/root/dhyee/dataset/subway"
OUT_ROOT="/root/dhyee/output_rtm"
FPS=15  # (참고: 여기선 사용 안 하지만 필요 시 활용)

mkdir -p "$OUT_ROOT"

# 공백/특수문자 안전 처리: -print0 / read -d ''
find "$INPUT_DIR" -maxdepth 1 -type f -name '*.mp4' -print0 | \
while IFS= read -r -d '' VID; do
  NAME="$(basename "${VID%.mp4}")"
  OUT_DIR="$OUT_ROOT/$NAME"
  mkdir -p "$OUT_DIR"

  echo ">>> Processing: $VID"
  python "$DEMO" \
    "$DET_CFG" \
    "$DET_CKPT" \
    "$POSE_CFG" \
    "$POSE_CKPT" \
    --input "$VID" \
    --output-root "$OUT_DIR" \
    --save-predictions \
    --device cuda:0 \
    --bbox-thr 0.4 \
    --use-oks-tracking \
    --body-only

  echo ">>> Done: $OUT_DIR"
done

echo "=== All videos processed ==="

