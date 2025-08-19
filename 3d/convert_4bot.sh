#!/usr/bin/env bash
set -euo pipefail

# ==== 경로 설정 ====
CONVERTER="/root/dhyee/convert_4bot.py"       # convert_4bot.py의 실제 경로로 수정
ROOT="/root/dhyee/output_rtm"                 # results_*.json들이 들어있는 상위 폴더
MIN_SCORE="0.0"

# ROOT 하위 모든 results_*.json 순회
# -print0 로 null-구분 → 특수문자/공백 안전
find "$ROOT" -type f -name 'results_*.json' -print0 | \
while IFS= read -r -d '' IN_JSON; do
  # 예: IN_JSON=/root/dhyee/output_rtm/2247456/results_2247456.json
  DIR="$(dirname "$IN_JSON")"                 # /root/dhyee/output_rtm/2247456
  BASE="$(basename "$IN_JSON")"               # results_2247456.json

  # name 추출: results_  접두어 / .json 접미어 제거 → 2247456
  NAME="${BASE#results_}"
  NAME="${NAME%.json}"

  # 출력 경로들(폴더는 절대경로로 지정)
  OUT_JSON="$DIR/${NAME}_4bot.json"
  OUT_JSON_KPS="$DIR/${NAME}_convert.json"
  OUT_NPY_DIR="$DIR/botsort_dets_npy"
  OUT_NPZ_DIR="$DIR/dets_kps_npz"

  mkdir -p "$OUT_NPY_DIR" "$OUT_NPZ_DIR"

  echo ">>> Converting: $IN_JSON"
  python "$CONVERTER" \
    --input "$IN_JSON" \
    --out-json "$OUT_JSON" \
    --out-npy-dir "$OUT_NPY_DIR" \
    --out-json-kps "$OUT_JSON_KPS" \
    --out-npz-dir "$OUT_NPZ_DIR" \
    --min-score "$MIN_SCORE"

  echo ">>> Done: $OUT_JSON"
done

echo "=== All conversions finished ==="

