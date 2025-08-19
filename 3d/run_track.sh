#!/usr/bin/env bash
# run_json_plus_track_all.sh
set -uo pipefail

# 0) 파이썬이 tracker 등 패키지를 찾도록 레포 루트를 PYTHONPATH에 추가
#export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# 1) 경로/옵션 설정
JSON_PLUS="python json_plus_track.py"       # json_plus_track.py 실행 커맨드
TRACK_ROOT="/root/dhyee/output_tracking"    # *_botsort.txt가 들어 있는 상위 폴더 (<ID> 하위에 존재)
RTM_ROOT="/root/dhyee/output_rtm"           # <ID>/<ID>_4shift.json 등 키포인트 JSON이 있는 상위 폴더
OUT_ROOT="/root/dhyee/output_tracking"      # 최종 출력 JSON 저장 폴더(예: /root/dhyee/output_tracking/<ID>_botsort.json)

MIN_IOU="0.05"
USE_CENTER="--use_center_fallback"          # 빼고 싶으면 빈 문자열로

# 2) 보조 함수: ID에 해당하는 keypoint JSON 찾기
find_key_json() {
  local id="$1"
  # 우선순위대로 시도 (환경에 맞게 추가/삭제해도 됩니다)
  local candidates=(
    "$RTM_ROOT/$id/${id}_4shift.json"
    "$RTM_ROOT/$id/${id}_convert.json"
    "$RTM_ROOT/$id/${id}_4bot.json"
    "$RTM_ROOT/$id/results_${id}.json"
  )
  for f in "${candidates[@]}"; do
    [[ -f "$f" ]] && { echo "$f"; return 0; }
  done
  return 1
}

# 3) 전체 순회: *_botsort.txt를 기준으로 ID 추출 → key JSON 매칭 → 실행
#   (하위 디렉터리에 있는 모든 *_botsort.txt 대상)
find "$TRACK_ROOT" -type f -name '*_botsort.txt' -print0 | \
while IFS= read -r -d '' BOTSORT_TXT; do
  # 예: /root/dhyee/output_tracking/2247456/2247456_botsort.txt
  base="$(basename "$BOTSORT_TXT")"             # 2247456_botsort.txt
  id="${base%_botsort.txt}"                     # 2247456

  # keypoint json 찾기
  KEY_JSON="$(find_key_json "$id")" || {
    echo "경고: keypoint JSON을 찾지 못했습니다. 건너뜀 → ID=$id"
    continue
  }

  # 출력 경로 (예시: /root/dhyee/output_tracking/2247456_botsort.json)
  OUT_JSON="$OUT_ROOT/${id}/${id}_botsort.json"

  echo ">>> JSON Merge 실행: ID=$id"
  echo "    botsort: $BOTSORT_TXT"
  echo "    keypoint: $KEY_JSON"
  echo "    out: $OUT_JSON"

  # 실패해도 다음으로 계속 진행하도록 || 처리
  $JSON_PLUS \
    --botsort "$BOTSORT_TXT" \
    --keypoint "$KEY_JSON" \
    --out "$OUT_JSON" \
    --min_iou "$MIN_IOU" \
    $USE_CENTER \
    || { echo "실패: ID=$id (계속 진행)"; continue; }

  echo ">>> 완료: $OUT_JSON"
done

echo "=== 전체 JSON 머지 완료 ==="

