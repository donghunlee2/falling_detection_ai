#!/usr/bin/env bash
set -euo pipefail

# 입력 이미지 루트와 출력 루트
ROOT="/root/Data/Training/Falling_Train"
OUT="/root/dhyee/dataset/subway"
FPS=15

mkdir -p "$OUT"

# 루트 바로 아래의 모든 하위 디렉터리를 순회
find "$ROOT" -mindepth 1 -maxdepth 1 -type d | while read -r DIR; do
  NAME="$(basename "$DIR")"
  LIST="$OUT/${NAME}.txt"
  MP4="$OUT/${NAME}.mp4"

  echo ">> 처리중: $NAME"

  # 1) concat 리스트 생성 (자연수 기준 정렬: sort -V)
  #    파일명이 매우 많거나 공백/특수문자 포함해도 안전하게 처리
  #    frame_*.jpg이 없으면 건너뜀
  if ! find "$DIR" -maxdepth 1 -type f -name 'frame_*.jpg' | grep -q .; then
    echo "   경고: 프레임이 없습니다: $DIR (건너뜀)"
    continue
  fi

  # 리스트 파일 생성
  # -print0 / -z 옵션으로 null-구분 정렬 → 안전한 파일명 처리
  find "$DIR" -maxdepth 1 -type f -name 'frame_*.jpg' -print0 \
    | sort -z -V \
    | while IFS= read -r -d '' F; do
        printf "file '%s'\n" "$F"
      done > "$LIST"

  # 2) ffmpeg로 mp4 생성
  ffmpeg -y -f concat -safe 0 -r "$FPS" -i "$LIST" \
    -c:v libx264 -pix_fmt yuv420p -movflags +faststart "$MP4"

  echo "   완료: $MP4"
done

echo "=== 전체 처리 완료 ==="

