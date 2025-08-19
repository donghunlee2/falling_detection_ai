#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BoT-SORT txt 결과를 RTMPose3D keypoint JSON에 병합하여
각 프레임의 instances 항목에 track_id를 추가합니다.

입력
- BoT-SORT TXT: frame,id,x,y,w,h,score, ... (MOTChallenge 형식)
- keypoint JSON: [{ "frame_id": int, "instances": [ { "bbox":[x1,y1,x2,y2], ... }, ... ] }, ...]

출력
- track_id가 주입된 JSON

사용 예)
python merge_tracks_into_keypoints.py \
  --botsort /path/to/2247456_botsort.txt \
  --keypoint /path/to/2247456_keypoint.json \
  --out /path/to/2247456_keypoint_with_track.json
"""
import argparse
import csv
import json
from typing import Dict, List, Tuple

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--botsort", required=True, help="BoT-SORT 결과 txt (frame,id,x,y,w,h,...)")
    p.add_argument("--keypoint", required=True, help="RTMPose3D keypoint json")
    p.add_argument("--out", required=True, help="출력 json 경로")
    p.add_argument("--min_iou", type=float, default=0.0,
                   help="IoU 매칭 최소 임계값(기본 0.0; 0보다 작지 않음)")
    p.add_argument("--use_center_fallback", action="store_true",
                   help="IoU 매칭 실패 시 중심점 거리로 보조 매칭 수행")
    return p.parse_args()

# ---------- 기본 유틸 ----------
def iou_xyxy(a: Tuple[float,float,float,float], b: Tuple[float,float,float,float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2-ax1) * max(0.0, ay2-ay1)
    area_b = max(0.0, bx2-bx1) * max(0.0, by2-by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0

def center_xyxy(b: Tuple[float,float,float,float]) -> Tuple[float,float]:
    x1,y1,x2,y2 = b
    return (0.5*(x1+x2), 0.5*(y1+y2))

# ---------- 입력 로더 ----------
def load_botsort(path: str) -> Dict[int, List[dict]]:
    """
    반환: {frame_id: [ {track_id:int, bbox_xyxy:(x1,y1,x2,y2)}, ... ]}
    """
    frames: Dict[int, List[dict]] = {}
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 6:
                continue
            frame = int(float(row[0]))
            tid   = int(float(row[1]))
            x     = float(row[2])
            y     = float(row[3])
            w     = float(row[4])
            h     = float(row[5])
            det = {
                "track_id": tid,
                "bbox_xyxy": (x, y, x + w, y + h)  # BoT-SORT는 xywh
            }
            frames.setdefault(frame, []).append(det)
    return frames

def load_keypoint(path: str) -> List[dict]:
    """
    예상 구조: [ { "frame_id":int, "instances":[ {"bbox":[x1,y1,x2,y2], ...}, ... ] }, ... ]
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------- 매칭/병합 ----------
def assign_track_ids(
    kp_frames: List[dict],
    botsort_frames: Dict[int, List[dict]],
    min_iou: float = 0.0,
    use_center_fallback: bool = False
) -> Dict[str, int]:
    """
    각 frame의 instances에 track_id 필드를 부착한다.
    greedy 1:1 매칭(IoU 최대) → (옵션) 중심점 거리 보조 매칭.
    반환: 통계 dict
    """
    stats = {"matched_iou": 0, "matched_center": 0, "no_candidate": 0}
    for entry in kp_frames:
        frame_id = int(entry.get("frame_id", -1))
        instances = entry.get("instances", [])
        dets = botsort_frames.get(frame_id, [])
        used = set()  # 동일 프레임에서 같은 det 중복 할당 방지

        for inst in instances:
            bbox = inst.get("bbox", None)
            if not bbox or len(bbox) != 4 or not dets:
                inst["track_id"] = -1
                stats["no_candidate"] += 1
                continue

            # 1) IoU로 최댓값 매칭
            best_iou, best_idx = -1.0, -1
            bb = tuple(map(float, bbox))
            for idx, d in enumerate(dets):
                if idx in used:
                    continue
                iou = iou_xyxy(bb, d["bbox_xyxy"])
                if iou > best_iou:
                    best_iou, best_idx = iou, idx

            if best_idx != -1 and best_iou >= max(0.0, min_iou):
                inst["track_id"] = dets[best_idx]["track_id"]
                used.add(best_idx)
                stats["matched_iou"] += 1
                continue

            # 2) (옵션) 중심점 거리 보조 매칭
            if use_center_fallback and dets:
                cx, cy = center_xyxy(bb)
                best_d2, best_idx2 = float("inf"), -1
                for idx, d in enumerate(dets):
                    if idx in used:
                        continue
                    dcx, dcy = center_xyxy(d["bbox_xyxy"])
                    d2 = (cx - dcx)**2 + (cy - dcy)**2
                    if d2 < best_d2:
                        best_d2, best_idx2 = d2, idx
                if best_idx2 != -1:
                    inst["track_id"] = dets[best_idx2]["track_id"]
                    used.add(best_idx2)
                    stats["matched_center"] += 1
                    continue

            # 매칭 실패
            inst["track_id"] = -1
            stats["no_candidate"] += 1
    return stats

# ---------- 엔트리포인트 ----------
def main():
    args = parse_args()
    botsort_frames = load_botsort(args.botsort)
    kp_frames = load_keypoint(args.keypoint)

    stats = assign_track_ids(
        kp_frames, botsort_frames,
        min_iou=args.min_iou,
        use_center_fallback=args.use_center_fallback
    )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(kp_frames, f, ensure_ascii=False, indent=2)

    print("[OK] wrote:", args.out)
    print("[stats]", stats)

if __name__ == "__main__":
    main()

