#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert results_output_*.json to:
  1) BoT-SORT-style bbox dets (unchanged): xyxy+score per frame
  2) NEW: Full keypoints for *every* instance in each frame

Outputs
-------
- <out_json>:           [{"frame_id": int, "dets_xyxy5": [[x1,y1,x2,y2,score], ...]}, ...]
- <out_json_kps>:       [{"frame_id": int, "instances": [{"bbox":[x1,y1,x2,y2], "score":float,
                                                          "keypoints":[[x,y,z],...],
                                                          "keypoint_scores":[s1,...]}, ...]}, ...]
- <out_npy_dir>/<fid>.npy     (same as before: (N,5) bbox+score)
- <out_npz_dir>/<fid>.npz     (NEW) contains two arrays:
                              - bboxes_xyxy5: (N,5)
                              - keypoints_xyz: (N,K,3)  (K can vary per file; we pad with NaNs to max-K in the frame)
                              - keypoint_scores: (N,K)  (NaN padded if lengths differ)

Usage
-----
python convert_json_with_kps.py --input results_output_*.json \
       --out-json botsort_dets_xyxy5.json \
       --out-npy-dir botsort_dets_npy \
       --out-json-kps dets_with_all_keypoints.json \
       --out-npz-dir dets_kps_npz \
       --min-score 0.0
"""

import argparse
import json
import os
from collections import defaultdict
import numpy as np
from typing import List, Dict, Any, Tuple

def extract_xyxy_from_bbox(bbox_field):
    """
    입력 bbox 필드가 다음 중 하나라고 가정하고 [x1,y1,x2,y2]로 반환:
      - [x1, y1, x2, y2]
      - [[x1, y1, x2, y2]]  (리스트 한 겹 더 감싼 케이스)
    """
    if bbox_field is None:
        return None
    if isinstance(bbox_field, (list, tuple)):
        if len(bbox_field) == 4 and all(isinstance(v, (int, float)) for v in bbox_field):
            return [float(bbox_field[0]), float(bbox_field[1]),
                    float(bbox_field[2]), float(bbox_field[3])]
        # [[x1,y1,x2,y2]] 형태
        if len(bbox_field) == 1 and isinstance(bbox_field[0], (list, tuple)) and len(bbox_field[0]) == 4:
            inner = bbox_field[0]
            return [float(inner[0]), float(inner[1]), float(inner[2]), float(inner[3])]
    return None

def pad_sequences(arrs: List[np.ndarray], pad_value: float = np.nan) -> np.ndarray:
    """
    arrs: list of arrays with shape (K_i, D)
    Returns: (N, K_max, D) padded with pad_value
    """
    if not arrs:
        return np.zeros((0, 0, 0), dtype=np.float32)
    N = len(arrs)
    K = max(a.shape[0] for a in arrs)
    D = arrs[0].shape[1] if arrs[0].ndim == 2 else 1
    out = np.full((N, K, D), pad_value, dtype=np.float32)
    for i, a in enumerate(arrs):
        if a.ndim == 1:
            a = a[:, None]
        k = a.shape[0]
        out[i, :k, :a.shape[1]] = a.astype(np.float32)
    return out

def convert(input_json_path: str,
            out_json_path: str,
            out_npy_dir: str,
            out_json_kps_path: str,
            out_npz_dir: str,
            min_score: float = 0.0) -> int:
    """
    results_output_*.json ->
      - BoT-SORT 입력 형식(JSON/NPY)
      - 전체 keypoints 포함 JSON/NPZ
    반환값: 유효 프레임 수
    """
    # 1) JSON 로드
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2) 프레임별 누적
    per_frame_xyxy5: Dict[int, List[List[float]]] = defaultdict(list)
    per_frame_instances: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    inst_list = data.get("instance_info", [])

    for frame_entry in inst_list:
        frame_id = int(frame_entry.get("frame_id", -1))
        if frame_id < 0:
            continue

        instances = frame_entry.get("instances", [])
        for inst in instances:
            bbox_xyxy = extract_xyxy_from_bbox(inst.get("bbox"))
            if bbox_xyxy is None:
                continue

            score = inst.get("bbox_score", inst.get("score", None))
            if score is None:
                score = 1.0
            score = float(score)
            if score < min_score:
                continue

            # bbox + score (BoT-SORT)
            x1, y1, x2, y2 = bbox_xyxy
            per_frame_xyxy5[frame_id].append([x1, y1, x2, y2, score])

            # full keypoints (모든 인스턴스 보존)
            kps = inst.get("keypoints", []) or []
            kps_scores = inst.get("keypoint_scores", []) or []
            per_frame_instances[frame_id].append({
                "bbox": [x1, y1, x2, y2],
                "score": score,
                "keypoints": kps,
                "keypoint_scores": kps_scores
            })

    # 3) 정렬
    frames_sorted_ids = sorted(set(list(per_frame_xyxy5.keys()) + list(per_frame_instances.keys())))
    frames_xyxy_export = [{"frame_id": fid, "dets_xyxy5": per_frame_xyxy5.get(fid, [])} for fid in frames_sorted_ids]
    frames_kps_export = [{"frame_id": fid, "instances": per_frame_instances.get(fid, [])} for fid in frames_sorted_ids]

    # 4) JSON 내보내기
    os.makedirs(os.path.dirname(out_json_path) or ".", exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(frames_xyxy_export, f, indent=2, ensure_ascii=False)

    os.makedirs(os.path.dirname(out_json_kps_path) or ".", exist_ok=True)
    with open(out_json_kps_path, "w", encoding="utf-8") as f:
        json.dump(frames_kps_export, f, indent=2, ensure_ascii=False)

    # 5) NPY/NPZ 저장
    os.makedirs(out_npy_dir, exist_ok=True)
    os.makedirs(out_npz_dir, exist_ok=True)

    for fid in frames_sorted_ids:
        dets = np.array(per_frame_xyxy5.get(fid, []), dtype=np.float32) if per_frame_xyxy5.get(fid) else np.zeros((0, 5), dtype=np.float32)
        np.save(os.path.join(out_npy_dir, f"{fid:06d}.npy"), dets)

        # Build NPZ with keypoints
        insts = per_frame_instances.get(fid, [])
        if insts:
            bboxes = np.array([i["bbox"] + [i["score"]] for i in insts], dtype=np.float32)  # (N,5)
            kps_list = [np.array(i["keypoints"], dtype=np.float32) if i["keypoints"] else np.zeros((0,3), dtype=np.float32) for i in insts]
            kps_scores_list = [np.array(i["keypoint_scores"], dtype=np.float32) if i["keypoint_scores"] else np.zeros((0,), dtype=np.float32) for i in insts]
            kps_pad = pad_sequences(kps_list, pad_value=np.nan)                  # (N,K,3)
            kps_scores_pad = pad_sequences([s.reshape(-1,1) for s in kps_scores_list], pad_value=np.nan)[:,:,0]  # (N,K)
        else:
            bboxes = np.zeros((0,5), dtype=np.float32)
            kps_pad = np.zeros((0,0,3), dtype=np.float32)
            kps_scores_pad = np.zeros((0,0), dtype=np.float32)

        np.savez_compressed(os.path.join(out_npz_dir, f"{fid:06d}.npz"),
                            bboxes_xyxy5=bboxes,
                            keypoints_xyz=kps_pad,
                            keypoint_scores=kps_scores_pad)

    return len(frames_sorted_ids)

def main():
    ap = argparse.ArgumentParser(description="Convert results_output_*.json to BoT-SORT det format and keep ALL keypoints.")
    ap.add_argument("--input", required=True, help="Path to results_output_*.json")
    ap.add_argument("--out-json", default="botsort_dets_xyxy5.json", help="Output JSON path (bbox only; legacy)")
    ap.add_argument("--out-npy-dir", default="botsort_dets_npy", help="Directory to save per-frame .npy (bbox only; legacy)")
    ap.add_argument("--out-json-kps", default="dets_with_all_keypoints.json", help="Output JSON path WITH keypoints")
    ap.add_argument("--out-npz-dir", default="dets_kps_npz", help="Directory to save per-frame .npz (bbox+keypoints)")
    ap.add_argument("--min-score", type=float, default=0.0, help="Minimum bbox score filter")
    args = ap.parse_args()

    nframes = convert(args.input, args.out_json, args.out_npy_dir, args.out_json_kps, args.out_npz_dir, args.min_score)
    print(f"[OK] Converted {nframes} frames ->")
    print(f"     JSON (bbox-only)        : {args.out_json}")
    print(f"     JSON (with keypoints)   : {args.out_json_kps}")
    print(f"     NPYs (bbox-only)        : {args.out_npy_dir}/000001.npy ...")
    print(f"     NPZs (bbox+keypoints)   : {args.out_npz_dir}/000001.npz ...")

if __name__ == "__main__":
    main()
