#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert pose JSONs (frames -> instances[track_id, keypoints]) to Shift-GCN skeleton text.

Output format per video (one file):
  1) First line: T  (number of frames)
  2) For each frame:
       - One line: N  (people count = number of persons with valid keypoints in this frame)
       - For each person (sorted by track_id):
           * One line: track_id (int)
           * One line: V (joints)
           * V lines:  "x y z"  (float float float)

Naming scheme for output files:
  00nA00m.skeleton
    n = --action-index  (zero-padded by --pad-n, default 3)
    m = sequential number starting from --start-num (zero-padded by --pad-m, default 3)

Key constraints:
- Every instance must have exactly --joints keypoints. If not, stop with an error.
- By default, any keypoint values are considered "valid".
  If --require-nonzero is provided, persons with ALL joints (0,0,0) are excluded
  from the frame count and not printed.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
import glob as _glob

TrackIdKeys = ("track_id", "tracking_id", "id", "person_id")

# ---------------------------
# File iteration (supports absolute/relative glob with **)
# ---------------------------

def list_input_files(input_arg: str) -> List[Path]:
    """Return sorted list of JSON files from a path/glob (absolute/relative, ** supported)."""
    p = Path(input_arg)
    files: List[Path] = []

    if any(ch in input_arg for ch in "*?[]"):
        for f in _glob.glob(input_arg, recursive=True):
            fp = Path(f)
            if fp.is_file() and fp.suffix.lower() == ".json":
                files.append(fp)
    elif p.is_dir():
        files = [fp for fp in p.rglob("*.json") if fp.is_file()]
    elif p.is_file() and p.suffix.lower() == ".json":
        files = [p]
    else:
        raise FileNotFoundError(f"Input not found or not a .json: {input_arg}")

    return sorted(files, key=lambda x: str(x.resolve()))

# ---------------------------
# JSON loading / parsing
# ---------------------------

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def extract_frames(obj: Any) -> List[Dict[str, Any]]:
    """
    Accepts:
      A) List[ {frame_id, instances: [...]}, ... ]
      B) Dict{ 'instance_info': [ {frame_id, instances}, ... ] }
    Returns: list of frames sorted by frame_id (if present).
    """
    if isinstance(obj, list):
        frames = obj
    elif isinstance(obj, dict) and "instance_info" in obj:
        frames = obj["instance_info"]
    else:
        raise ValueError("Unsupported JSON root structure. Expect a list of frames or dict with 'instance_info'.")

    frames = sorted(frames, key=lambda fr: fr.get("frame_id", 0))
    return frames

def get_track_id(inst: Dict[str, Any], fallback_index: int) -> int:
    for k in TrackIdKeys:
        if k in inst:
            try:
                return int(inst[k])
            except Exception:
                break
    # Fallback: within-frame order (1-based), not stable across frames
    return int(fallback_index)

def normalize_kpts(kpts: Any) -> List[List[float]]:
    """Flatten one extra nesting and ensure each point is [x, y, z]."""
    # Unwrap a single extra nesting if present (e.g., [[[...]]])
    if isinstance(kpts, list) and len(kpts) == 1 and isinstance(kpts[0], list):
        kpts = kpts[0]
    if not isinstance(kpts, list) or (kpts and not isinstance(kpts[0], list)):
        raise ValueError("Invalid keypoints structure; expected list of [x,y,z].")
    out = []
    for p in kpts:
        if not isinstance(p, list) or len(p) < 3:
            raise ValueError("Each keypoint must have at least 3 values [x, y, z].")
        out.append([float(p[0]), float(p[1]), float(p[2])])
    return out

def all_zero_kpts(kpts: List[List[float]]) -> bool:
    """Return True if ALL joints are exactly (0,0,0)."""
    for x, y, z in kpts:
        if x != 0.0 or y != 0.0 or z != 0.0:
            return False
    return True

# ---------------------------
# Naming scheme: 00nA00m
# ---------------------------

def build_name(action_index: int, m: int, pad_n: int, pad_m: int) -> str:
    n_str = str(action_index).zfill(pad_n)
    m_str = str(m).zfill(pad_m)
    return f"{n_str}A{m_str}"

def next_free_outpath(
    out_dir: Path,
    action_index: int,
    start_m: int,
    pad_n: int,
    pad_m: int,
    ext: str,
    overwrite: bool,
) -> Tuple[Path, int]:
    """
    Find the first available out path >= start_m that doesn't exist,
    unless overwrite=True, in which case we return exactly start_m.
    """
    if overwrite:
        name = build_name(action_index, start_m, pad_n, pad_m)
        return out_dir / f"{name}{ext}", start_m

    m = start_m
    while True:
        name = build_name(action_index, m, pad_n, pad_m)
        path = out_dir / f"{name}{ext}"
        if not path.exists():
            return path, m
        m += 1

# ---------------------------
# Core conversion
# ---------------------------

def convert_frames_to_lines(
    frames: List[Dict[str, Any]],
    joints: int,
    require_nonzero: bool = False,
) -> List[str]:
    """
    Convert frames to Shift-GCN lines with:
      - Header N = count of persons with valid keypoints in the frame.
      - Persons sorted by track_id; each prints: track_id, V, and V lines of x y z.
    """
    per_frame: List[Dict[int, List[List[float]]]] = []

    # 1) Gather valid persons per frame
    for fr in frames:
        instances = fr.get("instances", []) or []
        ids_to_kpts: Dict[int, List[List[float]]] = {}
        for idx, inst in enumerate(instances, start=1):
            tid = get_track_id(inst, idx)
            kpts = normalize_kpts(inst.get("keypoints", []))
            if len(kpts) != joints:
                raise ValueError(
                    f"keypoints length {len(kpts)} != --joints {joints} "
                    f"(frame_id={fr.get('frame_id')}, track_id={tid})"
                )
            if require_nonzero and all_zero_kpts(kpts):
                continue
            ids_to_kpts[int(tid)] = kpts
        per_frame.append(ids_to_kpts)

    # 2) Compose output lines
    lines: List[str] = []
    T = len(per_frame)
    lines.append(str(T))  # Put T at the top immediately

    for ids_to_kpts in per_frame:
        track_ids_sorted = sorted(ids_to_kpts.keys())
        header_n = len(track_ids_sorted)
        lines.append(str(header_n))
        for tid in track_ids_sorted:
            kpts = ids_to_kpts[tid]
            lines.append(str(int(tid)))
            lines.append(str(int(joints)))
            for x, y, z in kpts:
                lines.append(f"{x} {y} {z}")

    return lines

# ---------------------------
# CLI
# ---------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Convert frame-wise JSON to Shift-GCN skeleton text (N = count of valid persons)."
    )
    ap.add_argument("--input", required=True,
                    help="Input file, directory, or glob pattern (absolute/relative, ** supported).")
    ap.add_argument("--outdir", required=True, help="Output directory (single fixed folder).")
    ap.add_argument("--joints", type=int, required=True, help="Joint count V (e.g., 17).")

    # Naming scheme controls
    ap.add_argument("--action-index", type=int, required=True,
                    help="Action index n for name '00nA00m' (e.g., 1 -> 001A...).")
    ap.add_argument("--start-num", type=int, required=True,
                    help="Starting m for name '00nA00m' (e.g., 6 -> 001A006...).")
    ap.add_argument("--pad-n", type=int, default=3, help="Zero-pad width for n (default: 3).")
    ap.add_argument("--pad-m", type=int, default=3, help="Zero-pad width for m (default: 3).")
    ap.add_argument("--ext", default=".skeleton", help="Output extension (default: .skeleton)")
    ap.add_argument("--overwrite", action="store_true", default=False,
                    help="Overwrite if the 00nA00m file already exists (default: False).")

    # Valid-person rule
    ap.add_argument("--require-nonzero", action="store_true", default=False,
                    help="Exclude persons whose ALL joints are (0,0,0) from N and from output.")

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    files = list_input_files(args.input)
    if not files:
        print("[ERROR] No JSON files found.", file=sys.stderr)
        sys.exit(1)

    next_m = args.start_num

    try:
        for in_fp in files:
            data = load_json(in_fp)
            frames = extract_frames(data)
            lines = convert_frames_to_lines(
                frames=frames,
                joints=args.joints,
                require_nonzero=args.require_nonzero,
            )

            out_path, used_m = next_free_outpath(
                out_dir=outdir,
                action_index=args.action_index,
                start_m=next_m,
                pad_n=args.pad_n,
                pad_m=args.pad_m,
                ext=args.ext,
                overwrite=args.overwrite,
            )
            next_m = used_m + 1  # advance for the next file

            with out_path.open("w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")

            print(out_path)

    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

