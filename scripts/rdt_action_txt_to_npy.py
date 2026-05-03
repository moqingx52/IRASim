#!/usr/bin/env python3
"""
Convert RDT CSV outputs (same layout as train/*/action.txt) to flat numpy files
for scripts/rollout_minireal.py.

Input layout:
  <src>/<episode>/action.txt
    Header row + data rows; first column is index (Unnamed: 0), columns 1..26 are values.

Output layout:
  <out>/<episode>.npy   float32, shape [51, 26]

Run from IRASim root:
  python scripts/rdt_action_txt_to_npy.py \\
    --src /workspace/sample_result_rdt \\
    --out /workspace/sample_result_rdt_npy \\
    --only-test /workspace/test
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


def _load_action_body(action_txt: Path) -> tuple[list[str], list[list[str]]]:
    with action_txt.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError("empty file")
    header = rows[0]
    data_rows = [r for r in rows[1:] if r]
    return header, data_rows


def _rows_to_array(data_rows: list[list[str]], slice_mode: str):
    n = len(data_rows)
    if n < 51:
        raise ValueError(f"need at least 51 data rows, got {n}")
    if slice_mode == "strict" and n != 51:
        raise ValueError(f"strict mode: expected exactly 51 data rows, got {n}")
    if slice_mode == "last":
        picked = data_rows[-51:]
    elif slice_mode == "first":
        picked = data_rows[:51]
    else:
        picked = data_rows

    mat = []
    for r in picked:
        if len(r) < 27:
            raise ValueError(f"row has {len(r)} cols, need >= 27 (index + 26)")
        vals = [float(x) for x in r[1:27]]
        mat.append(vals)
    arr = np.asarray(mat, dtype=np.float32)
    if arr.shape != (51, 26):
        raise RuntimeError(f"internal shape bug: {arr.shape}")
    return arr


def main() -> None:
    parser = argparse.ArgumentParser(description="RDT action.txt -> episode.npy [51,26]")
    parser.add_argument("--src", type=str, required=True, help="Root with <episode>/action.txt")
    parser.add_argument("--out", type=str, required=True, help="Output dir for flat <episode>.npy")
    parser.add_argument(
        "--only-test",
        type=str,
        default=None,
        help="If set, only convert episodes whose names are subdirs of this folder (e.g. test set)",
    )
    parser.add_argument(
        "--action-filename",
        type=str,
        default="action.txt",
        help="Filename under each episode dir (default: action.txt)",
    )
    parser.add_argument(
        "--slice",
        type=str,
        choices=("strict", "last", "first"),
        default="strict",
        help="How to pick 51 rows if file has more than 51 (default: strict = error)",
    )
    args = parser.parse_args()

    src_root = Path(args.src).resolve()
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    allowed: set[str] | None = None
    if args.only_test:
        test_root = Path(args.only_test).resolve()
        if not test_root.is_dir():
            print(f"[fatal] --only-test not a directory: {test_root}", file=sys.stderr)
            raise SystemExit(1)
        allowed = {p.name for p in test_root.iterdir() if p.is_dir()}

    if not src_root.is_dir():
        print(f"[fatal] --src not a directory: {src_root}", file=sys.stderr)
        raise SystemExit(1)

    ok = skip = err = 0
    subdirs = sorted([p for p in src_root.iterdir() if p.is_dir()], key=lambda p: p.name)

    for ep_dir in subdirs:
        ep = ep_dir.name
        if allowed is not None and ep not in allowed:
            skip += 1
            continue

        action_path = ep_dir / args.action_filename
        if not action_path.is_file():
            print(f"[skip] {ep}: missing {action_path.name}")
            skip += 1
            continue

        try:
            _, data_rows = _load_action_body(action_path)
            arr = _rows_to_array(data_rows, args.slice)
        except Exception as e:
            print(f"[error] {ep}: {e}")
            err += 1
            continue

        out_path = out_root / f"{ep}.npy"
        np.save(out_path, arr)
        print(f"[ok] {ep} -> {out_path} {arr.shape} {arr.dtype}")
        ok += 1

    print(f"Done: ok={ok} skip={skip} err={err} out={out_root}")


if __name__ == "__main__":
    main()
