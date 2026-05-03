#!/usr/bin/env python3
"""
Partition test episode ids for disjoint multi-GPU rollout_minireal.py jobs.

When N is not divisible by W, the first (N %% W) shards get ceil(N/W) episodes each;
the rest get floor(N/W). Shards are contiguous in the same sort order as rollout
(sorted directory names under --test-data).

Example: N=100, W=8 -> sizes 13,13,13,13,12,12,12,12 (sum 100, no gaps, no duplicates).

Usage:
  # Human-readable table (recommended before launch)
  python scripts/shard_episodes_for_rollout.py --test-data /path/to/test --workers 8 --summary

  # Episodes for worker 3 only (comma-separated for --episodes)
  python scripts/shard_episodes_for_rollout.py --test-data /path/to/test --workers 8 --print-shard 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def list_episode_names(test_root: Path) -> list[str]:
    if not test_root.is_dir():
        raise FileNotFoundError(f"test-data is not a directory: {test_root}")
    return sorted(p.name for p in test_root.iterdir() if p.is_dir())


def partition_shards(names: list[str], workers: int) -> list[list[str]]:
    """Disjoint contiguous shards; first (n %% workers) shards are one longer."""
    n = len(names)
    if workers < 1:
        raise ValueError("workers must be >= 1")
    base, rem = divmod(n, workers)
    out: list[list[str]] = []
    idx = 0
    for w in range(workers):
        sz = base + (1 if w < rem else 0)
        out.append(names[idx : idx + sz])
        idx += sz
    assert idx == n, "internal partition error"
    flat = [e for s in out for e in s]
    assert len(flat) == n, "internal partition length"
    assert len(set(flat)) == n, "internal duplicate episode in partition"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-data", type=str, required=True, help="Same as rollout --test-data")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel rollout workers")
    parser.add_argument(
        "--print-shard",
        type=int,
        default=None,
        metavar="I",
        help="Print comma-separated episode ids for shard I only (stdout, one line)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print shard index, count, first/last episode to stderr",
    )
    args = parser.parse_args()
    if args.print_shard is None and not args.summary:
        parser.error("specify --summary and/or --print-shard I")

    test_root = Path(args.test_data).resolve()
    names = list_episode_names(test_root)
    n = len(names)
    shards = partition_shards(names, args.workers)

    if args.summary:
        print(f"test-data={test_root}", file=sys.stderr)
        print(f"episodes_total={n} workers={args.workers}", file=sys.stderr)
        for i, s in enumerate(shards):
            if not s:
                print(f"shard {i}: count=0 (empty)", file=sys.stderr)
                continue
            print(
                f"shard {i}: count={len(s)} first={s[0]!r} last={s[-1]!r}",
                file=sys.stderr,
            )
        seen: set[str] = set()
        for s in shards:
            dup = seen & set(s)
            if dup:
                raise SystemExit(f"duplicate across shards: {dup!r}")
            seen |= set(s)
        if seen != set(names):
            raise SystemExit("partition does not cover all episodes exactly")
        print("OK: shards are disjoint and cover all episodes.", file=sys.stderr)

    if args.print_shard is not None:
        i = args.print_shard
        if i < 0 or i >= args.workers:
            raise SystemExit(f"--print-shard must be in [0, {args.workers}), got {i}")
        print(",".join(shards[i]))


if __name__ == "__main__":
    main()
