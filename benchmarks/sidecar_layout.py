#!/usr/bin/env python3
"""Report the index sidecar layout of a table directory.

The paper quotes three numbers that come from this layout and not from
arithmetic: the sidecar total, the shard count, and the index to cache ratio.
They move whenever the compaction target or the cluster count changes, so they
are measured from the artifact that ships.

Usage:
    python benchmarks/sidecar_layout.py <db_dir> [--cache-gb 8]
"""

import argparse
import os
import sys


def human_mb(value: float) -> str:
    return f"{value:,.1f} MB"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db_dir", help="Table directory to inspect")
    parser.add_argument("--cache-gb", type=float, default=8.0,
                        help="Cache budget to compute the index to cache ratio")
    args = parser.parse_args()

    if not os.path.isdir(args.db_dir):
        print(f"no such directory: {args.db_dir}", file=sys.stderr)
        return 2

    sizes = []
    for root, _dirs, files in os.walk(args.db_dir):
        for name in files:
            if name.endswith(".puffin"):
                path = os.path.join(root, name)
                try:
                    sizes.append(os.path.getsize(path))
                except OSError:
                    pass

    if not sizes:
        print(f"no Puffin sidecars under {args.db_dir}", file=sys.stderr)
        return 1

    sizes.sort()
    count = len(sizes)
    total = sum(sizes)
    median = sizes[(count - 1) // 2] if count % 2 else (
        sizes[count // 2 - 1] + sizes[count // 2]
    ) / 2
    mb = 1024 * 1024

    print(f"sidecars        {count}")
    print(f"total           {total / mb / 1024:,.2f} GB")
    print(f"min             {human_mb(sizes[0] / mb)}")
    print(f"median          {human_mb(median / mb)}")
    print(f"max             {human_mb(sizes[-1] / mb)}")
    print(f"mean            {human_mb(total / count / mb)}")

    cache_bytes = args.cache_gb * 1024 ** 3
    if cache_bytes > 0:
        print(f"cache budget    {args.cache_gb:,.1f} GiB")
        print(f"index to cache  {total / cache_bytes:,.2f}x")

    in_range = sum(1 for s in sizes if 512 * mb <= s <= 1024 * mb)
    print(f"in 512 MB to 1 GiB  {in_range} of {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
