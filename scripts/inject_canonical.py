#!/usr/bin/env python3
"""Add a canonical URL to every built page.

The same HTML is served from https://benolabsai.com/docs/ (Beno Labs primary
domain) and may also be mirrored on GitHub Pages. Setting the canonical link
to https://benolabsai.com/docs/ consolidates search equity on the single
canonical domain.
"""

import pathlib
import re
import sys

BASE = "https://benolabsai.com/docs/"
SKIP_PREFIXES = ("rust/", "_modules/")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: inject_canonical.py <build-html-dir>")
        return 2
    out = pathlib.Path(sys.argv[1])
    if not out.is_dir():
        print(f"not a directory: {out}")
        return 2

    count = 0
    for page in out.rglob("*.html"):
        rel = page.relative_to(out).as_posix()
        if rel.startswith(SKIP_PREFIXES):
            continue
        text = page.read_text(errors="ignore")
        if 'rel="canonical"' in text:
            continue
        href = BASE if rel == "index.html" else BASE + rel
        tag = f'<link rel="canonical" href="{href}">'
        updated = re.sub(r"(<head[^>]*>)", r"\1\n" + tag.replace("\\", "\\\\"), text, count=1)
        if updated != text:
            page.write_text(updated)
            count += 1
    print(f"canonical URL injected into {count} page(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
