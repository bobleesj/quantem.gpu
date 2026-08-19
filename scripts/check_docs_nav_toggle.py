#!/usr/bin/env python3
"""Guard the built documentation's mobile navigation against theme drift."""

from __future__ import annotations

import re
import sys
from pathlib import Path


TOGGLE_RE = re.compile(
    r'<(?:button|label)[^>]*class="[^"]*\bprimary-toggle\b', re.IGNORECASE
)
SHIM_RE = re.compile(r"nav-toggle-fix\.js", re.IGNORECASE)


def main() -> int:
    """Check every themed HTML page for a usable primary navigation toggle."""
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/_build/html")
    pages = [
        path
        for path in root.rglob("*.html")
        if "_sources" not in path.parts and "_static" not in path.parts
    ]
    if not pages:
        print(f"check_docs_nav_toggle: no built pages under {root}", file=sys.stderr)
        return 1

    failures: list[tuple[Path, str]] = []
    checked = 0
    for page in pages:
        text = page.read_text(encoding="utf-8", errors="replace")
        if "bd-main" not in text:
            continue
        checked += 1
        toggles = len(TOGGLE_RE.findall(text))
        if toggles == 0:
            failures.append((page.relative_to(root), "no primary-toggle control"))
        elif toggles > 1 and not SHIM_RE.search(text):
            failures.append(
                (
                    page.relative_to(root),
                    f"{toggles} controls but nav-toggle-fix.js is not loaded",
                )
            )

    if failures:
        print("check_docs_nav_toggle: FAIL", file=sys.stderr)
        for page, reason in failures[:20]:
            print(f"  {page}: {reason}", file=sys.stderr)
        return 1
    if not checked:
        print("check_docs_nav_toggle: no themed pages found", file=sys.stderr)
        return 1

    print(f"check_docs_nav_toggle: OK -- {checked} pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
