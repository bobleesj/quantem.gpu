#!/usr/bin/env python3
"""Check repository-local Markdown links and built HTML assets."""

from __future__ import annotations

import argparse
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
EXTERNAL_SCHEMES = {"data", "ftp", "http", "https", "javascript", "mailto"}


def markdown_without_fences(text: str) -> str:
    """Return Markdown with fenced code removed from link inspection."""
    lines: list[str] = []
    fence: str | None = None
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("```"):
            marker = "```"
        elif stripped.startswith("~~~"):
            marker = "~~~"
        else:
            marker = None
        if marker:
            fence = None if fence == marker else marker if fence is None else fence
            continue
        if fence is None:
            lines.append(line)
    return "\n".join(lines)


def local_path(target: str) -> str | None:
    """Return the decoded local path component, or ``None`` for external links."""
    target = target.strip().strip("<>")
    if not target or target.startswith("#") or target.startswith("//"):
        return None
    parsed = urlsplit(target)
    if parsed.scheme.lower() in EXTERNAL_SCHEMES or parsed.netloc:
        return None
    return unquote(parsed.path)


def check_markdown(root: Path) -> list[str]:
    """Check local Markdown links relative to their source documents."""
    docs = [
        path
        for path in sorted((root / "docs").rglob("*.md"))
        if "_build" not in path.parts
    ]
    paths = [root / "README.md", root / "CONTRIBUTING.md", *docs]
    failures: list[str] = []
    for source in paths:
        if not source.is_file():
            failures.append(f"missing source document: {source.relative_to(root)}")
            continue
        text = markdown_without_fences(source.read_text(encoding="utf-8"))
        for match in MARKDOWN_LINK.finditer(text):
            target = match.group(1).split(maxsplit=1)[0]
            path_text = local_path(target)
            if path_text is None:
                continue
            candidate = (source.parent / path_text).resolve()
            if not candidate.exists():
                line = text.count("\n", 0, match.start()) + 1
                failures.append(
                    f"{source.relative_to(root)}:{line}: missing local target {target}"
                )
    return failures


class AssetParser(HTMLParser):
    """Collect link and asset targets from one generated HTML page."""

    def __init__(self) -> None:
        super().__init__()
        self.targets: list[str] = []

    def handle_starttag(self, _tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if name in {"href", "src"} and value:
                self.targets.append(value)


def check_html(root: Path) -> list[str]:
    """Check links and assets emitted by the documentation build."""
    failures: list[str] = []
    pages = [
        page
        for page in sorted(root.rglob("*.html"))
        if "_sources" not in page.parts and "_static" not in page.parts
    ]
    if not pages:
        return [f"no built HTML pages under {root}"]

    for page in pages:
        parser = AssetParser()
        parser.feed(page.read_text(encoding="utf-8", errors="replace"))
        for target in parser.targets:
            path_text = local_path(target)
            if path_text is None:
                continue
            if path_text.startswith("/"):
                candidate = root / path_text.lstrip("/")
            else:
                candidate = page.parent / path_text
            candidate = candidate.resolve()
            if candidate.is_dir():
                candidate = candidate / "index.html"
            if not candidate.exists():
                failures.append(
                    f"{page.relative_to(root)}: missing built target {target}"
                )
    return failures


def main() -> int:
    """Run source checks and, when requested, generated-site checks."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--html-root", type=Path)
    args = parser.parse_args()

    repo = args.repo.resolve()
    failures = check_markdown(repo)
    if args.html_root is not None:
        html_root = args.html_root
        if not html_root.is_absolute():
            html_root = repo / html_root
        failures.extend(check_html(html_root.resolve()))

    if failures:
        print("check_docs_links: FAIL", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1

    scope = "source and built HTML" if args.html_root is not None else "source"
    print(f"check_docs_links: OK -- {scope} links resolve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
