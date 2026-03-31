"""
Test: Internal link validation
Ensures all relative links in Markdown files resolve to actual files on disk.

Why this matters:
  - CONTRIBUTING.md explicitly requires "Test that all links work" before submitting a PR.
  - With 80+ files and hundreds of cross-references, broken links are easy to introduce
    (e.g. renaming a file without updating all references to it).
  - Currently there is NO automated check, so broken links can silently persist.
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent

# Markdown link pattern: [text](href)
_LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")


def _all_markdown_files():
    return sorted(REPO_ROOT.rglob("*.md"))


def _extract_relative_links(content: str) -> list[tuple[str, str]]:
    """
    Return (link_text, href) pairs for every relative link in content.
    Excludes:
      - Absolute URLs (http/https/ftp/mailto)
      - Pure anchor links (#section)
    """
    links = []
    for m in _LINK_RE.finditer(content):
        text, href = m.group(1), m.group(2)
        if href.startswith(("#", "http", "ftp", "mailto")):
            continue
        links.append((text, href))
    return links


@pytest.mark.parametrize(
    "md_file",
    _all_markdown_files(),
    ids=lambda f: str(f.relative_to(REPO_ROOT)),
)
def test_relative_links_resolve(md_file: Path):
    """Every relative link in a Markdown file must point to an existing file."""
    content = md_file.read_text(encoding="utf-8")
    links = _extract_relative_links(content)

    broken = []
    for text, href in links:
        # Strip anchor fragment before resolving
        path_part = href.split("#")[0]
        if not path_part:
            continue  # pure in-page anchor — already excluded above, but be safe

        target = (md_file.parent / path_part).resolve()
        if not target.exists():
            broken.append(f'  [{text}]({href})')

    assert not broken, (
        f"Broken relative links in {md_file.relative_to(REPO_ROOT)}:\n"
        + "\n".join(broken)
    )
