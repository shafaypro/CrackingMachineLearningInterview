"""
Test: Code block language tags
Ensures every fenced code block in Markdown guide files opens with a language specifier.

Why this matters:
  - CONTRIBUTING.md explicitly requires: "Always specify the language" for code blocks.
  - Language tags enable syntax highlighting on GitHub Pages and in editors.
  - Analysis shows ~177 untagged code blocks currently exist across the repo.
  - Untagged blocks render as plain text, degrading readability for readers.

Excluded files:
  - README.md, CONTRIBUTING.md, CODE_OF_CONDUCT.md — top-level meta files where
    plain ``` blocks are intentional (e.g. illustrating raw Markdown syntax).
"""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent

_TOP_LEVEL_EXCLUDE = {"README.md", "CONTRIBUTING.md", "CODE_OF_CONDUCT.md"}


def _guide_markdown_files():
    return sorted(
        f
        for f in REPO_ROOT.rglob("*.md")
        if f.name not in _TOP_LEVEL_EXCLUDE or f.parent != REPO_ROOT
    )


def _untagged_code_block_lines(content: str) -> list[int]:
    """
    Return line numbers of fenced code block openers that have no language tag.

    Algorithm: walk lines tracking whether we are inside a fenced block.
      - A line whose stripped form is exactly ``` toggles the fence state.
        If we were outside, this opens an *untagged* block — record it.
      - A line whose stripped form starts with ``` followed by a non-backtick
        character opens a *tagged* block (``` python, ```bash, etc.).
      - The matching ``` closes any open block.
    """
    lines = content.splitlines()
    in_block = False
    untagged: list[int] = []

    for lineno, raw in enumerate(lines, start=1):
        stripped = raw.strip()
        if not in_block:
            if stripped.startswith("```"):
                in_block = True
                lang = stripped[3:].strip()
                if not lang:
                    untagged.append(lineno)
        else:
            if stripped == "```":
                in_block = False

    return untagged


@pytest.mark.parametrize(
    "md_file",
    _guide_markdown_files(),
    ids=lambda f: str(f.relative_to(REPO_ROOT)),
)
def test_code_blocks_have_language_tags(md_file: Path):
    """Every fenced code block must declare a language (e.g. ```python)."""
    content = md_file.read_text(encoding="utf-8")
    bad_lines = _untagged_code_block_lines(content)

    assert not bad_lines, (
        f"{md_file.relative_to(REPO_ROOT)} has {len(bad_lines)} code block(s) "
        f"without a language tag at line(s): {bad_lines}\n"
        "Fix: replace ``` with ```python, ```bash, ```sql, etc."
    )
