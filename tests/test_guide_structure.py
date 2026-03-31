"""
Test: Guide structure compliance
Ensures every topic guide (intro_*.md) contains the sections required by CONTRIBUTING.md.

Required sections per CONTRIBUTING.md:
  1. A single top-level heading  (# Title)
  2. Table of Contents           (## Table of Contents)
  3. At least one code example   (``` block)
  4. At least one Markdown table (| col | col |)
  5. An Interview Q&A section    (## ... Interview ... or #### question pattern)
  6. A Common Pitfalls section   (## Common Pitfall...)
  7. A Related Topics section    (## Related...)

Why this matters:
  - Analysis shows ~45 files are missing "Related Topics" and ~60 are missing
    "Common Pitfalls". These omissions make guides harder to navigate and less
    useful for interview preparation.
  - Without automated enforcement, new guides can be merged incomplete.

Scope: only `intro_*.md` files in topic directories, since those are the guides
described by CONTRIBUTING.md. README.md files in subdirectories and top-level
meta files are excluded.
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent

_TOPIC_DIRS = {
    "ai_genai",
    "classical_ml",
    "cloud_ml",
    "data_engineering",
    "deep_learning",
    "devops",
    "frameworks",
    "mlops",
    "system_design",
}


def _guide_files():
    """Return all intro_*.md files inside the recognised topic directories."""
    guides = []
    for d in _TOPIC_DIRS:
        topic_dir = REPO_ROOT / d
        if topic_dir.is_dir():
            guides.extend(topic_dir.glob("intro_*.md"))
    return sorted(guides)


# ---------------------------------------------------------------------------
# Individual structure checks
# ---------------------------------------------------------------------------

def _has_single_h1(content: str) -> bool:
    """File must have exactly one H1 heading."""
    h1s = re.findall(r"^# .+", content, re.MULTILINE)
    return len(h1s) == 1


def _has_table_of_contents(content: str) -> bool:
    """File must have a Table of Contents section."""
    return bool(re.search(r"^#{1,3}\s+Table of Contents", content, re.MULTILINE | re.IGNORECASE))


def _has_code_example(content: str) -> bool:
    """File must contain at least one fenced code block."""
    return "```" in content


def _has_markdown_table(content: str) -> bool:
    """File must contain at least one Markdown table (pipe-delimited row + separator)."""
    return bool(re.search(r"^\|.+\|", content, re.MULTILINE))


def _has_qa_section(content: str) -> bool:
    """File must have an interview Q&A section or #### level questions."""
    has_qa_heading = bool(re.search(
        r"^#{1,3}\s+.*(interview|q&a|questions? and answers?|interview questions?)",
        content, re.MULTILINE | re.IGNORECASE,
    ))
    # Alternatively: at least 3 level-4 headings that look like questions
    level4_questions = re.findall(r"^#{4}\s+.+\?", content, re.MULTILINE)
    return has_qa_heading or len(level4_questions) >= 3


def _has_common_pitfalls(content: str) -> bool:
    """File must have a Common Pitfalls / Common Mistakes section."""
    return bool(re.search(
        r"^#{1,3}\s+common (pitfall|mistake|error|issue)",
        content, re.MULTILINE | re.IGNORECASE,
    ))


def _has_related_topics(content: str) -> bool:
    """File must have a Related Topics / Related section."""
    return bool(re.search(r"^#{1,3}\s+related", content, re.MULTILINE | re.IGNORECASE))


# Map check name -> (check_function, fix hint)
_CHECKS = {
    "single H1 heading": (
        _has_single_h1,
        "Add exactly one `# Title` at the top of the file.",
    ),
    "Table of Contents section": (
        _has_table_of_contents,
        "Add `## Table of Contents` with links to each major section.",
    ),
    "at least one code example": (
        _has_code_example,
        "Add a working code example in a fenced code block (e.g. ```python).",
    ),
    "at least one Markdown table": (
        _has_markdown_table,
        "Add a comparison or summary table using pipe syntax.",
    ),
    "Interview Q&A section": (
        _has_qa_section,
        "Add `## Interview Questions` with at least 8 Q&A pairs.",
    ),
    "Common Pitfalls section": (
        _has_common_pitfalls,
        "Add `## Common Pitfalls` section (table format: Problem | Fix).",
    ),
    "Related Topics section": (
        _has_related_topics,
        "Add `## Related Topics` section with links to related guides.",
    ),
}


@pytest.mark.parametrize(
    "md_file",
    _guide_files(),
    ids=lambda f: str(f.relative_to(REPO_ROOT)),
)
def test_guide_has_required_sections(md_file: Path):
    """Every intro_*.md guide must contain all sections required by CONTRIBUTING.md."""
    content = md_file.read_text(encoding="utf-8")
    missing = []

    for name, (check_fn, hint) in _CHECKS.items():
        if not check_fn(content):
            missing.append(f"  MISSING {name}\n    Hint: {hint}")

    assert not missing, (
        f"{md_file.relative_to(REPO_ROOT)} is missing required sections:\n"
        + "\n".join(missing)
    )
