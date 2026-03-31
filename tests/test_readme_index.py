"""
Test: README.md index completeness
Ensures every intro_*.md guide file in a topic directory is referenced somewhere
in the root README.md.

Why this matters:
  - CONTRIBUTING.md review checklist states: "New guide is linked from README.md"
  - README.md is the primary navigation hub; unlisted guides are effectively invisible.
  - Currently there is no automated check, so a new guide can be merged and
    immediately become unreachable from the site's main index.

A guide is considered "referenced" if its relative path (e.g. ai_genai/intro_rag.md)
or its bare filename (e.g. intro_rag.md) appears anywhere in README.md.
"""

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


@pytest.fixture(scope="module")
def readme_content():
    return (REPO_ROOT / "README.md").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "guide_file",
    _guide_files(),
    ids=lambda f: str(f.relative_to(REPO_ROOT)),
)
def test_guide_is_referenced_in_readme(guide_file: Path, readme_content: str):
    """Every guide file must be referenced in the root README.md."""
    relative_path = str(guide_file.relative_to(REPO_ROOT))
    filename = guide_file.name

    referenced = (relative_path in readme_content) or (filename in readme_content)

    assert referenced, (
        f"{relative_path} is not referenced in README.md.\n"
        "Fix: add a link to this guide under the appropriate track section in README.md."
    )
