#!/usr/bin/env python3
"""Check that every relative link in the repo's Markdown files resolves.

Checks:
  * relative file links  -> the target file or directory exists
  * same-file anchors    -> "#anchor" matches a heading in that file
  * cross-file anchors   -> "other.md#anchor" matches a heading in other.md
  * index.html           -> every `path: '...'` entry in the study-hub nav exists

External (http/https/mailto) links are not fetched; this is an offline check
meant to run fast in CI.

Usage:
    python3 tools/check_links.py            # check the whole repo
    python3 tools/check_links.py FILE...    # check specific Markdown files
Exit code is 1 if any broken link is found.
"""
import os
import re
import sys
from functools import lru_cache

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = {".git", "node_modules", "MakeMeAwesomeScratch", "Research Papers"}

LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")


def slugify(heading):
    """GitHub-style heading anchor."""
    text = re.sub(r"<[^>]+>", "", heading)           # strip inline HTML
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # [label](url) -> label
    text = text.replace("`", "").strip().lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def outside_fences(lines):
    """Yield (lineno, line) for lines outside fenced code blocks."""
    in_fence = False
    for n, line in enumerate(lines, 1):
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            yield n, line


@lru_cache(maxsize=None)
def anchors_for(path):
    seen, anchors = {}, set()
    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")
    for _, line in outside_fences(lines):
        m = HEADING_RE.match(line)
        if not m:
            continue
        slug = slugify(m.group(2))
        # GitHub de-duplicates repeated headings with -1, -2, ...
        count = seen.get(slug, 0)
        anchors.add(slug if count == 0 else f"{slug}-{count}")
        seen[slug] = count + 1
    # explicit <a id="..."> / <a name="..."> anchors
    with open(path, encoding="utf-8") as f:
        anchors.update(re.findall(r"<a\s+(?:id|name)=\"([^\"]+)\"", f.read()))
    return anchors


def check_markdown(path):
    errors = []
    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")
    base = os.path.dirname(path)
    for n, line in outside_fences(lines):
        # links inside inline code spans are examples, not real links
        for target in LINK_RE.findall(re.sub(r"`[^`]*`", "", line)):
            if re.match(r"^[a-z][a-z0-9+.-]*:", target, re.I):   # http:, https:, mailto:
                continue
            file_part, _, anchor = target.partition("#")
            if not file_part:
                if anchor and anchor.lower() not in anchors_for(path):
                    errors.append((n, target, "anchor not found"))
                continue
            resolved = os.path.normpath(os.path.join(base, file_part))
            if not os.path.exists(resolved):
                errors.append((n, target, "file not found"))
            elif anchor and resolved.endswith(".md") and anchor.lower() not in anchors_for(resolved):
                errors.append((n, target, "anchor not found in target"))
    return errors


def check_index_html():
    path = os.path.join(ROOT, "index.html")
    if not os.path.exists(path):
        return []
    errors = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            for target in re.findall(r"path:\s*'([^']+)'", line):
                if not os.path.exists(os.path.join(ROOT, target)):
                    errors.append((n, target, "nav entry points to missing file"))
    return errors


def all_markdown():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in sorted(filenames):
            if name.endswith(".md"):
                yield os.path.join(dirpath, name)


def main(argv):
    files = [os.path.abspath(a) for a in argv] if argv else list(all_markdown())
    report = {path: check_markdown(path) for path in files}
    if not argv:
        report[os.path.join(ROOT, "index.html")] = check_index_html()
    broken = 0
    for path, errors in report.items():
        for n, target, why in errors:
            broken += 1
            print(f"{os.path.relpath(path, ROOT)}:{n}: {target}  ({why})")
    print(f"\nChecked {len(files)} Markdown files: {broken} broken link(s).")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
