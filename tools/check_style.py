#!/usr/bin/env python3
"""Flag typographic dashes that the repo's style guide replaces with plain punctuation.

The guides use colons, commas, parentheses or plain hyphens instead of em
dashes (U+2014) and en dashes (U+2013). This check keeps them from creeping
back in through new content.

Usage:
    python3 tools/check_style.py            # check the whole repo
    python3 tools/check_style.py FILE...    # check specific files
Exit code is 1 if any are found.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = {".git", "node_modules", "MakeMeAwesomeScratch", "Research Papers"}
EXTENSIONS = (".md", ".html", ".json")
BANNED = {"—": "em dash", "–": "en dash"}


def files_to_check(argv):
    if argv:
        return [os.path.abspath(a) for a in argv]
    out = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        out += [os.path.join(dirpath, f) for f in sorted(filenames) if f.endswith(EXTENSIONS)]
    return out


def main(argv):
    problems = 0
    files = files_to_check(argv)
    for path in files:
        with open(path, encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                for ch, name in BANNED.items():
                    if ch in line:
                        problems += 1
                        col = line.index(ch) + 1
                        print(f"{os.path.relpath(path, ROOT)}:{n}:{col}: {name} found; "
                              f"use a colon, comma, parentheses or a hyphen instead")
    print(f"\nChecked {len(files)} files: {problems} style problem(s).")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
