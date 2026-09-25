#!/usr/bin/env python3
"""Build data/questions.json, the question bank behind practice.html.

Sources:
  * every Markdown guide's interview section -- "## Interview Q&A" (and the
    common spellings "Interview Questions", "Interview Q and A",
    "Common Interview Questions", ...). Questions are written as
      - "###" / "####" headings (answer = content until the next heading of
        the same or a higher level), or
      - bold lead lines such as "**Q3: What is X?** 🟡 Intermediate", or
      - numbered bold items such as "1. **Question?** -> short answer".
  * README.md's "# Classic Question Bank" ("#### " headings; answers are
    indented blocks or prose/code until the next "####", "##" or the
    "REFERENCED FROM" line).

Links inside answers are rewritten so they work from the repo root, where
practice.html lives: relative ".md" links become Study Hub routes
("index.html#<repo-path>") and relative images/files become repo-root paths.

Usage:
    python3 tools/build_question_bank.py           # (re)write data/questions.json
    python3 tools/build_question_bank.py --check   # exit 1 if the file is stale
Standard library only.
"""
import argparse
import hashlib
import json
import os
import posixpath
import re
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(ROOT, "data", "questions.json")
SKIP_DIRS = {".git", "node_modules", "MakeMeAwesomeScratch", "Research Papers", "data", "tools"}
SCHEMA_VERSION = 1

FENCE_RE = re.compile(r"^\s*(```|~~~)")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
SECTION_RE = re.compile(
    r"^(?:\d+\.\s*)?(?:common\s+|advanced\s+)?interview\s+"
    r"(?:q\s*(?:&|and)\s*a|questions(?:\s*(?:&|and)\s*answers)?)"
    r"(?:\s*\(.*\))?\s*:?$",
    re.I,
)
# Headings inside an interview section that are not questions and end it.
STOP_HEADING_RE = re.compile(
    r"^(related|see also|further reading|references|resources|next steps|"
    r"summary|key takeaways|sources|navigation)\b", re.I)
Q_PREFIX_RE = re.compile(r"^Q\s*\d*\s*[:.)]\s*", re.I)
STRONG_BOLD_Q_RE = re.compile(r"^\*\*(Q\s*\d*\s*[:.)].+?)\*\*(.*)$", re.I)
WEAK_BOLD_Q_RE = re.compile(r"^\*\*([^*].*?)\*\*(.*)$")
NUM_BOLD_Q_RE = re.compile(r"^\d+[.)]\s+\*\*(.+?)\*\*\s*(.*)$")
DIFF_RE = re.compile(r"(?:🟢|🟡|🔴)?\s*\b(Beginner|Intermediate|Advanced)\b", re.I)
LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s")
LINK_RE = re.compile(r"(!?)\[((?:[^\[\]]|\[[^\[\]]*\])*)\]\(\s*<?([^)\s>]+)>?(\s+\"[^\"]*\")?\s*\)")
HTML_SRC_RE = re.compile(r"""(<img\b[^>]*?\bsrc=)(["'])([^"']+)\2""", re.I)
SRC_NOISE_RE = re.compile(
    r"\s*\[\[[^\[\]]*\]\]\([^)]*\)"        # [[src]](url)
    r"|\s*\[\[[^\[\]]*\]\([^)]*\)\]"       # [[src](url)]
)


# ─────────────────────────────────────────────────────────────── helpers
def repo_files():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
        for name in sorted(filenames):
            if name.lower().endswith(".md"):
                yield os.path.relpath(os.path.join(dirpath, name), ROOT).replace(os.sep, "/")


def fence_mask(lines):
    """For each line, True if it is inside (or delimits) a fenced code block."""
    mask, inside, marker = [], False, None
    for line in lines:
        m = FENCE_RE.match(line)
        if m and (not inside or m.group(1) == marker):
            mask.append(True)
            inside, marker = (not inside), m.group(1)
            continue
        mask.append(inside)
    return mask


def clean_inline(text):
    """Markdown-ish heading/bold text -> plain question string."""
    text = SRC_NOISE_RE.sub("", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)   # [label](url) -> label
    text = text.replace("**", "").replace("__", "")
    text = re.sub(r"(?<![\w*])[*_]([^*_\s][^*_]*?)[*_](?![\w*])", r"\1", text)  # *not* -> not
    text = Q_PREFIX_RE.sub("", text.strip())
    text = re.sub(r"^\d+\)\s+", "", text)                   # "12) What ..." (classic bank)
    text = re.sub(r"\s+", " ", text).strip()
    return text.rstrip(":").strip()


def doc_title(lines, mask, fallback):
    for line, fenced in zip(lines, mask):
        if fenced:
            continue
        m = re.match(r"^#\s+(.+?)\s*#*\s*$", line)
        if m:
            t = re.sub(r"[`*_]", "", m.group(1))
            t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)
            # Drop leading emoji / symbols ("🤖 LangChain" -> "LangChain")
            t = re.sub(r"^[^\w(]+", "", t).strip()
            return t or fallback
    return fallback


def resolve(src_path, target):
    base = posixpath.dirname(src_path)
    return posixpath.normpath(posixpath.join(base, target)).lstrip("/")


def rewrite_url(src_path, url, is_image):
    if url.startswith("#"):
        # A same-page anchor in the guide: practice.html is a different page,
        # so send the reader to the guide itself.
        return url if is_image else "index.html#" + src_path
    if re.match(r"^(?:[a-z][a-z0-9+.-]*:|//)", url, re.I):
        return url  # absolute, mailto:, data:
    path, _, frag = url.partition("#")
    path = path.split("?")[0]
    if not path:
        return url
    resolved = resolve(src_path, path)
    if resolved.startswith(".."):
        return url
    full = os.path.join(ROOT, resolved)
    if not is_image:
        if resolved.lower().endswith(".md"):
            return "index.html#" + resolved
        if os.path.isdir(full) and os.path.isfile(os.path.join(full, "README.md")):
            return "index.html#" + posixpath.join(resolved, "README.md")
    return resolved + ("#" + frag if frag and not is_image else "")


def rewrite_links(src_path, md):
    lines = md.split("\n")
    mask = fence_mask(lines)
    out = []
    for line, fenced in zip(lines, mask):
        if not fenced and "](" in line:
            line = LINK_RE.sub(
                lambda m: "%s[%s](%s%s)" % (m.group(1), m.group(2),
                                            rewrite_url(src_path, m.group(3), bool(m.group(1))),
                                            m.group(4) or ""), line)
        if not fenced and "<img" in line.lower():
            line = HTML_SRC_RE.sub(
                lambda m: m.group(1) + m.group(2) + rewrite_url(src_path, m.group(3), True) + m.group(2), line)
        out.append(line)
    return "\n".join(out)


def tidy_answer(lines):
    """Trim blank lines / trailing horizontal rules; unquote all-blockquote answers."""
    lines = [l.rstrip() for l in lines]
    while lines and (not lines[-1].strip() or re.match(r"^\s*(-{3,}|\*{3,}|_{3,})\s*$", lines[-1])):
        lines.pop()
    while lines and not lines[0].strip():
        lines.pop(0)
    body = [l for l in lines if l.strip()]
    if body and all(l.lstrip().startswith(">") for l in body):
        lines = [re.sub(r"^\s*>\s?", "", l) for l in lines]
    return "\n".join(lines).strip("\n")


def split_difficulty(rest):
    """'🟡 Intermediate' after a bold question -> ('intermediate', '')."""
    m = DIFF_RE.search(rest)
    if m and len(DIFF_RE.sub("", rest).strip(" \t—–-|·")) == 0:
        return m.group(1).lower(), ""
    return None, rest.strip()


def estimate_difficulty(question, answer):
    q = question.lower()
    words = len(re.findall(r"\w+", answer))
    if re.search(r"\b(design|architect|at scale|trade-?offs?|production|debug|million|billion|walk me through)\b", q):
        return "advanced" if words > 120 else "intermediate"
    if re.match(r"^(what is|what are|define|what does)\b", q) and words < 140:
        return "beginner"
    if words < 55:
        return "beginner"
    if words > 150:
        return "advanced"
    return "intermediate"


# ─────────────────────────────────────────────────────────────── guides
def interview_sections(lines, mask):
    """Yield (start, end, level) line ranges of interview sections (exclusive)."""
    i, n = 0, len(lines)
    while i < n:
        m = None if mask[i] else HEADING_RE.match(lines[i])
        if m and len(m.group(1)) == 2 and SECTION_RE.match(clean_inline(m.group(2))):
            j = i + 1
            while j < n:
                h = None if mask[j] else HEADING_RE.match(lines[j])
                if h and len(h.group(1)) <= 2:
                    break
                j += 1
            yield i + 1, j
            i = j
        else:
            i += 1


def parse_section(lines, mask):
    """Return [(question, answer_lines, difficulty_or_None)] for one section."""
    n = len(lines)
    # Truncate at the first "stop" heading (Related Guides, ...).
    for k in range(n):
        h = None if mask[k] else HEADING_RE.match(lines[k])
        if h and STOP_HEADING_RE.match(clean_inline(h.group(2))):
            lines, mask, n = lines[:k], mask[:k], k
            break

    headings = []
    for k in range(n):
        h = None if mask[k] else HEADING_RE.match(lines[k])
        if h:
            headings.append((k, len(h.group(1)), h.group(2)))

    def bold_starts(regex, need_question):
        out = []
        for k in range(n):
            if mask[k]:
                continue
            m = regex.match(lines[k])
            if not m:
                continue
            q = m.group(1).strip()
            if need_question and not clean_inline(q).rstrip(" .").endswith("?"):
                continue
            # Only whole-line bold leads (optionally followed by a difficulty tag or
            # an inline answer); skip "**Label:** text" definition lines.
            if clean_inline(q).endswith(":") or q.rstrip().endswith(":"):
                continue
            out.append(k)
        return out

    strong = bold_starts(STRONG_BOLD_Q_RE, False)
    weak = [] if strong else [k for k in bold_starts(WEAK_BOLD_Q_RE, True)
                              if not WEAK_BOLD_Q_RE.match(lines[k]).group(2).strip()
                              or split_difficulty(WEAK_BOLD_Q_RE.match(lines[k]).group(2))[0]]
    bold = strong or weak
    results = []

    if bold:
        regex = STRONG_BOLD_Q_RE if strong else WEAK_BOLD_Q_RE
        head_lines = {k for k, _, _ in headings}
        for idx, k in enumerate(bold):
            m = regex.match(lines[k])
            diff, rest = split_difficulty(m.group(2))
            end = bold[idx + 1] if idx + 1 < len(bold) else n
            nxt = [h for h in head_lines if k < h < end]
            if nxt:
                end = min(nxt)
            body = ([rest] if rest else []) + lines[k + 1:end]
            results.append((clean_inline(m.group(1)), body, diff))
        return results

    q_heads = [(k, lvl, t) for k, lvl, t in headings if lvl >= 3]
    if q_heads:
        for idx, (k, lvl, text) in enumerate(q_heads):
            end = n
            for k2, lvl2, _ in headings:
                if k2 > k and lvl2 <= lvl:
                    end = k2
                    break
            # Headings that only group deeper question headings are categories.
            if any(k < k2 < end and lvl2 > lvl for k2, lvl2, _ in q_heads):
                continue
            q = clean_inline(text)
            diff, _ = split_difficulty(q) if DIFF_RE.search(q) else (None, q)
            q = DIFF_RE.sub("", q).strip(" —–-|") if diff else q
            results.append((q, lines[k + 1:end], diff))
        return results

    # Numbered bold items: "1. **Question?** -> answer" with indented continuations.
    starts = [k for k in range(n) if not mask[k] and NUM_BOLD_Q_RE.match(lines[k])]
    for idx, k in enumerate(starts):
        m = NUM_BOLD_Q_RE.match(lines[k])
        rest = re.sub(r"^(?:→|->|—|–|:)\s*", "", m.group(2).strip())
        end = starts[idx + 1] if idx + 1 < len(starts) else n
        body = [rest]
        for line in lines[k + 1:end]:
            if line.strip() and not line.startswith((" ", "\t")):
                break
            body.append(line.strip())
        results.append((clean_inline(m.group(1)), body, None))
    return results


def parse_guide(path, text):
    lines = text.split("\n")
    mask = fence_mask(lines)
    title = doc_title(lines, mask, posixpath.basename(path))
    found = []
    for start, end in interview_sections(lines, mask):
        found.extend(parse_section(lines[start:end], mask[start:end]))
    return title, found


# ─────────────────────────────────────────────────────── classic bank
def fence_indented(lines):
    """Turn README-style 4+-space indented answer blocks into ```text fences."""
    out, i, n = [], 0, len(lines)
    mask = fence_mask(lines)
    prev_content = ""
    while i < n:
        line = lines[i]
        if mask[i]:                      # already inside a ``` fence: keep verbatim
            out.append(line)
            prev_content = "```"
            i += 1
            continue
        indented = line.startswith("    ") or line.startswith("\t")
        if indented and line.strip() and not LIST_ITEM_RE.match(prev_content) and not prev_content.startswith((" ", "\t")):
            j = i
            block = []
            while j < n and not mask[j] and (lines[j].startswith(("    ", "\t")) or not lines[j].strip()):
                block.append(lines[j].replace("\t", "    "))
                j += 1
            while block and not block[-1].strip():
                block.pop()
                j -= 1
            indent = min(len(b) - len(b.lstrip()) for b in block if b.strip())
            if out and out[-1].strip():
                out.append("")
            out.append("```text")
            out.extend(b[indent:].rstrip() for b in block)
            out.append("```")
            if j < n and lines[j].strip():
                out.append("")
            prev_content = "```"
            i = j
            continue
        out.append(line)
        if line.strip():
            prev_content = line
        i += 1
    return out


def parse_classic(text):
    lines = text.split("\n")
    mask = fence_mask(lines)
    start = next((k for k, l in enumerate(lines) if not mask[k] and re.match(r"^#\s+Classic Question Bank\s*$", l)), None)
    if start is None:
        return []
    items, cur = [], None
    for k in range(start + 1, len(lines)):
        line = lines[k]
        if not mask[k]:
            if re.match(r"^#{1,2}\s", line):
                break
            if line.startswith("#### "):
                cur = [clean_inline(line[5:]), []]
                items.append(cur)
                continue
            if line.strip().upper().startswith("REFERENCED FROM"):
                cur = None
                continue
        if cur is not None:
            cur[1].append(line)
    return [(q, fence_indented(body), None) for q, body in items if q]


# ─────────────────────────────────────────────────────────────── build
def make_id(source, question, seen):
    base = hashlib.sha1((source + "\n" + question).encode("utf-8")).hexdigest()[:12]
    ident, n = base, 2
    while ident in seen:
        ident = "%s-%d" % (base, n)
        n += 1
    seen.add(ident)
    return ident


def build():
    records, seen = [], set()

    def add(path, title, track, pairs):
        for order, (q, body, diff) in enumerate(pairs):
            answer = rewrite_links(path, tidy_answer(body))
            if not q or not answer.strip():
                continue
            records.append({
                "id": make_id(path, q, seen),
                "question": q,
                "answer_md": answer,
                "source_path": path,
                "source_title": title,
                "track": track,
                "difficulty": diff or estimate_difficulty(q, answer),
                "difficulty_source": "author" if diff else "heuristic",
                "order": order,
            })

    for path in repo_files():
        with open(os.path.join(ROOT, path), encoding="utf-8") as fh:
            text = fh.read().replace("\r\n", "\n")
        if path == "README.md":
            add(path, "Classic Question Bank", "classic", parse_classic(text))
            continue
        if "/" not in path:
            continue
        title, pairs = parse_guide(path, text)
        add(path, title, path.split("/")[0], pairs)

    records.sort(key=lambda r: (r["track"] != "classic", r["track"], r["source_path"], r["order"]))
    tracks = Counter(r["track"] for r in records)
    return {
        "schema": SCHEMA_VERSION,
        "generator": "tools/build_question_bank.py",
        "count": len(records),
        "tracks": dict(sorted(tracks.items())),
        "sources": len({r["source_path"] for r in records}),
        "questions": records,
    }


def serialize(data):
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="do not write; exit 1 if data/questions.json is missing or stale")
    ap.add_argument("--quiet", action="store_true", help="only print errors")
    args = ap.parse_args(argv)

    data = build()
    payload = serialize(data)
    if not args.quiet:
        width = max(len(t) for t in data["tracks"]) if data["tracks"] else 5
        for track, n in data["tracks"].items():
            print("  %-*s %4d" % (width, track, n))
        print("  %-*s %4d questions from %d sources" % (width, "total", data["count"], data["sources"]))

    rel = os.path.relpath(OUT_PATH, ROOT)
    if args.check:
        try:
            with open(OUT_PATH, encoding="utf-8") as fh:
                current = fh.read()
        except OSError:
            current = None
        if current != payload:
            print("%s is %s. Run: python3 tools/build_question_bank.py"
                  % (rel, "missing" if current is None else "stale"), file=sys.stderr)
            return 1
        if not args.quiet:
            print("%s is up to date." % rel)
        return 0

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(payload)
    if not args.quiet:
        print("Wrote %s" % rel)
    return 0


if __name__ == "__main__":
    sys.exit(main())
