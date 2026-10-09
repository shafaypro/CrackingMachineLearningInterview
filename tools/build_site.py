#!/usr/bin/env python3
"""Build a static, crawlable HTML page for every Markdown guide.

The study hub (index.html) renders guides in the browser from
`index.html#path.md`. Search engines treat everything after `#` as one page
and do not reliably run the JavaScript that loads the text, so on their own
the guides are invisible in search. This script writes:

  * guides/<path>.html for every guide (README.md files become index.html),
    each with its own <title>, meta description, canonical URL, Open Graph
    and Twitter tags, and schema.org TechArticle + BreadcrumbList data;
  * guides/index.html, a crawlable list of every guide grouped by track;
  * sitemap.xml at the repo root.

Usage:
    pip install markdown-it-py==4.0.0
    python3 tools/build_site.py           # write the files
    python3 tools/build_site.py --check   # exit 1 if any output is stale

The output is deterministic (sorted inputs, no timestamps), so CI can check
that it is up to date the same way it checks data/questions.json.
"""

import html
import json
import os
import posixpath
import re
import sys
from urllib.parse import quote, unquote

from markdown_it import MarkdownIt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_links import SKIP_DIRS, slugify  # noqa: E402  (same anchors as the link checker)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITE_URL = "https://shafaypro.github.io/CrackingMachineLearningInterview/"
REPO_URL = "https://github.com/shafaypro/CrackingMachineLearningInterview"
SITE_NAME = "Cracking ML Interviews"
OUT_DIR = "guides"
EXCLUDE_DIRS = SKIP_DIRS | {".github", OUT_DIR}
DESC_LEN = 155
DESC_OVERRIDES = {
    "README.md": ("Hundreds of classic machine learning interview questions with answers: bias and "
                  "variance, regularization, evaluation metrics, trees, SVMs, clustering and more, "
                  "plus links to every study track."),
}
SKIP_DESC = ("table of contents", "please check", "referenced from", "back to", "note:")
TOP_PAGES = ["", "practice.html", "flashcards.html"]

TRACK_RE = re.compile(r"title:\s*'((?:[^'\\]|\\.)*)'")
FILE_RE = re.compile(r"\{\s*label:\s*'((?:[^'\\]|\\.)*)',\s*path:\s*'([^']+)'\s*\}")
HTML_ATTR_RE = re.compile(r"""(\b(?:src|href)=)(["'])([^"']+)\2""", re.I)
TAG_RE = re.compile(r"<[^>]+>")
EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF☀-➿⬀-⯿️‍]+")


# ── inputs ──────────────────────────────────────────────────────────────────

def find_guides():
    """Every Markdown file the link checker covers, as sorted repo paths."""
    out = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDE_DIRS)
        for name in filenames:
            if name.lower().endswith(".md"):
                rel = os.path.relpath(os.path.join(dirpath, name), ROOT)
                out.append(rel.replace(os.sep, "/"))
    return sorted(out)


def read_tracks():
    """[(track title, [(label, path), ...]), ...] in study-hub order."""
    with open(os.path.join(ROOT, "index.html"), encoding="utf-8") as f:
        src = f.read()
    start = src.index("const TRACKS = [")
    src = src[start:src.index("\n];", start)]
    tracks = []
    for block in re.split(r"\n  \{\n", src)[1:]:
        title = TRACK_RE.search(block)
        files = FILE_RE.findall(block)
        if title and files:
            unesc = lambda s: s.replace("\\'", "'")
            tracks.append((unesc(title.group(1)),
                           [(unesc(l), p) for l, p in files]))
    return tracks


# ── paths ───────────────────────────────────────────────────────────────────

def out_path(md):
    """Repo path of the generated page for a guide."""
    head, name = posixpath.split(md)
    stem = "index" if name.lower() == "readme.md" and head else name[:-3]
    if not head and name.lower() == "readme.md":
        stem = "readme"
    return posixpath.join(OUT_DIR, head, stem + ".html")


def page_url(repo_path):
    """Absolute public URL for a repo path (index.html shown as its folder)."""
    p = repo_path
    if p == "index.html" or p.endswith("/index.html"):
        p = p[: -len("index.html")]
    return SITE_URL + quote(p)


def rel_link(from_file, to_path):
    """Relative URL from one generated file to a repo path."""
    rel = posixpath.relpath(to_path, posixpath.dirname(from_file) or ".")
    if rel.endswith("/index.html"):
        rel = rel[: -len("index.html")]
    elif rel == "index.html":
        rel = "./"
    return quote(rel, safe="/#.-_~")


# ── markdown rendering ──────────────────────────────────────────────────────

def make_parser():
    md = MarkdownIt("commonmark", {"html": True, "linkify": False})
    md.enable(["table", "strikethrough"])
    return md


def rewrite_target(target, src_md, out_file, guides):
    """Point a link from the Markdown source at the right file on the site."""
    if not target or target.startswith(("#", "mailto:", "data:")) or \
            re.match(r"^[a-z][a-z0-9+.-]*:", target, re.I) or target.startswith("//"):
        return target
    path, _, frag = target.partition("#")
    path = unquote(path.split("?", 1)[0])
    resolved = posixpath.normpath(posixpath.join(posixpath.dirname(src_md), path))
    if resolved == ".":
        resolved = ""
    if resolved.startswith(".."):
        return target
    if resolved in guides:
        dest = out_path(resolved)
    elif os.path.isdir(os.path.join(ROOT, resolved or ".")):
        readme = posixpath.join(resolved, "README.md") if resolved else "README.md"
        if readme not in guides:
            return f"{REPO_URL}/tree/master/{quote(resolved)}"
        dest = out_path(readme)
    else:
        dest = resolved          # images, scripts, practice.html, data files ...
    link = rel_link(out_file, dest)
    return link + ("#" + frag if frag else "")


def plain_text(inline_src):
    """Readable text from a line of inline Markdown."""
    t = TAG_RE.sub("", inline_src)
    t = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", t)
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)
    t = re.sub(r"[*_`~]+", "", t)
    t = EMOJI_RE.sub("", t)
    return re.sub(r"\s+", " ", t).strip()


def render(md_parser, src_md, out_file, guides):
    with open(os.path.join(ROOT, src_md), encoding="utf-8") as f:
        text = f.read()
    tokens = md_parser.parse(text)
    title, intro, seen, faq = None, [], {}, 0
    in_qa = past_intro = False
    for i, tok in enumerate(tokens):
        if tok.type == "heading_open":
            raw = tokens[i + 1].content
            slug = slugify(raw)
            n = seen.get(slug, 0)
            seen[slug] = n + 1
            tok.attrSet("id", slug if n == 0 else f"{slug}-{n}")
            level = int(tok.tag[1])
            if level == 1 and title is None:
                title = plain_text(raw)
            if level == 2:
                past_intro = True
                in_qa = bool(re.search(r"interview|q\s*&\s*a|questions", raw, re.I))
            elif in_qa and level >= 3 and raw.rstrip().endswith("?"):
                faq += 1
        elif (tok.type == "paragraph_open" and tok.level == 0 and title is not None
              and not past_intro and sum(map(len, intro)) < 120):
            # Description: the opening prose, skipping lead-ins that end in ":"
            # and attribution or navigation lines.
            candidate = plain_text(tokens[i + 1].content)
            if (len(candidate) >= 30 and not candidate.endswith(":")
                    and not candidate.lower().startswith(SKIP_DESC)):
                intro.append(candidate)
        if tok.type == "inline" and tok.children:
            for child in tok.children:
                if child.type == "link_open":
                    child.attrSet("href", rewrite_target(child.attrGet("href"), src_md, out_file, guides))
                elif child.type == "image":
                    child.attrSet("src", rewrite_target(child.attrGet("src"), src_md, out_file, guides))
                    child.attrSet("loading", "lazy")
                elif child.type == "html_inline":
                    child.content = HTML_ATTR_RE.sub(
                        lambda m: m.group(1) + m.group(2) + rewrite_target(m.group(3), src_md, out_file, guides) + m.group(2),
                        child.content)
        elif tok.type == "html_block":
            tok.content = HTML_ATTR_RE.sub(
                lambda m: m.group(1) + m.group(2) + rewrite_target(m.group(3), src_md, out_file, guides) + m.group(2),
                tok.content)
    body = md_parser.renderer.render(tokens, md_parser.options, {})
    body = body.replace("<table>", '<div class="table-wrap"><table>').replace("</table>", "</table></div>")
    if title is None:
        title = posixpath.basename(src_md)[:-3].replace("_", " ").replace("-", " ").title()
    desc = DESC_OVERRIDES.get(src_md) or " ".join(intro) or f"{title}: concepts, worked examples and interview questions with answers."
    return title, shorten(desc), body, faq


def shorten(text, limit=DESC_LEN):
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].rstrip(",;:.")
    return cut + "…"


# ── page template ───────────────────────────────────────────────────────────

THEME_BOOT = ("try{var t=localStorage.getItem('cml-theme');"
              "if(t==='light'||t==='dark')document.documentElement.dataset.theme=t;}catch(e){}")
HLJS = "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/"


def json_ld(data):
    return json.dumps(data, ensure_ascii=False, indent=1).replace("</", "<\\/")


def head(out_file, title, desc, url, ld, og_type="article"):
    css = rel_link(out_file, f"{OUT_DIR}/assets/guide.css")
    e = html.escape
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{e(title)}</title>
<meta name="description" content="{e(desc)}">
<link rel="canonical" href="{e(url)}">
<meta property="og:type" content="{og_type}">
<meta property="og:site_name" content="{SITE_NAME}">
<meta property="og:title" content="{e(title)}">
<meta property="og:description" content="{e(desc)}">
<meta property="og:url" content="{e(url)}">
<meta name="twitter:card" content="summary">
<meta name="twitter:title" content="{e(title)}">
<meta name="twitter:description" content="{e(desc)}">
<meta name="color-scheme" content="light dark">
<script>{THEME_BOOT}</script>
<link rel="stylesheet" href="{css}">
<link rel="stylesheet" href="{HLJS}styles/github.min.css" media="(prefers-color-scheme: light)">
<link rel="stylesheet" href="{HLJS}styles/github-dark.min.css" media="(prefers-color-scheme: dark)">
<script src="{HLJS}highlight.min.js" defer></script>
<script>document.addEventListener('DOMContentLoaded',function(){{if(window.hljs)hljs.highlightAll();}});</script>
<script type="application/ld+json">
{json_ld(ld)}
</script>
</head>
"""


def site_header(out_file):
    r = lambda p: rel_link(out_file, p)
    return f"""<body>
<a class="skip" href="#content">Skip to content</a>
<header class="top">
  <a class="brand" href="{r('index.html')}"><span class="mark" aria-hidden="true">ML</span>{SITE_NAME}</a>
  <nav aria-label="Site">
    <a href="{r('index.html')}">Study hub</a>
    <a href="{r(OUT_DIR + '/index.html')}">All guides</a>
    <a href="{r('practice.html')}">Practice</a>
    <a href="{r('flashcards.html')}">Flashcards</a>
    <a href="{REPO_URL}">GitHub</a>
  </nav>
</header>
"""


def footer():
    return f"""<footer class="foot">
  <p>Free and open source on <a href="{REPO_URL}">GitHub</a>. Corrections and new questions are welcome.</p>
</footer>
</body>
</html>
"""


def guide_page(src_md, out_file, title, desc, body, faq, crumbs, prev_next):
    url = page_url(out_file)
    page_title = title
    if "/" in src_md and not src_md.startswith("docs/") and not re.search(r"interview|guide", title, re.I) \
            and len(title) <= 40:
        page_title += " Interview Guide"
    page_title = f"{page_title} | {SITE_NAME}"
    ld = {
        "@context": "https://schema.org",
        "@graph": [
            {
                "@type": "TechArticle",
                "headline": title[:110],
                "description": desc,
                "url": url,
                "inLanguage": "en",
                "isAccessibleForFree": True,
                "isPartOf": {"@type": "WebSite", "name": SITE_NAME, "url": SITE_URL},
                "author": {"@type": "Organization", "name": SITE_NAME, "url": REPO_URL},
            },
            {
                "@type": "BreadcrumbList",
                "itemListElement": [
                    {"@type": "ListItem", "position": i + 1, "name": name,
                     "item": page_url(path) + (f"#{frag}" if frag else "")}
                    for i, (name, path, frag) in enumerate(crumbs)
                ],
            },
        ],
    }
    e = html.escape
    crumb_html = " <span aria-hidden=\"true\">/</span> ".join(
        f'<a href="{e(rel_link(out_file, path) + (f"#{frag}" if frag else ""))}">{e(n)}</a>'
        for n, path, frag in crumbs[:-1]) + f' <span aria-hidden="true">/</span> <span aria-current="page">{e(crumbs[-1][0])}</span>'
    hub = rel_link(out_file, "index.html") + "#" + quote(src_md, safe="/")
    meta = f'<a href="{e(hub)}">Open in the interactive study hub</a>'
    if faq:
        meta += f' <span aria-hidden="true">·</span> {faq} interview questions, also in the <a href="{e(rel_link(out_file, "practice.html"))}">practice drill</a>'
    pn = ""
    if prev_next[0] or prev_next[1]:
        cells = []
        for rel, (lbl, md) in zip(("prev", "next"), prev_next):
            if md:
                word = "Previous" if rel == "prev" else "Next"
                cells.append(f'<a rel="{rel}" class="{rel}" href="{e(rel_link(out_file, out_path(md)))}"><small>{word}</small>{e(lbl)}</a>')
            else:
                cells.append("<span></span>")
        pn = f'<nav class="pager" aria-label="Track">{"".join(cells)}</nav>\n'
    edit = f"{REPO_URL}/blob/master/{quote(src_md)}"
    return (head(out_file, page_title, desc, url, ld)
            + site_header(out_file)
            + f'<main id="content">\n<nav class="crumbs" aria-label="Breadcrumb">{crumb_html}</nav>\n'
            + f'<p class="meta">{meta}</p>\n<article class="md">\n{body}</article>\n'
            + pn
            + f'<p class="edit"><a href="{e(edit)}">View or edit this guide on GitHub</a></p>\n</main>\n'
            + footer())


def index_page(sections, titles):
    out_file = f"{OUT_DIR}/index.html"
    url = page_url(out_file)
    title = f"All interview prep guides | {SITE_NAME}"
    desc = ("Every guide in the Cracking ML Interviews study hub: machine learning, deep learning, GenAI, "
            "MLOps, AWS, GCP and Azure, data engineering, system design and coding practice.")
    e = html.escape
    items, parts = [], []
    for name, entries in sections:
        lis = "\n".join(
            f'    <li><a href="{e(rel_link(out_file, out_path(md)))}">{e(label)}</a></li>'
            for label, md in entries)
        parts.append(f'<section>\n  <h2 id="{e(slugify(name))}">{e(name)}</h2>\n  <ul>\n{lis}\n  </ul>\n</section>')
        items += [page_url(out_path(md)) for _, md in entries]
    ld = {
        "@context": "https://schema.org",
        "@type": "CollectionPage",
        "name": title,
        "description": desc,
        "url": url,
        "mainEntity": {
            "@type": "ItemList",
            "numberOfItems": len(items),
            "itemListElement": [{"@type": "ListItem", "position": i + 1, "url": u} for i, u in enumerate(items)],
        },
    }
    return (head(out_file, title, desc, url, ld, og_type="website")
            + site_header(out_file)
            + '<main id="content">\n<h1>All guides</h1>\n'
            + f'<p class="lede">{len(items)} free guides for machine learning, AI and data interviews. '
            + f'Open any of them in the <a href="{rel_link(out_file, "index.html")}">interactive study hub</a>, '
            + f'or drill the questions in the <a href="{rel_link(out_file, "practice.html")}">practice mode</a>.</p>\n'
            + "\n".join(parts) + "\n</main>\n" + footer())


def sitemap(urls):
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">']
    lines += [f"  <url><loc>{html.escape(u)}</loc></url>" for u in urls]
    lines.append("</urlset>")
    return "\n".join(lines) + "\n"


# ── build ───────────────────────────────────────────────────────────────────

def build():
    guides = find_guides()
    guide_set = set(guides)
    tracks = read_tracks()
    parser = make_parser()

    # Track membership drives breadcrumbs, prev/next and the index grouping.
    track_of, order = {}, []
    for t_title, files in tracks:
        files = [(l, p) for l, p in files if p in guide_set]
        for i, (label, path) in enumerate(files):
            if path in track_of:
                continue
            prev = files[i - 1] if i > 0 else (None, None)
            nxt = files[i + 1] if i + 1 < len(files) else (None, None)
            track_of[path] = (t_title, label, prev, nxt)
        order.append((t_title, files))

    outputs, titles = {}, {}
    for md in guides:
        out = out_path(md)
        title, desc, body, faq = render(parser, md, out, guide_set)
        titles[md] = title
        if md in track_of:
            t_title, label, prev, nxt = track_of[md]
        else:
            folder = posixpath.dirname(md)
            section = f"More in {folder}" if folder else "Repository documents"
            t_title, label, prev, nxt = section, title, (None, None), (None, None)
        crumbs = [("Study hub", "index.html", ""),
                  ("All guides", f"{OUT_DIR}/index.html", ""),
                  (t_title, f"{OUT_DIR}/index.html", slugify(t_title)),
                  (label, out, "")]
        outputs[out] = guide_page(md, out, title, desc, body, faq, crumbs, (prev, nxt))

    # Guides that are not in any study-hub track still get listed, by folder.
    listed = {p for _, files in order for _, p in files}
    extra = {}
    for md in guides:
        if md not in listed:
            extra.setdefault(posixpath.dirname(md) or "Repository", []).append((titles[md], md))
    sections = [(t, f) for t, f in order if f] + \
               [(f"More in {k}" if k != "Repository" else "Repository documents", v) for k, v in sorted(extra.items())]
    outputs[f"{OUT_DIR}/index.html"] = index_page(sections, titles)

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "guide.css"), encoding="utf-8") as f:
        outputs[f"{OUT_DIR}/assets/guide.css"] = f.read()

    urls = [SITE_URL + p for p in TOP_PAGES] + [page_url(f"{OUT_DIR}/index.html")]
    urls += [page_url(out_path(md)) for md in guides]
    outputs["sitemap.xml"] = sitemap(urls)
    return outputs


def existing_outputs():
    found = set()
    base = os.path.join(ROOT, OUT_DIR)
    for dirpath, _, filenames in os.walk(base):
        for name in filenames:
            found.add(os.path.relpath(os.path.join(dirpath, name), ROOT).replace(os.sep, "/"))
    return found


def main():
    check = "--check" in sys.argv[1:]
    outputs = build()
    stale = existing_outputs() - set(outputs)
    changed = []
    for rel, content in sorted(outputs.items()):
        full = os.path.join(ROOT, rel)
        try:
            with open(full, encoding="utf-8") as f:
                same = f.read() == content
        except FileNotFoundError:
            same = False
        if not same:
            changed.append(rel)
            if not check:
                os.makedirs(os.path.dirname(full), exist_ok=True)
                with open(full, "w", encoding="utf-8") as f:
                    f.write(content)
    if not check:
        for rel in sorted(stale):
            os.remove(os.path.join(ROOT, rel))
    pages = sum(1 for r in outputs if r.endswith(".html"))
    if check:
        if changed or stale:
            for rel in changed[:20]:
                print(f"  out of date: {rel}")
            for rel in sorted(stale)[:20]:
                print(f"  no longer generated: {rel}")
            print("Static guide pages are stale. Run: python3 tools/build_site.py")
            return 1
        print(f"Static site is up to date ({pages} pages).")
        return 0
    print(f"Wrote {len(changed)} file(s), removed {len(stale)}; {pages} pages total.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
