#!/usr/bin/env python3
"""Renders the project's README into a single documentation PDF.

The README stays the single source of truth for the prose (quickstart,
architecture, gotchas, ...) - this script converts it as-is instead of
duplicating it, so the two can never drift apart. Two appendices are
generated fresh from the live repo state on every run: the docker-compose
service list and the top-level directory tree.

Usage: generate_project_documentation.py [--readme PATH] [--output PATH]
"""

import argparse
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path

import markdown
import matplotlib as mpl
import yaml
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    Image,
    ListFlowable,
    ListItem,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_README = ROOT / "README.md"
DEFAULT_OUTPUT = ROOT / "documentation" / "project_documentation.pdf"
DEFAULT_COMPOSE_FILE = ROOT / "docker-compose.yaml"
COVER_IMAGE = ROOT / "branding" / "Alligator_Background.png"

EXCLUDED_NAMES = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
    "mlruns",
    "minio_data",
    ".claude",
    "test-reports",
    "security-reports",
}
MAX_TREE_ENTRIES_PER_DIR = 25
# Style names that get a PDF outline (bookmark) entry, mapped to nesting level.
HEADING_LEVELS = {"H1": 0, "H2": 0, "H3": 1, "H4": 2}
VOID_TAGS = {"br", "hr", "img", "input", "meta", "link"}
INLINE_WRAP = {
    "strong": ("<b>", "</b>"),
    "b": ("<b>", "</b>"),
    "em": ("<i>", "</i>"),
    "i": ("<i>", "</i>"),
    "del": ("<strike>", "</strike>"),
    "s": ("<strike>", "</strike>"),
    "sup": ("<super>", "</super>"),
    "sub": ("<sub>", "</sub>"),
}


# --- Minimal HTML -> DOM -----------------------------------------------------


class Node:
    __slots__ = ("attrs", "children", "tag", "text")

    def __init__(
        self, tag: str | None = None, attrs: list | None = None, text: str | None = None
    ) -> None:
        self.tag = tag
        self.attrs = dict(attrs or [])
        self.children: list[Node] = []
        self.text = text


class DomBuilder(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = Node("root")
        self.stack = [self.root]

    def handle_starttag(self, tag: str, attrs: list) -> None:
        node = Node(tag, attrs)
        self.stack[-1].children.append(node)
        if tag not in VOID_TAGS:
            self.stack.append(node)

    def handle_startendtag(self, tag: str, attrs: list) -> None:
        self.stack[-1].children.append(Node(tag, attrs))

    def handle_endtag(self, tag: str) -> None:
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].tag == tag:
                del self.stack[i:]
                return

    def handle_data(self, data: str) -> None:
        self.stack[-1].children.append(Node(None, None, data))


def parse_html(html_text: str) -> Node:
    builder = DomBuilder()
    builder.feed(html_text)
    return builder.root


def markdown_to_dom(text: str) -> Node:
    # python-markdown leaves raw HTML blocks (like the README's <details>)
    # unprocessed unless md_in_html is told to recurse into them.
    text = text.replace("<details>", '<details markdown="1">')
    # DejaVu (registered below for the rest of the Unicode README uses) has no
    # glyph for the emoji-presentation check mark; the plain check mark reads
    # identically in a status column and is universally covered.
    text = text.replace("✅", "✓")
    html = markdown.markdown(
        text, extensions=["extra", "sane_lists", "md_in_html"], output_format="xhtml"
    )
    return parse_html(html)


# --- Inline rendering (-> reportlab mini-HTML) -------------------------------


def escape_text(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_inline(node: Node) -> str:
    parts = []
    for child in node.children:
        if child.tag is None:
            parts.append(escape_text(child.text or ""))
        elif child.tag == "br":
            parts.append("<br/>")
        elif child.tag == "code":
            inner = render_inline(child)
            parts.append(f'<font face="{MONO}" size="8.4" color="#9d174d">{inner}</font>')
        elif child.tag == "a":
            href = escape_text(child.attrs.get("href", ""))
            inner = render_inline(child) or href
            if href:
                parts.append(f'<font color="#2563eb"><a href="{href}">{inner}</a></font>')
            else:
                parts.append(inner)
        elif child.tag in INLINE_WRAP:
            open_tag, close_tag = INLINE_WRAP[child.tag]
            parts.append(open_tag + render_inline(child) + close_tag)
        else:
            parts.append(render_inline(child))
    return "".join(parts)


def collect_text(node: Node) -> str:
    parts = []
    for child in node.children:
        if child.tag is None:
            parts.append(child.text or "")
        else:
            parts.append(collect_text(child))
    return "".join(parts)


# --- Block rendering ----------------------------------------------------------


class RenderContext:
    def __init__(self, styles: dict, avail_width: float) -> None:
        self.styles = styles
        self.avail_width = avail_width


def cell_style(styles: dict, header: bool, align: str) -> ParagraphStyle:
    base = styles["TableHeadCell"] if header else styles["TableCell"]
    alignment = {"left": 0, "center": 1, "right": 2}.get(align, 0)
    return ParagraphStyle(name=f"{base.name}_{align}_{id(base)}", parent=base, alignment=alignment)


def cell_align(node: Node) -> str:
    style_attr = node.attrs.get("style", "")
    if "right" in style_attr:
        return "right"
    if "center" in style_attr:
        return "center"
    return "left"


def compute_col_widths(rows: list, ncols: int, avail_width: float) -> list[float]:
    lengths = [4] * ncols
    for row in rows:
        for i, cell in enumerate(row):
            text = cell.getPlainText() if hasattr(cell, "getPlainText") else ""
            lengths[i] = max(lengths[i], len(text))
    total = sum(lengths)
    min_w = avail_width * 0.09
    raw = [max(min_w, avail_width * length / total) for length in lengths]
    scale = avail_width / sum(raw)
    return [w * scale for w in raw]


def render_table(node: Node, ctx: RenderContext) -> Table | None:
    styles = ctx.styles
    thead = next((c for c in node.children if c.tag == "thead"), None)
    tbody = next((c for c in node.children if c.tag == "tbody"), None)
    header_trs = [tr for tr in (thead.children if thead else []) if tr.tag == "tr"]
    body_source = tbody.children if tbody else node.children
    body_trs = [tr for tr in body_source if tr.tag == "tr"]

    def row_cells(tr: Node, header: bool) -> list:
        cells = []
        for td in tr.children:
            if td.tag not in ("td", "th"):
                continue
            align = cell_align(td)
            style = cell_style(styles, header or td.tag == "th", align)
            cells.append(Paragraph(render_inline(td) or "&nbsp;", style))
        return cells

    data = [row_cells(tr, header=True) for tr in header_trs]
    n_header_rows = len(data)
    data += [row_cells(tr, header=False) for tr in body_trs]
    if not data:
        return None
    ncols = max(len(row) for row in data)
    if ncols == 0:
        return None
    for row in data:
        while len(row) < ncols:
            row.append(Paragraph("", styles["TableCell"]))

    col_widths = compute_col_widths(data, ncols, ctx.avail_width)
    table = Table(data, colWidths=col_widths, repeatRows=n_header_rows or 0)
    style_cmds = [
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9c9c9")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if n_header_rows:
        style_cmds += [
            ("BACKGROUND", (0, 0), (-1, n_header_rows - 1), colors.HexColor("#2c3e50")),
            ("TEXTCOLOR", (0, 0), (-1, n_header_rows - 1), colors.white),
        ]
    style_cmds.append(
        ("ROWBACKGROUNDS", (0, n_header_rows), (-1, -1), [colors.white, colors.HexColor("#f5f6f7")])
    )
    table.setStyle(TableStyle(style_cmds))
    return table


def render_pre(node: Node, ctx: RenderContext) -> Table:
    code_node = next((c for c in node.children if c.tag == "code"), node)
    raw = collect_text(code_node).rstrip("\n")
    pre = Preformatted(raw, ctx.styles["CodeBlock"])
    table = Table([[pre]], colWidths=[ctx.avail_width])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f4f4f4")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#d8d8d8")),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def render_blockquote(node: Node, ctx: RenderContext) -> Table:
    inner = []
    render_children(node, inner, ctx)
    table = Table([[inner]], colWidths=[ctx.avail_width])
    table.setStyle(
        TableStyle(
            [
                ("LINEBEFORE", (0, 0), (0, -1), 2.5, colors.HexColor("#9ca3af")),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    return table


def render_list(node: Node, ordered: bool, ctx: RenderContext) -> ListFlowable:
    items = []
    for li in node.children:
        if li.tag != "li":
            continue
        nested = [c for c in li.children if c.tag in ("ul", "ol")]
        text_children = [c for c in li.children if c.tag not in ("ul", "ol")]
        if len(text_children) == 1 and text_children[0].tag == "p":
            text_html = render_inline(text_children[0])
        else:
            wrapper = Node()
            wrapper.children = text_children
            text_html = render_inline(wrapper)
        flowables = [Paragraph(text_html or "&nbsp;", ctx.styles["Body"])]
        for nested_list in nested:
            flowables.append(render_list(nested_list, nested_list.tag == "ol", ctx))
        items.append(ListItem(flowables, leftIndent=0, spaceBefore=1, spaceAfter=1))
    return ListFlowable(
        items,
        bulletType="1" if ordered else "bullet",
        leftIndent=16,
        bulletFontSize=8.5,
        spaceBefore=3,
        spaceAfter=6,
    )


def render_children(
    node: Node, story: list, ctx: RenderContext, skip_first_h1: bool = False
) -> None:
    styles = ctx.styles
    skipped_h1 = False
    for child in node.children:
        tag = child.tag
        if tag is None:
            continue
        if tag == "h1":
            if skip_first_h1 and not skipped_h1:
                skipped_h1 = True
                continue
            story.append(Paragraph(render_inline(child), styles["H1"]))
        elif tag == "h2":
            story.append(PageBreak())
            story.append(Paragraph(render_inline(child), styles["H2"]))
        elif tag == "h3":
            story.append(Paragraph(render_inline(child), styles["H3"]))
        elif tag == "h4":
            story.append(Paragraph(render_inline(child), styles["H4"]))
        elif tag == "p":
            text = render_inline(child)
            if text.strip():
                story.append(Paragraph(text, styles["Body"]))
        elif tag == "ul":
            story.append(render_list(child, ordered=False, ctx=ctx))
        elif tag == "ol":
            story.append(render_list(child, ordered=True, ctx=ctx))
        elif tag == "table":
            table = render_table(child, ctx)
            if table is not None:
                story.append(table)
                story.append(Spacer(1, 0.35 * cm))
        elif tag in ("pre",):
            story.append(render_pre(child, ctx))
            story.append(Spacer(1, 0.25 * cm))
        elif tag == "blockquote":
            story.append(render_blockquote(child, ctx))
            story.append(Spacer(1, 0.2 * cm))
        elif tag == "hr":
            story.append(Spacer(1, 0.15 * cm))
            story.append(HRFlowable(width="100%", color=colors.HexColor("#d8d8d8"), thickness=0.7))
            story.append(Spacer(1, 0.15 * cm))
        elif tag == "details":
            summary = next((c for c in child.children if c.tag == "summary"), None)
            if summary is not None:
                story.append(Paragraph(render_inline(summary), styles["DetailsSummary"]))
            rest = Node()
            rest.children = [c for c in child.children if c.tag != "summary"]
            render_children(rest, story, ctx)
        elif tag in ("div", "section", "span"):
            render_children(child, story, ctx)
        # anything else (img, comments, ...) is silently skipped


# --- Styles -------------------------------------------------------------------

SANS = "DejaVuSans"
SANS_BOLD = "DejaVuSans-Bold"
SANS_OBLIQUE = "DejaVuSans-Oblique"
SANS_BOLD_OBLIQUE = "DejaVuSans-BoldOblique"
MONO = "DejaVuSansMono"


def register_fonts() -> None:
    # The README uses characters outside the base-14 fonts' WinAnsiEncoding
    # (arrows, box-drawing, >=, ...); DejaVu covers them and ships inside the
    # matplotlib wheel this project already depends on, so no extra download
    # or font file is needed.
    if SANS in pdfmetrics.getRegisteredFontNames():
        return
    font_dir = Path(mpl.get_data_path()) / "fonts" / "ttf"
    pdfmetrics.registerFont(TTFont(SANS, str(font_dir / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont(SANS_BOLD, str(font_dir / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFont(TTFont(SANS_OBLIQUE, str(font_dir / "DejaVuSans-Oblique.ttf")))
    pdfmetrics.registerFont(TTFont(SANS_BOLD_OBLIQUE, str(font_dir / "DejaVuSans-BoldOblique.ttf")))
    pdfmetrics.registerFont(TTFont(MONO, str(font_dir / "DejaVuSansMono.ttf")))
    pdfmetrics.registerFontFamily(
        SANS, normal=SANS, bold=SANS_BOLD, italic=SANS_OBLIQUE, boldItalic=SANS_BOLD_OBLIQUE
    )


def build_styles() -> dict:
    register_fonts()
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="CoverTitle",
            fontName=SANS_BOLD,
            fontSize=27,
            leading=32,
            textColor=colors.HexColor("#111827"),
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CoverSubtitle",
            fontName=SANS,
            fontSize=13,
            leading=17,
            textColor=colors.HexColor("#4b5563"),
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Meta",
            fontName=SANS,
            fontSize=9.5,
            leading=13,
            textColor=colors.HexColor("#6b7280"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="H1",
            fontName=SANS_BOLD,
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#111827"),
            spaceBefore=2,
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="H2",
            fontName=SANS_BOLD,
            fontSize=15,
            leading=19,
            textColor=colors.HexColor("#1d4ed8"),
            spaceBefore=4,
            spaceAfter=9,
        )
    )
    styles.add(
        ParagraphStyle(
            name="H3",
            fontName=SANS_BOLD,
            fontSize=12,
            leading=15,
            textColor=colors.HexColor("#374151"),
            spaceBefore=12,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="H4",
            fontName=SANS_BOLD,
            fontSize=10.5,
            leading=13,
            textColor=colors.HexColor("#4b5563"),
            spaceBefore=8,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Body",
            fontName=SANS,
            fontSize=9.6,
            leading=13.5,
            spaceAfter=6,
            alignment=TA_JUSTIFY,
        )
    )
    styles.add(ParagraphStyle(name="TableHeadCell", fontName=SANS_BOLD, fontSize=8, leading=10))
    styles.add(ParagraphStyle(name="TableCell", fontName=SANS, fontSize=8, leading=10.5))
    styles.add(
        ParagraphStyle(
            name="CodeBlock",
            fontName=MONO,
            fontSize=7.6,
            leading=9.8,
            textColor=colors.HexColor("#1f2937"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="DetailsSummary",
            fontName=SANS_BOLD,
            fontSize=10,
            leading=13,
            textColor=colors.HexColor("#1d4ed8"),
            spaceBefore=8,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TOCTitle",
            fontName=SANS_BOLD,
            fontSize=17,
            leading=21,
            textColor=colors.HexColor("#111827"),
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TOCLevel0",
            fontName=SANS_BOLD,
            fontSize=10.5,
            leading=16,
            leftIndent=0,
            spaceBefore=6,
            textColor=colors.HexColor("#111827"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="TOCLevel1",
            fontName=SANS,
            fontSize=9.5,
            leading=14,
            leftIndent=14,
            spaceBefore=2,
            textColor=colors.HexColor("#374151"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="TOCLevel2",
            fontName=SANS,
            fontSize=9,
            leading=13,
            leftIndent=28,
            spaceBefore=1,
            textColor=colors.HexColor("#6b7280"),
        )
    )
    return styles


# --- PDF scaffolding: footer + outline bookmarks -----------------------------


def make_footer(generated_at: str) -> object:
    def _footer(canv: object, doc: object) -> None:
        canv.saveState()
        canv.setFont(SANS, 7.5)
        canv.setFillColor(colors.HexColor("#9ca3af"))
        canv.drawString(doc.leftMargin, 1.0 * cm, f"Alligator Metaswamp — generated {generated_at}")
        canv.drawRightString(A4[0] - doc.rightMargin, 1.0 * cm, f"Page {doc.page}")
        canv.restoreState()

    return _footer


_SLUG_STRIP_RE = re.compile(r"[^\w\s-]", re.UNICODE)


def github_slug(text: str, seen: dict[str, int]) -> str:
    # Matches GitHub's heading-anchor algorithm closely enough to resolve the
    # README's own "[step 2b](#2b-promote-a-model-to-prod)"-style anchor links.
    base = re.sub(r"\s+", "-", _SLUG_STRIP_RE.sub("", text.strip().lower()))
    n = seen.get(base, 0)
    seen[base] = n + 1
    return base if n == 0 else f"{base}-{n}"


class OutlineDocTemplate(BaseDocTemplate):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._slug_seen: dict[str, int] = {}

    def build(self, flowables: list, **kwargs: object) -> None:
        # multiBuild() re-runs build() per pass to settle the TOC's page
        # numbers; slugs must be identical every pass or the TOC's links
        # (captured on an earlier pass) end up pointing at stale bookmark names.
        self._slug_seen = {}
        super().build(flowables, **kwargs)

    def afterFlowable(self, flowable: object) -> None:  # noqa: N802 - reportlab's hook name
        if not isinstance(flowable, Paragraph):
            return
        level = HEADING_LEVELS.get(flowable.style.name)
        if level is None:
            return
        text = flowable.getPlainText()
        key = github_slug(text, self._slug_seen)
        self.canv.bookmarkPage(key)
        self.canv.addOutlineEntry(text, key, level=level, closed=level > 0)
        self.notify("TOCEntry", (level, text, self.page, key))


# --- Live appendices -----------------------------------------------------------


def git_commit_hash() -> str:
    git = shutil.which("git")
    if git is None:
        return "unknown"
    try:
        out = subprocess.run(  # noqa: S603 - fixed args, resolved trusted binary
            [git, "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def build_compose_appendix(compose_file: Path, ctx: RenderContext) -> list:
    styles = ctx.styles
    story = [
        PageBreak(),
        Paragraph("Appendix A — Docker Compose services (live)", styles["H2"]),
        Paragraph(
            f"Parsed from {compose_file.relative_to(ROOT)} at generation time, so this "
            "table reflects the stack as it exists right now, not a snapshot of the README prose.",
            styles["Body"],
        ),
    ]
    if not compose_file.exists():
        story.append(Paragraph("Compose file not found.", styles["Body"]))
        return story

    with compose_file.open() as f:
        compose = yaml.safe_load(f)
    services = (compose or {}).get("services", {})

    rows = [["Service", "Image / build", "Ports", "Depends on"]]
    for name, raw_spec in sorted(services.items()):
        spec = raw_spec or {}
        if "image" in spec:
            image = spec["image"]
        elif "build" in spec:
            build = spec["build"]
            context = build.get("context", ".") if isinstance(build, dict) else build
            image = f"built from {context}"
        else:
            image = "—"
        ports = ", ".join(str(p) for p in spec.get("ports", [])) or "—"
        depends_on = spec.get("depends_on", [])
        if isinstance(depends_on, dict):
            depends_on = list(depends_on.keys())
        depends = ", ".join(depends_on) or "—"
        rows.append([name, image, ports, depends])

    def esc(row: list) -> list:
        return [escape_text(str(cell)) for cell in row]

    data = [[Paragraph(f"<b>{c}</b>", styles["TableHeadCell"]) for c in esc(rows[0])]] + [
        [Paragraph(c, styles["TableCell"]) for c in esc(row)] for row in rows[1:]
    ]

    col_widths = compute_col_widths(data, 4, ctx.avail_width)
    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9c9c9")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f6f7")]),
            ]
        )
    )
    story.append(table)
    return story


def is_excluded(path: Path) -> bool:
    return path.name in EXCLUDED_NAMES or (
        path.name.startswith(".") and path.name not in {".env.example"}
    )


def build_tree_lines(root: Path, max_depth: int = 2) -> list:
    lines = [root.name + "/"]

    def walk(dir_path: Path, prefix: str, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            entries = sorted(
                (p for p in dir_path.iterdir() if not is_excluded(p)),
                key=lambda p: (p.is_file(), p.name.lower()),
            )
        except PermissionError:
            return
        # ASCII rather than box-drawing chars: the PDF's Courier base font uses
        # WinAnsiEncoding and has no glyphs for U+2500-range tree connectors.
        shown, hidden = entries[:MAX_TREE_ENTRIES_PER_DIR], entries[MAX_TREE_ENTRIES_PER_DIR:]
        for i, entry in enumerate(shown):
            last = i == len(shown) - 1 and not hidden
            connector = "`-- " if last else "|-- "
            lines.append(f"{prefix}{connector}{entry.name}{'/' if entry.is_dir() else ''}")
            if entry.is_dir():
                extension = "    " if last else "|   "
                walk(entry, prefix + extension, depth + 1)
        if hidden:
            lines.append(f"{prefix}`-- ... ({len(hidden)} more)")

    walk(root, "", 1)
    return lines


def build_tree_appendix(ctx: RenderContext) -> list:
    styles = ctx.styles
    story = [
        PageBreak(),
        Paragraph("Appendix B — Repository layout (live)", styles["H2"]),
        Paragraph(
            "Walked from the repository root at generation time (build artefacts, caches "
            "and virtual environments excluded), two levels deep.",
            styles["Body"],
        ),
    ]
    tree_text = "\n".join(build_tree_lines(ROOT))
    # Preformatted natively splits across page frames; a Table wrapper (used
    # elsewhere for the boxed look) does not, and errors out on content this long.
    story.append(Preformatted(tree_text, styles["CodeBlock"]))
    return story


def fitted_image(path: Path, max_width: float, max_height: float) -> Image:
    # Scales to fit within the given box, preserving aspect ratio (no cropping) -
    # the source is a landscape illustration, a portrait page is not.
    natural_width, natural_height = ImageReader(str(path)).getSize()
    scale = min(max_width / natural_width, max_height / natural_height)
    image = Image(str(path), width=natural_width * scale, height=natural_height * scale)
    image.hAlign = "CENTER"
    return image


def build_cover(styles: dict, generated_at: str, commit: str, avail_width: float) -> list:
    return [
        Spacer(1, 1 * cm),
        fitted_image(COVER_IMAGE, avail_width, 9 * cm),
        Spacer(1, 1 * cm),
        Paragraph("Alligator Metaswamp", styles["CoverTitle"]),
        Paragraph("Project documentation", styles["CoverSubtitle"]),
        Spacer(1, 1 * cm),
        Paragraph(f"Generated: {generated_at}", styles["Meta"]),
        Paragraph(f"Git commit: {commit}", styles["Meta"]),
        Paragraph(
            "Rendered from README.md by scripts/generate_project_documentation.py",
            styles["Meta"],
        ),
        Spacer(1, 0.6 * cm),
        HRFlowable(width="100%", color=colors.HexColor("#d8d8d8"), thickness=0.8),
    ]


def build_closing_page(avail_width: float, avail_height: float) -> list:
    image = fitted_image(COVER_IMAGE, avail_width, avail_height)
    return [PageBreak(), Spacer(1, max(0, (avail_height - image.drawHeight) / 2)), image]


def build_toc(styles: dict) -> list:
    toc = TableOfContents()
    toc.levelStyles = [styles["TOCLevel0"], styles["TOCLevel1"], styles["TOCLevel2"]]
    toc.dotsMinLevel = 0
    return [PageBreak(), Paragraph("Table of contents", styles["TOCTitle"]), toc]


# --- Main ----------------------------------------------------------------------


def build_pdf(readme_path: Path, output_path: Path, compose_file: Path) -> None:
    readme_text = readme_path.read_text(encoding="utf-8")
    dom = markdown_to_dom(readme_text)

    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    commit = git_commit_hash()
    styles = build_styles()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = OutlineDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.6 * cm,
        title="Alligator Metaswamp — Project Documentation",
        author="scripts/generate_project_documentation.py",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
    template = PageTemplate(id="main", frames=[frame], onPage=make_footer(generated_at))
    doc.addPageTemplates([template])

    ctx = RenderContext(styles=styles, avail_width=doc.width)

    story = build_cover(styles, generated_at, commit, doc.width)
    story += build_toc(styles)
    story.append(PageBreak())
    render_children(dom, story, ctx, skip_first_h1=True)
    story += build_compose_appendix(compose_file, ctx)
    story += build_tree_appendix(ctx)
    story += build_closing_page(doc.width, doc.height)

    doc.multiBuild(story)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--readme", type=Path, default=DEFAULT_README, help="Source README (markdown)."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output PDF path.")
    parser.add_argument(
        "--compose-file",
        type=Path,
        default=DEFAULT_COMPOSE_FILE,
        help="docker-compose file used for the live services appendix.",
    )
    args = parser.parse_args()

    if not args.readme.exists():
        print(f"README not found: {args.readme}", file=sys.stderr)
        return 1

    build_pdf(args.readme, args.output, args.compose_file)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
