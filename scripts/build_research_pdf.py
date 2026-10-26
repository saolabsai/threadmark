"""Build the technical note: pip install -e '.[paper]'; python scripts/build_research_pdf.py.

Reads the checked-in Markdown; no model calls, API keys, or network are used.
Equation PNGs are transient build assets under ignored runs/paper-preview/.
"""

from __future__ import annotations

import argparse
import hashlib
import html
from pathlib import Path
import re
from urllib.parse import urljoin

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager, mathtext  # noqa: E402
from PIL import Image as PILImage  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    HRFlowable,
    Image,
    KeepTogether,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
PAPER_URL = "https://github.com/saolabsai/threadmark/blob/main/research/threadmark.md"
INK = colors.HexColor("#172C37")
ACCENT = colors.HexColor("#14645B")
MUTED = colors.HexColor("#536674")
PAGE_WIDTH, PAGE_HEIGHT = A4
TEXT_WIDTH = PAGE_WIDTH - 104


def normalize(text: str) -> str:
    return text.translate(
        str.maketrans({"\u2014": " - ", "\u2013": "-", "\u2011": "-"})
    )


class Paper:
    def __init__(self, assets: Path) -> None:
        self.assets = assets
        assets.mkdir(parents=True, exist_ok=True)
        for family, filename in (
            ("PaperSerif", "DejaVuSerif.ttf"),
            ("PaperSerif-Bold", "DejaVuSerif-Bold.ttf"),
            ("PaperSerif-Italic", "DejaVuSerif-Italic.ttf"),
            ("PaperSans", "DejaVuSans.ttf"),
            ("PaperSans-Bold", "DejaVuSans-Bold.ttf"),
            ("PaperMono", "DejaVuSansMono.ttf"),
        ):
            font_path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / filename
            pdfmetrics.registerFont(TTFont(family, str(font_path)))
        pdfmetrics.registerFontFamily(
            "PaperSerif",
            normal="PaperSerif",
            bold="PaperSerif-Bold",
            italic="PaperSerif-Italic",
            boldItalic="PaperSerif-Bold",
        )
        pdfmetrics.registerFontFamily(
            "PaperSans", normal="PaperSans", bold="PaperSans-Bold"
        )
        styles = getSampleStyleSheet()
        self.body = ParagraphStyle(
            "PaperBody",
            fontName="PaperSerif",
            fontSize=10.4,
            leading=15.3,
            textColor=INK,
            spaceAfter=8,
            allowWidows=0,
            allowOrphans=0,
        )
        self.title = ParagraphStyle(
            "PaperTitle",
            parent=self.body,
            fontName="PaperSans-Bold",
            fontSize=24,
            leading=30,
            textColor=INK,
            spaceAfter=15,
        )
        self.subtitle = ParagraphStyle(
            "PaperSubtitle",
            parent=self.body,
            fontName="PaperSans",
            fontSize=9,
            leading=14,
            textColor=ACCENT,
            spaceAfter=14,
        )
        self.heading = ParagraphStyle(
            "PaperHeading",
            parent=self.body,
            fontName="PaperSans-Bold",
            fontSize=13.2,
            leading=18,
            textColor=ACCENT,
            spaceBefore=15,
            spaceAfter=9,
            keepWithNext=True,
        )
        self.cell = ParagraphStyle(
            "PaperCell",
            parent=self.body,
            fontSize=9.1,
            leading=13.1,
            spaceAfter=0,
        )
        self.code = ParagraphStyle(
            "PaperCode",
            parent=styles["Code"],
            fontName="PaperMono",
            fontSize=7.2,
            leading=11,
            textColor=INK,
            backColor=colors.HexColor("#F1F5F5"),
            borderPadding=9,
            spaceBefore=6,
            spaceAfter=12,
        )

    def equation(self, source: str, font_size: float) -> tuple[Path, float, float]:
        source = re.sub(r"\s+", " ", source.strip())
        source = re.sub(r"\\ge\b", r"\\geq", source)
        source = re.sub(r"\\le\b", r"\\leq", source)
        digest = hashlib.sha256(f"{font_size}:{source}".encode()).hexdigest()[:24]
        output = self.assets / f"math-{digest}.png"
        dpi = 240
        if not output.exists():
            mathtext.math_to_image(
                f"${source}$",
                str(output),
                dpi=dpi,
                color="#172C37",
                prop=font_manager.FontProperties(size=font_size),
            )
        with PILImage.open(output) as bitmap:
            width, height = bitmap.size
        return output, width * 72 / dpi, height * 72 / dpi

    def inline(self, text: str, font_size: float = 10.4) -> str:
        text = normalize(text)
        fragments: list[str] = []

        def save(markup: str) -> str:
            fragments.append(markup)
            return f"PAPERTOKEN{len(fragments) - 1}ENDTOKEN"

        def math(match: re.Match[str]) -> str:
            path, width, height = self.equation(match.group(1), font_size)
            return save(
                f'<img src="{path.as_posix()}" width="{width:.2f}" '
                f'height="{height:.2f}" valign="middle"/>'
            )

        # Currency amounts have no closing delimiter; do not mistake them for TeX.
        text = re.sub(r"\$(?![0-9])([^$\n]+)\$", math, text)
        text = re.sub(
            r"`([^`]+)`",
            lambda m: save(
                f'<font face="PaperMono" size="8.5">{html.escape(m.group(1))}</font>'
            ),
            text,
        )

        def link(match: re.Match[str]) -> str:
            target = urljoin(PAPER_URL, match.group(2))
            return save(
                f'<link href="{html.escape(target, quote=True)}" color="#14645B">'
                f"{html.escape(match.group(1))}</link>"
            )

        text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", link, text)
        text = html.escape(text)
        text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
        text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", text)
        for index, fragment in enumerate(fragments):
            text = text.replace(f"PAPERTOKEN{index}ENDTOKEN", fragment)
        return text

    def table(self, lines: list[str]) -> Table:
        rows = [[item.strip() for item in row.strip("|").split("|")] for row in lines]
        rows = [
            row
            for row in rows
            if not all(re.fullmatch(r":?-+:?", item) for item in row)
        ]
        cells = [
            [Paragraph(self.inline(cell, 9.1), self.cell) for cell in row]
            for row in rows
        ]
        widths = [
            TEXT_WIDTH * x
            for x in ([0.36, 0.64] if len(rows[0]) == 2 else [0.48, 0.26, 0.26])
        ]
        table = Table(cells, colWidths=widths, hAlign="LEFT", repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E6F0ED")),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.6, ACCENT),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#F6F8F8")],
                    ),
                    ("LEFTPADDING", (0, 0), (-1, -1), 9),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                    ("TOPPADDING", (0, 0), (-1, -1), 7),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ]
            )
        )
        return table

    def story(self, source: str) -> list:
        blocks: list[str] = []
        current: list[str] = []
        fence: str | None = None
        for line in source.strip().splitlines():
            marker = (
                "```"
                if line.startswith("```")
                else "$$" if line.strip() == "$$" else None
            )
            if marker and fence is None:
                if current:
                    blocks.append("\n".join(current))
                current = [line]
                fence = marker
            elif marker == fence and fence is not None:
                current.append(line)
                blocks.append("\n".join(current))
                current, fence = [], None
            elif not line.strip() and fence is None:
                if current:
                    blocks.append("\n".join(current))
                    current = []
            else:
                current.append(line)
        if current:
            blocks.append("\n".join(current))
        if fence is not None:
            raise ValueError(f"Unclosed Markdown fence: {fence}")
        story = []
        for block in blocks:
            block = block.strip()
            if block.startswith("# "):
                story.append(Paragraph(self.inline(block[2:]), self.title))
                continue
            if block.startswith("## "):
                story.append(Paragraph(self.inline(block[3:]), self.heading))
                continue
            if block.startswith("$$"):
                path, width, height = self.equation(block.strip("$\r\n "), 12)
                scale = min(1.0, (TEXT_WIDTH - 12) / width)
                if scale < 0.72:
                    raise ValueError(f"Equation too wide to read: {block}")
                equation = Image(str(path), width=width * scale, height=height * scale)
                equation.hAlign = "CENTER"
                story.append(KeepTogether([Spacer(1, 4), equation, Spacer(1, 12)]))
                continue
            if block.startswith("```"):
                story.append(
                    Preformatted(
                        normalize("\n".join(block.splitlines()[1:-1])), self.code
                    )
                )
                continue
            if block.startswith("|"):
                story.extend([self.table(block.splitlines()), Spacer(1, 12)])
                continue
            if block.startswith("**SaoLabs"):
                story.append(
                    Paragraph(self.inline(block.replace("**", "")), self.subtitle)
                )
                story.extend(
                    [HRFlowable(width="100%", color=ACCENT, thickness=1), Spacer(1, 14)]
                )
                continue
            if re.match(r"\d+\. ", block):
                for line in block.splitlines():
                    story.append(Paragraph(self.inline(line), self.body))
                continue
            story.append(
                Paragraph(self.inline(" ".join(block.splitlines())), self.body)
            )
        return story


def decorate(canvas, document) -> None:
    canvas.saveState()
    canvas.setFillColor(MUTED)
    canvas.setFont("PaperSans", 8)
    canvas.drawString(52, PAGE_HEIGHT - 30, "SAOLABS AI  /  THREADMARK")
    canvas.drawRightString(PAGE_WIDTH - 52, PAGE_HEIGHT - 30, "TECHNICAL RESEARCH NOTE")
    canvas.setStrokeColor(colors.HexColor("#D5DEDF"))
    canvas.line(52, 42, PAGE_WIDTH - 52, 42)
    canvas.drawString(52, 28, "27 October 2026  |  Experimental engineering research")
    canvas.drawRightString(PAGE_WIDTH - 52, 28, str(document.page))
    canvas.restoreState()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "research" / "threadmark.pdf"
    )
    args = parser.parse_args()
    source = (ROOT / "research" / "threadmark.md").read_text(encoding="utf-8")
    paper = Paper(ROOT / "runs" / "paper-preview" / "equations")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(args.output),
        pagesize=A4,
        rightMargin=52,
        leftMargin=52,
        topMargin=54,
        bottomMargin=58,
        title="Threadmark: source-backed working memory for evolving workflows",
        author="SaoLabs AI",
        subject="Architecture, equations, benchmark evidence, and related research",
        pageCompression=1,
    )
    document.build(paper.story(source), onFirstPage=decorate, onLaterPages=decorate)
    print(f"Built {args.output.name} ({args.output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
