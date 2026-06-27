"""Built-in sample document + page renderer for offline demos and CPU-only tests."""

from __future__ import annotations

import glob
import textwrap
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont

SAMPLE_DOC = """Quarterly Business Report

Overview

The company reported steady growth in the third quarter of 2021. Revenue increased
by 12% compared to the previous year, driven by strong demand in the northern region.

Key Highlights

- Revenue reached $4.5 million, exceeding the forecast.
- Operating costs fell by nearly a third after the new policy.
- Customer satisfaction improved across all product lines.

Outlook

The board expects continued growth in the coming year and has approved a budget of
$6 million for expansion. A detailed plan will be presented on December 31st."""

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]


def _font(size: int):
    for p in _FONT_CANDIDATES:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    try:
        import matplotlib
        return ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"), size)
    except Exception:
        return ImageFont.load_default()


def render_page(text: str = SAMPLE_DOC, width: int = 1000, font_size: int = 26) -> Image.Image:
    font = _font(font_size)
    margin, line_h = 60, int(font_size * 1.5)
    lines: List[str] = []
    for para in text.split("\n"):
        para = para.rstrip()
        if not para:
            lines.append("")
            continue
        wrapped = textwrap.wrap(para, width=max(20, (width - 2 * margin) // (font_size // 2)))
        lines.extend(wrapped or [""])
    height = margin * 2 + line_h * max(1, len(lines))
    img = Image.new("RGB", (width, height), (250, 250, 250))
    d = ImageDraw.Draw(img)
    y = margin
    for ln in lines:
        d.text((margin, y), ln, fill=(15, 15, 15), font=font)
        y += line_h
    return img


def write_samples(out_dir: str | Path) -> List[Tuple[str, str]]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pairs: List[Tuple[str, str]] = []
    txt = out / "sample_doc.txt"
    txt.write_text(SAMPLE_DOC, encoding="utf-8")
    pairs.append((str(txt), "clean text"))
    png = out / "sample_page.png"
    render_page().save(png)
    pairs.append((str(png), "rendered page"))
    return pairs


__all__ = ["SAMPLE_DOC", "render_page", "write_samples"]
