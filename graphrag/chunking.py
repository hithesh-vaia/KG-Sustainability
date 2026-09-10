"""Document loading, markdown-table linearization, and token-based chunking."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path

from langchain_text_splitters import TokenTextSplitter

from .config import CONFIG

TEXT_EXT = {".txt", ".md", ".markdown"}
# everything else is handed to Microsoft MarkItDown, which converts it to markdown
MARKITDOWN_EXT = {
    ".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls", ".csv",
    ".html", ".htm", ".xml", ".json", ".epub",
}
SUPPORTED = TEXT_EXT | MARKITDOWN_EXT


@lru_cache(maxsize=1)
def _markitdown():
    from markitdown import MarkItDown

    return MarkItDown(enable_plugins=False)


def _read_markitdown(path: Path) -> str:
    """Convert any office / pdf / html file to markdown via Microsoft MarkItDown.

    For PDFs, MarkItDown interleaves "<!-- Page number: N -->" comments; the
    extraction prompt reads those for provenance.page_number and
    pipeline._strip_markers drops them before embedding.
    """
    return _markitdown().convert(str(path)).text_content


def _read_file(path: Path) -> str:
    if path.suffix.lower() in MARKITDOWN_EXT:
        return linearize_tables(_read_markitdown(path))
    return path.read_text(encoding="utf-8", errors="ignore")


# --------------------------------------------------------------------------
# markdown table linearization
#
# PDF->markdown tables lose their meaning in two ways that wreck extraction:
#   1. empty spacer columns push values out of line with their header, e.g.
#        | Parameter          | FY 2024-25 |     | FY 2023-24 |     |
#        | Plastic Waste (A)  |            | 4.30|            | 4.99|
#      so "4.30" sits under the wrong header index.
#   2. a caption line between the header and the data splits one logical table
#      into two markdown tables, and token chunking can then separate them
#      entirely -- leaving data rows with no year/column labels at all.
#
# We rebuild each table with empty columns dropped and the header re-attached,
# then append one self-describing line per cell so a value can never be read
# against the wrong row or column, even if the table is later split by chunking.
# --------------------------------------------------------------------------

_ROW = re.compile(r"^\s*\|.*\|\s*$")
_SEP = re.compile(r"^\s*\|[\s:|-]+\|\s*$")
_HAS_DIGIT = re.compile(r"\d")


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _drop_empty_columns(rows: list[list[str]]) -> list[list[str]]:
    """Remove columns that are blank in every row (markitdown's spacer columns)."""
    if not rows:
        return rows
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    keep = [i for i in range(width) if any(r[i] for r in rows)]
    return [[r[i] for i in keep] for r in rows]


def _squeeze(row: list[str]) -> list[str]:
    return [c for c in row if c]


def _align(header: list[str], data: list[list[str]]) -> tuple[list[str], list[list[str]]]:
    """Line values up with their column labels.

    markitdown often emits the header labels and the data values in *different*
    columns (header at indices 1,3; values at 2,4), so no column is blank in every
    row and column-wise dropping cannot help. When squeezing the blanks out of each
    row independently makes the widths agree for most rows, that is the true
    alignment; otherwise fall back to dropping all-blank columns.
    """
    hdr = _squeeze(header)
    squeezed = [_squeeze(r) for r in data]
    agree = sum(1 for r in squeezed if len(r) == len(hdr))
    if hdr and squeezed and agree >= max(1, len(squeezed) * 2 // 3):
        fixed = [r + [""] * (len(hdr) - len(r)) if len(r) < len(hdr) else r[: len(hdr)]
                 for r in squeezed]
        return hdr, fixed
    rows = _drop_empty_columns([header] + data)
    return rows[0], rows[1:]


def _is_number(cell: str) -> bool:
    """True for a bare measurement value ('8.54', '1,234', '0', '12%') but not for
    a column label that merely contains digits ('FY 2025-26@', 'Unit')."""
    c = cell.strip().replace(",", "").rstrip("%*^@").strip()
    if not c:
        return False
    try:
        float(c)
        return True
    except ValueError:
        return False


def _looks_like_data(row: list[str]) -> bool:
    """A row carrying at least one bare number is data, not a header.

    Judged on the squeezed row: markitdown's empty spacer cells would otherwise
    dominate the denominator and make real data rows look like headers.
    """
    tail = _squeeze(row)[1:]
    return any(_is_number(c) for c in tail)


def _facts(caption: str, header: list[str], data: list[list[str]]) -> list[str]:
    """One self-describing line per cell: '<caption> | <row> | <column> = <value>'."""
    _BLANK = {"-", "--", "NA", "N/A", ""}
    out: list[str] = []
    for row in data:
        label = row[0] or "(unlabelled row)"
        for i, value in enumerate(row[1:], start=1):
            if value in _BLANK:
                continue
            col = header[i] if i < len(header) else ""
            if col == value:
                continue           # self-referential ("MT = MT", "58.70 = 58.70")
            if col in _BLANK:
                col = f"column {i}"
            prefix = f"{caption} | " if caption else ""
            out.append(f"- {prefix}{label} | {col} = {value}")
    return out


def linearize_tables(text: str) -> str:
    """Rewrite every markdown table as a compacted table plus per-cell fact lines."""
    lines = text.splitlines()
    out: list[str] = []
    header: list[str] = []       # last header seen, carried across a caption break

    caption = ""
    i = 0
    while i < len(lines):
        if not _ROW.match(lines[i]):
            stripped = lines[i].strip()
            # remember a short non-table line as the caption for the next table
            if stripped and not stripped.startswith("<!--") and len(stripped) < 120:
                caption = stripped
            out.append(lines[i])
            i += 1
            continue

        block = []
        while i < len(lines) and (_ROW.match(lines[i]) or _SEP.match(lines[i])):
            if not _SEP.match(lines[i]):
                block.append(_cells(lines[i]))
            i += 1
        if not block:
            continue

        # A block whose first row is already data belongs to the header we saw
        # before the caption break; otherwise the first row IS the header.
        # Compare squeezed widths -- the raw widths are padded with spacer cells.
        carry = (
            header
            and _looks_like_data(block[0])
            and len(_squeeze(header)) == len(_squeeze(block[0]))
        )
        if carry:
            hdr, data = _align(header, block)
        else:
            hdr, data = _align(block[0], block[1:])
            header = block[0]

        if not data:                       # header-only block: keep it for the next one
            continue

        out.append("| " + " | ".join(hdr) + " |")
        out.append("| " + " | ".join("---" for _ in hdr) + " |")
        for r in data:
            out.append("| " + " | ".join(r) + " |")
        facts = _facts(caption, hdr, data)
        if facts:
            out.append("")
            out.extend(facts)
        out.append("")
    return "\n".join(out)


@dataclass
class Chunk:
    id: str
    doc_id: str
    chunk_index: int
    text: str

    def dict(self) -> dict:
        return asdict(self)


def _chunk_id(doc_id: str, index: int) -> str:
    return hashlib.sha256(f"{doc_id}:{index}".encode()).hexdigest()[:16]


def load_documents(input_dir: Path | None = None) -> dict[str, str]:
    """Return {doc_id: text} for every supported file in input_dir."""
    input_dir = Path(input_dir or CONFIG.input_dir)
    docs: dict[str, str] = {}
    for path in sorted(input_dir.rglob("*")):
        if path.suffix.lower() in SUPPORTED and path.is_file():
            try:
                text = _read_file(path)
            except Exception as exc:  # pragma: no cover - depends on optional libs
                from .log import get_logger

                get_logger().warning("skipped %s: %s", path.name, exc)
                continue
            if text.strip():
                docs[path.stem] = text
    return docs


def chunk_documents(docs: dict[str, str]) -> list[Chunk]:
    splitter = TokenTextSplitter(
        chunk_size=CONFIG.chunk_tokens, chunk_overlap=CONFIG.chunk_overlap
    )
    chunks: list[Chunk] = []
    for doc_id, text in docs.items():
        for i, piece in enumerate(splitter.split_text(text)):
            piece = piece.strip()
            if piece:
                chunks.append(Chunk(_chunk_id(doc_id, i), doc_id, i, piece))
    return chunks
