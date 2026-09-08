"""Document loading + token-based chunking."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path

from langchain_text_splitters import TokenTextSplitter

from .config import CONFIG

TEXT_EXT = {".txt", ".md", ".markdown"}
PDF_EXT = {".pdf"}
EXCEL_EXT = {".xlsx", ".xls", ".csv"}
SUPPORTED = TEXT_EXT | PDF_EXT | EXCEL_EXT


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    out = []
    for i, page in enumerate(reader.pages, 1):
        out.append(f"\n===== PAGE {i} =====\n{page.extract_text() or ''}")
    return "\n".join(out)


def _read_tabular(path: Path) -> str:
    import pandas as pd

    if path.suffix.lower() == ".csv":
        sheets = {"Sheet1": pd.read_csv(path)}
    else:
        sheets = pd.read_excel(path, sheet_name=None)  # all sheets
    parts = []
    for name, df in sheets.items():
        parts.append(f"# Sheet: {name}\n" + df.to_markdown(index=False))
    return "\n\n".join(parts)


def _read_file(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in PDF_EXT:
        return _read_pdf(path)
    if ext in EXCEL_EXT:
        return _read_tabular(path)
    return path.read_text(encoding="utf-8", errors="ignore")


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
