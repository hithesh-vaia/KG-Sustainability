from graphrag.chunking import chunk_documents, linearize_tables, load_documents

# markitdown's PDF output: spacer columns offset the values from their headers,
# and a caption line severs the header from the data rows.
OFFSET_TABLE = """Total Waste Generated (in Metric Tonnes)
| Parameter | FY 2024-25 |     | FY 2023-24 |     |
| --------- | ---------- | --- | ---------- | --- |
Waste by category
| Plastic Waste (A)                     |     | 4.30  |     | 4.99 |
| ------------------------------------- | --- | ----- | --- | ---- |
| Construction and Demolition Waste (D) |     | 69.00 |     | 2.61 |
"""


def test_linearize_realigns_offset_columns_and_carries_header():
    out = linearize_tables(OFFSET_TABLE)
    # spacer columns gone, header re-attached to the data block after the caption
    assert "| Parameter | FY 2024-25 | FY 2023-24 |" in out
    assert "| Construction and Demolition Waste (D) | 69.00 | 2.61 |" in out
    # each cell restated with its row and column label, so chunking cannot orphan it
    assert "Construction and Demolition Waste (D) | FY 2024-25 = 69.00" in out
    assert "Construction and Demolition Waste (D) | FY 2023-24 = 2.61" in out
    assert "Plastic Waste (A) | FY 2024-25 = 4.30" in out


def test_linearize_leaves_plain_prose_untouched():
    prose = "IDBI Bank reported Scope 2 emissions of 57,763.14 tCO2e.\n\nSecond line."
    assert linearize_tables(prose) == prose


def test_linearize_handles_normal_table():
    out = linearize_tables("| name | val |\n| --- | --- |\n| foo | 1 |\n")
    assert "| name | val |" in out
    assert "foo | val = 1" in out


def test_chunk_ids_deterministic_and_nonempty():
    docs = {"doc": "word " * 5000}
    a = chunk_documents(docs)
    b = chunk_documents(docs)
    assert len(a) > 1
    assert [c.id for c in a] == [c.id for c in b]
    assert all(c.text.strip() for c in a)
    assert all(c.doc_id == "doc" for c in a)


def test_load_documents_text_and_markitdown(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    (tmp_path / "b.md").write_text("world")
    (tmp_path / "data.csv").write_text("name,val\nfoo,1\nbar,2\n")
    # unsupported extensions are ignored by the extension filter
    (tmp_path / "skip.rtf").write_text("nope")
    (tmp_path / "skip.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    docs = load_documents(tmp_path)
    assert set(docs) == {"a", "b", "data"}
    assert docs["a"] == "hello"
    # .csv is routed through MarkItDown -> markdown table
    assert "| name | val |" in docs["data"]
