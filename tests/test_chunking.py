from graphrag.chunking import chunk_documents, load_documents


def test_chunk_ids_deterministic_and_nonempty():
    docs = {"doc": "word " * 5000}
    a = chunk_documents(docs)
    b = chunk_documents(docs)
    assert len(a) > 1
    assert [c.id for c in a] == [c.id for c in b]
    assert all(c.text.strip() for c in a)
    assert all(c.doc_id == "doc" for c in a)


def test_load_documents(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    (tmp_path / "b.md").write_text("world")
    (tmp_path / "c.pdf").write_text("ignored")
    docs = load_documents(tmp_path)
    assert set(docs) == {"a", "b"}
