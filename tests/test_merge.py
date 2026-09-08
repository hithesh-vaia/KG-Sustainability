from graphrag.ingest.extract import ChunkExtraction, ExtractedEntity, ExtractedRelationship
from graphrag.ingest.merge import merge_extractions


def _e(name, etype="Company", comm="Organization", desc="", props=None, chunk="c1", conf=0.9):
    return ExtractedEntity(name=name, entity_type=etype, community=comm, description=desc,
                           properties=props or {}, provenance={"confidence": conf},
                           confidence=conf, source_chunk=chunk)


def _r(s, t, rt="HAS_MEASUREMENT", chunk="c1", conf=0.9):
    return ExtractedRelationship(source=s, target=t, rel_type=rt, description="",
                                 provenance={"confidence": conf}, confidence=conf, source_chunk=chunk)


def test_merge_dedupes_by_name_and_counts_degree():
    exts = [
        ChunkExtraction("c1", [_e("IDBI BANK", desc="a bank"),
                               _e("SCOPE 2 (FY25)", "Scope2Emission", "Environmental")],
                        [_r("IDBI BANK", "SCOPE 2 (FY25)")]),
        ChunkExtraction("c2", [_e("IDBI BANK", desc="an Indian bank", chunk="c2")], []),
    ]
    ents, rels = merge_extractions(exts, llm=None, summarize=False)
    by = {e.name: e for e in ents}
    assert set(by) == {"IDBI BANK", "SCOPE 2 (FY25)"}
    assert by["IDBI BANK"].source_chunks == ["c1", "c2"]
    assert by["IDBI BANK"].degree == 1
    assert len(rels) == 1 and rels[0].rel_type == "HAS_MEASUREMENT"


def test_type_settled_by_majority_vote():
    exts = [ChunkExtraction("c1", [_e("X", "Company"), _e("X", "Company"), _e("X", "Subsidiary")], [])]
    ents, _ = merge_extractions(exts, llm=None, summarize=False)
    assert ents[0].entity_type == "Company"


def test_relationship_only_entities_are_created():
    exts = [ChunkExtraction("c1", [], [_r("A", "B", "OVERSEES")])]
    ents, rels = merge_extractions(exts, llm=None, summarize=False)
    assert {e.name for e in ents} == {"A", "B"}
    assert len(rels) == 1
