from graphrag.ingest.extract import parse_extraction

PAYLOAD = {
    "document": {"name": "BRSR FY 2024-25", "document_type": "BRSRReport"},
    "entities": [
        {"id": "e1", "name": "IDBI Bank", "entity_type": "Company",
         "community": "Organization", "properties": {},
         "provenance": {"source_text": "IDBI Bank ...", "confidence": 0.99}},
        {"id": "e2", "name": "Scope 2 emissions", "entity_type": "Scope2Emission",
         "community": "Environmental",
         "properties": {"value": 57763.14, "unit": "tCO2e", "reporting_period": "FY 2024-25"},
         "provenance": {"source_text": "Scope 2 emissions were 57,763.14 tCO2e", "confidence": 0.98}},
    ],
    "relationships": [
        {"id": "r1", "source_id": "e1", "relationship": "HAS_MEASUREMENT",
         "target_id": "e2", "properties": {},
         "provenance": {"source_text": "...", "confidence": 0.95}},
    ],
}


def test_parse_entities_types_and_period_naming():
    res = parse_extraction(PAYLOAD, "chunk1")
    names = {e.name for e in res.entities}
    assert "IDBI BANK" in names
    # period-scoped type gets the period appended to keep years distinct
    assert "SCOPE 2 EMISSIONS (FY 2024-25)" in names
    scope = next(e for e in res.entities if e.name.startswith("SCOPE 2"))
    assert scope.entity_type == "Scope2Emission"
    assert scope.community == "Environmental"
    assert scope.confidence == 0.98
    assert scope.properties["value"] == 57763.14


def test_parse_relationship_id_resolution():
    res = parse_extraction(PAYLOAD, "chunk1")
    assert len(res.relationships) == 1
    r = res.relationships[0]
    assert r.source == "IDBI BANK"
    assert r.target == "SCOPE 2 EMISSIONS (FY 2024-25)"
    assert r.rel_type == "HAS_MEASUREMENT"


def test_invalid_terms_snap_to_defaults():
    res = parse_extraction(
        {"entities": [{"id": "x", "name": "Thing", "entity_type": "Nonsense",
                       "community": "Nonsense", "properties": {}, "provenance": {}}],
         "relationships": [{"source_id": "x", "target_id": "x",
                            "relationship": "FLARGLE", "provenance": {}}]},
        "c",
    )
    assert res.entities[0].entity_type == "Metric"
    assert res.entities[0].community == "Measurement"
    assert res.relationships == []  # self-loop dropped


def test_garbage_returns_empty():
    assert parse_extraction({}, "c").entities == []
    assert parse_extraction("nope", "c").relationships == []
