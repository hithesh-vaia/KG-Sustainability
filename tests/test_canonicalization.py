from graphrag.ingest.extract import canon_period, split_period_from_name, parse_extraction
from graphrag.ingest.merge import build_alias_map


def test_canon_period_spellings_collapse():
    for raw in ["FY 2024-25", "FY2024-25", "fy 2024-2025", "2024-25", "(FY 2024-25)", "FY 2024-25*"]:
        assert canon_period(raw) == "FY 2024-25", raw


def test_split_period_from_name():
    assert split_period_from_name("TOTAL SCOPE 1 EMISSIONS FY 2024-25") == \
        ("TOTAL SCOPE 1 EMISSIONS", "FY 2024-25")
    assert split_period_from_name("TOTAL SCOPE 1 EMISSIONS (FY 2024-25)") == \
        ("TOTAL SCOPE 1 EMISSIONS", "FY 2024-25")
    assert split_period_from_name("WHISTLEBLOWER POLICY") == ("WHISTLEBLOWER POLICY", "")


def _ent(name, period=None):
    return {"id": name, "name": name, "entity_type": "Measurement", "community": "Measurement",
            "properties": {"value": 1, **({"reporting_period": period} if period else {})},
            "provenance": {"confidence": 0.9}}


def test_period_in_name_or_props_yields_one_node():
    # the model writes the year into the name in one chunk and into properties in
    # another; both must land on the same canonical entity name
    a = parse_extraction({"entities": [_ent("Total Scope 1 Emissions FY 2024-25")]}, "c1")
    b = parse_extraction({"entities": [_ent("Total Scope 1 Emissions", "FY 2024-25")]}, "c2")
    c = parse_extraction({"entities": [_ent("Total Scope 1 Emissions (fy2024-2025)")]}, "c3")
    names = {a.entities[0].name, b.entities[0].name, c.entities[0].name}
    assert names == {"TOTAL SCOPE 1 EMISSIONS (FY 2024-25)"}, names


def test_hyphen_variants_collapse():
    a = parse_extraction({"entities": [_ent("Permanent Employees - Female", "FY 2023-24")]}, "c1")
    b = parse_extraction({"entities": [_ent("Permanent Employees Female", "FY 2023-24")]}, "c2")
    assert a.entities[0].name == b.entities[0].name


def test_domain_is_definitional_not_the_models_guess():
    payload = {"entities": [{"id": "x", "name": "Scope 2", "entity_type": "Scope2Emission",
                             "community": "Measurement", "properties": {}, "provenance": {}}]}
    assert parse_extraction(payload, "c").entities[0].community == "Environmental"


def test_alias_map_collapses_legal_suffixes():
    m = build_alias_map(["IDBI BANK", "IDBI BANK", "IDBI BANK LTD.", "IDBI BANK LIMITED"])
    assert m == {"IDBI BANK LTD.": "IDBI BANK", "IDBI BANK LIMITED": "IDBI BANK"}
    # unrelated organisations are untouched
    assert build_alias_map(["IDBI BANK", "IDBI INTECH LTD."]) == {}


def test_refine_type_only_upgrades_generic():
    from graphrag import ontology as O
    assert O.refine_type("TOTAL SCOPE 1 EMISSIONS", "Measurement") == "Scope1Emission"
    assert O.refine_type("TOTAL EMPLOYEES", "Measurement") == "Employee"
    # never overwrites a considered classification
    assert O.refine_type("TOTAL SCOPE 1 EMISSIONS", "Policy") == "Policy"


def test_generic_types_keep_the_models_domain_guess():
    from graphrag import ontology as O
    # employee headcount typed generically -> trust the model's "Social"
    assert O.domain_for("Measurement", "Social") == "Social"
    # a specific type is definitional
    assert O.domain_for("Scope2Emission", "Measurement") == "Environmental"


def test_every_refinable_type_stays_period_scoped():
    from graphrag import ontology as O
    produced = {t for _, t in O._NAME_TYPE_RULES}
    assert produced <= O.PERIOD_SCOPED_TYPES, produced - O.PERIOD_SCOPED_TYPES


def _E(name, etype="Measurement"):
    from graphrag.ingest.merge import Entity
    return Entity(name=name, entity_type=etype, community="Measurement", description="")


def test_period_siblings_chain_consecutive_years_only():
    from graphrag.ingest.merge import link_period_siblings
    ents = [_E("TOTAL SCOPE 1 EMISSIONS (FY 2022-23)"),
            _E("TOTAL SCOPE 1 EMISSIONS (FY 2023-24)"),
            _E("TOTAL SCOPE 1 EMISSIONS (FY 2024-25)")]
    rels = link_period_siblings(ents)
    assert len(rels) == 2                      # N years -> N-1 edges, not N^2
    assert all(r.properties["derived"] is True for r in rels)
    assert {(r.source, r.target) for r in rels} == {
        ("TOTAL SCOPE 1 EMISSIONS (FY 2023-24)", "TOTAL SCOPE 1 EMISSIONS (FY 2022-23)"),
        ("TOTAL SCOPE 1 EMISSIONS (FY 2024-25)", "TOTAL SCOPE 1 EMISSIONS (FY 2023-24)")}


def test_period_siblings_do_not_link_different_metrics():
    from graphrag.ingest.merge import link_period_siblings
    assert link_period_siblings([_E("SCOPE 1 (FY 2023-24)"), _E("SCOPE 2 (FY 2024-25)")]) == []
    # nor entities without a period
    assert link_period_siblings([_E("WHISTLEBLOWER POLICY"), _E("ESG POLICY")]) == []


def test_link_orphan_figures_connects_unowned_measurements():
    from graphrag.ingest.merge import Entity, Relationship, link_orphan_figures
    ents = [
        _E("TOTAL SCOPE 2 EMISSIONS (FY 2024-25)", "Scope2Emission"),
        _E("PERMANENT EMPLOYEES (FY 2024-25)", "Employee"),
        _E("WHISTLEBLOWER POLICY", "Policy"),          # not a figure -> no edge
        _E("SCOPE 1 EMISSIONS (FY 2023-24)", "Scope1Emission"),
    ]
    ents.append(Entity(name="IDBI BANK", entity_type="Company", community="Organization", description=""))
    rels = [Relationship("IDBI BANK", "SCOPE 1 EMISSIONS (FY 2023-24)", "EMITS", "")]  # already owned
    edges, org = link_orphan_figures(ents, rels, "IDBI Bank")
    got = {(e.source, e.rel_type, e.target) for e in edges}
    assert ("IDBI BANK", "EMITS", "TOTAL SCOPE 2 EMISSIONS (FY 2024-25)") in got
    assert ("IDBI BANK", "EMPLOYS", "PERMANENT EMPLOYEES (FY 2024-25)") in got
    assert not any("WHISTLEBLOWER" in t for _, _, t in got)          # policy untouched
    assert not any("SCOPE 1" in t for _, _, t in got)                # already owned
    assert all(e.properties["derived"] is True for e in edges)
    assert org is None                                                # org entity already present
