"""Opt-in end-to-end eval smoke test. Runs only with RUN_EVAL=1 and a live Neo4j.

    RUN_EVAL=1 .venv/bin/python -m pytest tests/test_eval_smoke.py -q
"""
import os

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_EVAL") != "1", reason="set RUN_EVAL=1 to run the live eval smoke test"
)

SMOKE_IDS = {"scope2-fy25", "total-employees-fy25", "assurance-provider-fy25"}


def test_smoke_subset_passes():
    from graphrag.graphdb import Neo4jClient
    from graphrag.llm import LLM
    from eval.run_eval import load_gold, _run_one, score_item

    db = Neo4jClient()
    try:
        db.driver.verify_connectivity()
    except Exception:
        pytest.skip("Neo4j not reachable")

    llm = LLM()
    gold = [g for g in load_gold() if g["id"] in SMOKE_IDS]
    assert gold, "smoke ids not found in gold.yaml"
    try:
        failures = []
        for item in gold:
            _method, runs = _run_one(item, db, llm, None)
            res = score_item(item, runs, use_judge=False, llm=llm)
            if not res["answer_pass"]:
                failures.append((item["id"], res["missing"], res["answer"][:200]))
        assert not failures, f"eval smoke failures: {failures}"
    finally:
        db.close()
