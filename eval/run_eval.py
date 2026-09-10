"""Run the gold Q&A set against the live retrieval pipeline and score it.

    python -m eval.run_eval                       # deterministic scoring
    python -m eval.run_eval --judge               # + LLM-as-judge
    python -m eval.run_eval --only environmental
    python -m eval.run_eval --compare eval/results/20260908-190000.json
    python -m eval.run_eval --min-pass 0.8        # exit 1 if below
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import yaml

from graphrag.config import CONFIG
from graphrag.graphdb import Neo4jClient
from graphrag.llm import LLM
from graphrag.log import setup_logging
from graphrag.retrieve.global_search import global_search
from graphrag.retrieve.local_search import local_search

from .judge import judge as judge_answer
from .normalize import contains_all, contains_any

EVAL_DIR = Path(__file__).parent
GOLD = EVAL_DIR / "gold.yaml"
RESULTS = EVAL_DIR / "results"


def load_gold(path: Path | None = None) -> list[dict]:
    """Load one gold file, or every eval/gold*.yaml when no path is given."""
    paths = [path] if path else sorted(EVAL_DIR.glob("gold*.yaml"))
    items: list[dict] = []
    seen: dict[str, Path] = {}
    for p in paths:
        for item in yaml.safe_load(p.read_text()) or []:
            if item["id"] in seen:
                raise ValueError(f"duplicate gold id {item['id']!r} in {p.name} "
                                 f"(already in {seen[item['id']].name})")
            seen[item["id"]] = p
            items.append(item)
    return items


def _run_one(item: dict, db: Neo4jClient, llm: LLM, method_override: str | None,
             chunk_search: bool = False, top_k: int = 15):
    method = method_override or item.get("method", "local")
    runs: dict[str, dict] = {}
    if method in ("local", "both"):
        r = local_search(item["question"], db, llm, top_k=top_k, chunk_search=chunk_search)
        runs["local"] = {"answer": r.answer, "context": r.context,
                         "entities": r.entities, "chunks": r.chunks}
    if method in ("global", "both"):
        r = global_search(item["question"], db, llm)
        runs["global"] = {"answer": r.answer, "context": "",   "themes": r.themes,}
    return method, runs


def score_item(item: dict, runs: dict, use_judge: bool, llm: LLM) -> dict:
    must = item.get("must_include", [])
    mustnot = item.get("must_not_include", [])

    variants = []
    for mode, run in runs.items():
        ok, missing = contains_all(run["answer"], must)
        bad = contains_any(run["answer"], mustnot)
        answer_pass = ok and not bad
        ctx_ok, _ = contains_all(run.get("context", ""), must) if run.get("context") else (None, [])
        variants.append({
            "mode": mode, "answer": run["answer"],
            "answer_pass": answer_pass, "missing": missing, "unexpected": bad,
            "context_pass": ctx_ok,
        })

    # pick the winning variant (a pass beats a fail; else first)
    best = next((v for v in variants if v["answer_pass"]), variants[0])
    result = {
        "id": item["id"], "category": item.get("category", "?"),
        "question": item["question"], "expected": item.get("expected", ""),
        "chosen_mode": best["mode"], "answer": best["answer"],
        "answer_pass": best["answer_pass"], "missing": best["missing"],
        "unexpected": best["unexpected"], "context_pass": best["context_pass"],
        "variants": [{k: v[k] for k in ("mode", "answer_pass", "context_pass")} for v in variants],
    }
    if use_judge:
        result["judge"] = judge_answer(item, best["answer"], llm)
    return result


def _table(results: list[dict], use_judge: bool) -> str:
    rows = ["| id | cat | ans | ctx | judge | note |", "|---|---|---|---|---|---|"]
    for r in results:
        ans = "✅" if r["answer_pass"] else "❌"
        ctx = {True: "✅", False: "❌", None: "–"}[r["context_pass"]]
        jv = r.get("judge", {}).get("verdict", "–") if use_judge else "–"
        note = ""
        if not r["answer_pass"] and r["missing"]:
            note = "missing: " + ", ".join(r["missing"])[:50]
        if r["unexpected"]:
            note = "unexpected: " + ", ".join(r["unexpected"])[:50]
        rows.append(f"| {r['id']} | {r['category']} | {ans} | {ctx} | {jv} | {note} |")
    return "\n".join(rows)


def _summary(results: list[dict], use_judge: bool, usage: str) -> dict:
    n = len(results)
    passed = sum(r["answer_pass"] for r in results)
    ctx_have = [r for r in results if r["context_pass"] is not None]
    ctx_pass = sum(r["context_pass"] for r in ctx_have)
    by_cat: dict[str, list] = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r["answer_pass"])
    s = {
        "n": n, "answer_pass": passed, "answer_rate": round(passed / n, 3) if n else 0,
        "context_pass": ctx_pass, "context_rate": round(ctx_pass / len(ctx_have), 3) if ctx_have else None,
        "by_category": {c: f"{sum(v)}/{len(v)}" for c, v in sorted(by_cat.items())},
        "llm_usage": usage,
    }
    if use_judge:
        jc = {"correct": 0, "partial": 0, "incorrect": 0, "error": 0}
        for r in results:
            jc[r.get("judge", {}).get("verdict", "error")] = jc.get(r.get("judge", {}).get("verdict", "error"), 0) + 1
        s["judge"] = jc
    return s


def compare(current: list[dict], prev_path: Path) -> None:
    prev = {r["id"]: r["answer_pass"] for r in json.loads(prev_path.read_text())["results"]}
    fixed, regressed = [], []
    for r in current:
        was = prev.get(r["id"])
        if was is None:
            continue
        if r["answer_pass"] and not was:
            fixed.append(r["id"])
        elif not r["answer_pass"] and was:
            regressed.append(r["id"])
    print(f"\nvs {prev_path.name}:  FIXED {fixed or '—'}   REGRESSED {regressed or '—'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--only", help="filter by category")
    ap.add_argument("--ids", help="comma-separated ids to run")
    ap.add_argument("--method-override", choices=["local", "global", "both"])
    ap.add_argument("--chunk-search", action="store_true",
                    help="local search also runs a direct query->chunk vector search "
                         "(plain-RAG safety net for prose/table facts no entity captured)")
    ap.add_argument("--top-k", type=int, default=15, help="seed entities for local search")
    ap.add_argument("--compare", type=Path)
    ap.add_argument("--min-pass", type=float, default=0.0)
    ap.add_argument("--out", type=Path, default=RESULTS)
    args = ap.parse_args()

    setup_logging("eval")
    gold = load_gold()
    if args.only:
        gold = [g for g in gold if g.get("category") == args.only]
    if args.ids:
        want = set(args.ids.split(","))
        gold = [g for g in gold if g["id"] in want]
    if not gold:
        print("no gold items match the filter")
        return 1

    db = Neo4jClient()
    llm = LLM()
    results: list[dict] = []
    try:
        db.driver.verify_connectivity()
        for i, item in enumerate(gold, 1):
            method, runs = _run_one(item, db, llm, args.method_override,
                                    chunk_search=args.chunk_search, top_k=args.top_k)
            results.append(score_item(item, runs, args.judge, llm))
            print(f"  [{i}/{len(gold)}] {item['id']:<26} "
                  f"{'PASS' if results[-1]['answer_pass'] else 'FAIL'} ({results[-1]['chosen_mode']})")
    finally:
        db.close()

    summary = _summary(results, args.judge, llm.usage.summary())
    table = _table(results, args.judge)
    print("\n" + table)
    print("\n" + json.dumps(summary, indent=2))

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    payload = {"stamp": stamp, "model": llm.model, "judge": args.judge,
               "chunk_search": args.chunk_search, "top_k": args.top_k,
               "embed_provider": CONFIG.embed_provider, "embed_dim": CONFIG.embed_dim,
               "summary": summary, "results": results}
    (args.out / f"{stamp}.json").write_text(json.dumps(payload, indent=2))
    (args.out / "latest.md").write_text(
        f"# eval {stamp}  (model {llm.model})\n\n"
        f"answer pass **{summary['answer_pass']}/{summary['n']}** "
        f"({summary['answer_rate']:.0%}) · context {summary.get('context_rate')}\n\n"
        + table + "\n\n```json\n" + json.dumps(summary, indent=2) + "\n```\n"
    )
    print(f"\nwrote {args.out / (stamp + '.json')}  and  {args.out / 'latest.md'}")

    if args.compare:
        compare(results, args.compare)

    if summary["answer_rate"] < args.min_pass:
        print(f"\nFAIL: answer rate {summary['answer_rate']:.0%} < --min-pass {args.min_pass:.0%}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
