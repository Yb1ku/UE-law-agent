"""Evaluation runner for the EU legal RAG pipeline.

Measures for each in-scope query:
  - retrieval_recall   : fraction of expected article numbers found in retrieved chunks
  - citation_accurate  : no phantom citations (uses existing verify_citations)
  - phantom_citations  : list of article numbers cited but absent from retrieved chunks
  - fact_score         : fraction of key numeric facts from expected_answer found in response
  - facts_missing      : list of expected facts absent from the response
  - judge_score        : LLM-as-judge correctness score (0=wrong, 1=partial, 2=correct)
  - judge_reason       : one-sentence explanation from the judge
  - latency_s          : wall-clock time for query_engine.query()

For out-of-scope queries, the response is recorded for manual review.

Usage (run from project root):
    python src/eval_runner.py
    python src/eval_runner.py --dataset data/eval_dataset.json --top-k 8
    python src/eval_runner.py --out data/eval_results_baseline.json
    python src/eval_runner.py --no-judge   # skip LLM-as-judge to save time
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# Make src/ importable when running from project root (mirrors how app.py runs)
sys.path.insert(0, str(Path(__file__).parent))

from llama_index.core import Settings  # noqa: E402

from rag_pipeline import create_legal_rag_tool, verify_citations  # noqa: E402

# ---------------------------------------------------------------------------
# Fact-based completeness
# ---------------------------------------------------------------------------

_FACT_PATTERNS = [
    # EUR amounts: "EUR 20,000,000"
    re.compile(r'EUR[\s\xa0][\d,]+(?:,\d{3})*', re.IGNORECASE),
    # Percentages: "4%"
    re.compile(r'\d+(?:\.\d+)?\s*%'),
    # Time periods: "72 hours", "30 calendar days", "14 working days", "5 years"
    re.compile(r'\d+(?:\s+calendar)?\s+(?:hours?|days?|months?|years?|working\s+days?)', re.IGNORECASE),
]


def extract_key_facts(expected_answer: str) -> list[str]:
    """Extract verifiable numeric facts from an expected answer string."""
    seen: dict[str, None] = {}
    for pattern in _FACT_PATTERNS:
        for match in pattern.finditer(expected_answer):
            key = re.sub(r'\s+', ' ', match.group()).strip()
            seen[key] = None
    return list(seen)


def _normalize(s: str) -> str:
    return re.sub(r'\s+', ' ', s).strip().lower()


def fact_completeness(response: str, expected_answer: str) -> dict:
    """Check which key numeric facts from expected_answer appear in response.

    Returns:
        fact_score   : float 0-1 (1.0 when no facts to check)
        facts_found  : list of matched fact strings
        facts_missing: list of unmatched fact strings
    """
    facts = extract_key_facts(expected_answer)
    if not facts:
        return {"fact_score": 1.0, "facts_found": [], "facts_missing": []}

    resp_norm = _normalize(response)
    found, missing = [], []
    for fact in facts:
        if _normalize(fact) in resp_norm:
            found.append(fact)
        else:
            # Fallback: match just the numeric token (handles comma/space formatting differences)
            num = re.search(r'[\d,]+', fact)
            if num and num.group().replace(',', '') in response.replace(',', '').replace(' ', ''):
                found.append(fact)
            else:
                missing.append(fact)

    return {
        "fact_score": round(len(found) / len(facts), 3),
        "facts_found": found,
        "facts_missing": missing,
    }


# ---------------------------------------------------------------------------
# LLM-as-judge
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """\
You are an expert evaluator for a legal question-answering system.
Compare the system response to the reference answer and assign a score.

SCORING CRITERIA:
  2 - Correct  : Response captures all key facts from the reference answer.
  1 - Partial  : Response captures some but not all key facts.
  0 - Incorrect: Response is wrong, irrelevant, or missing critical information.

Question        : {query}
Reference answer: {expected_answer}
System response : {response}

Respond with ONLY a valid JSON object on a single line, no markdown:
{{"score": <0|1|2>, "reason": "<one concise sentence>"}}"""


def llm_judge(query: str, expected_answer: str, response: str) -> dict:
    """Call Settings.llm to score the response against the expected answer.

    Returns:
        judge_score  : int 0/1/2
        judge_reason : str explanation
    """
    prompt = _JUDGE_PROMPT.format(
        query=query,
        expected_answer=expected_answer,
        response=response,
    )
    raw = Settings.llm.complete(prompt).text.strip()

    # Extract the JSON object even if the model adds extra text
    json_match = re.search(r'\{.*?"score"\s*:\s*([012]).*?\}', raw, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            return {
                "judge_score": int(parsed["score"]),
                "judge_reason": str(parsed.get("reason", "")).strip(),
            }
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

    # Fallback: grab the first digit 0/1/2 from the output
    digit = re.search(r'[012]', raw)
    return {
        "judge_score": int(digit.group()) if digit else -1,
        "judge_reason": raw[:200],
    }


# ---------------------------------------------------------------------------
# Retrieval recall
# ---------------------------------------------------------------------------


def retrieval_recall(
    source_nodes: list,
    expected_articles: list[str],
    expected_document: str | None = None,
) -> float:
    """Fraction of expected article numbers found in retrieved chunks.

    When *expected_document* is provided, only chunks from that document are
    considered — this prevents a GDPR Article 12 chunk from counting as a hit
    for a Data Governance Act Article 12 question.
    """
    if not expected_articles:
        return 1.0

    found: set[str] = set()
    for node_with_score in source_nodes:
        node = node_with_score.node if hasattr(node_with_score, "node") else node_with_score
        # Skip chunks from the wrong document when the caller specifies one.
        if expected_document:
            node_doc = node.metadata.get("document", "")
            if node_doc and node_doc != expected_document:
                continue
        for art in str(node.metadata.get("chunk_articles", "")).split(", "):
            if art.strip():
                found.add(art.strip())

    hits = sum(1 for art in expected_articles if art in found)
    return hits / len(expected_articles)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_eval(
    dataset_path: str,
    similarity_top_k: int = 5,
    run_judge: bool = True,
) -> list[dict]:
    """Run every dataset entry through the query engine and return raw results."""
    tool = create_legal_rag_tool(similarity_top_k=similarity_top_k)
    query_engine = tool.query_engine

    with open(dataset_path) as f:
        dataset = json.load(f)

    results: list[dict] = []

    for i, item in enumerate(dataset, 1):
        print(f"[{i:>2}/{len(dataset)}] {item['id']}", flush=True)

        t0 = time.time()
        response = query_engine.query(item["query"])
        latency = time.time() - t0

        response_text = str(response)
        source_nodes = response.source_nodes

        if item["out_of_scope"]:
            results.append({
                "id": item["id"],
                "out_of_scope": True,
                "query": item["query"],
                "response": response_text,
                "latency_s": round(latency, 2),
            })
        else:
            recall = retrieval_recall(
                source_nodes,
                item["expected_articles"],
                expected_document=item.get("expected_document"),
            )
            citation = verify_citations(response_text, source_nodes)
            fact = fact_completeness(response_text, item["expected_answer"])

            result = {
                "id": item["id"],
                "out_of_scope": False,
                "expected_document": item["expected_document"],
                "query": item["query"],
                "expected_answer": item["expected_answer"],
                "response": response_text,
                "retrieval_recall": round(recall, 3),
                "phantom_citations": citation["phantom_articles"],
                "citation_accurate": citation["is_accurate"],
                "fact_score": fact["fact_score"],
                "facts_found": fact["facts_found"],
                "facts_missing": fact["facts_missing"],
                "latency_s": round(latency, 2),
            }

            if run_judge:
                judge = llm_judge(item["query"], item["expected_answer"], response_text)
                result["judge_score"] = judge["judge_score"]
                result["judge_reason"] = judge["judge_reason"]

            results.append(result)

    return results


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------


def print_summary(results: list[dict]) -> None:
    in_scope = [r for r in results if not r["out_of_scope"]]
    out_scope = [r for r in results if r["out_of_scope"]]

    avg_recall = sum(r["retrieval_recall"] for r in in_scope) / len(in_scope) if in_scope else 0
    n_accurate = sum(1 for r in in_scope if r["citation_accurate"])
    citation_rate = n_accurate / len(in_scope) if in_scope else 0

    fact_scores = [r["fact_score"] for r in in_scope if r.get("fact_score") is not None]
    avg_fact = sum(fact_scores) / len(fact_scores) if fact_scores else None

    has_judge = any("judge_score" in r for r in in_scope)
    judge_scores = [r["judge_score"] for r in in_scope if r.get("judge_score", -1) >= 0]
    avg_judge = sum(judge_scores) / len(judge_scores) if judge_scores else None

    avg_latency = sum(r["latency_s"] for r in results) / len(results) if results else 0

    print("\n" + "=" * 56)
    print("EVAL SUMMARY")
    print("=" * 56)
    print(f"Total entries         : {len(results)}")
    print(f"In-scope              : {len(in_scope)}")
    print(f"Out-of-scope          : {len(out_scope)}")
    print(f"Avg retrieval recall  : {avg_recall:.1%}")
    print(f"Citation accuracy     : {citation_rate:.1%}  ({n_accurate}/{len(in_scope)} clean)")
    if avg_fact is not None:
        n_full = sum(1 for r in in_scope if r.get("fact_score", 0) == 1.0)
        print(f"Avg fact completeness : {avg_fact:.1%}  ({n_full}/{len(in_scope)} perfect)")
    if has_judge and avg_judge is not None:
        n_correct = sum(1 for s in judge_scores if s == 2)
        n_partial = sum(1 for s in judge_scores if s == 1)
        n_wrong   = sum(1 for s in judge_scores if s == 0)
        print(f"Avg judge score       : {avg_judge:.2f}/2  "
              f"({n_correct} correct / {n_partial} partial / {n_wrong} wrong)")
    print(f"Avg latency           : {avg_latency:.1f}s")

    # Per-document breakdown
    by_doc: dict[str, list[dict]] = {}
    for r in in_scope:
        by_doc.setdefault(r["expected_document"], []).append(r)

    print("\nPer-document breakdown:")
    header = f"  {'Document':<30} {'Recall':>7} {'Fact':>7}"
    if has_judge:
        header += f" {'Judge':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for doc, rows in sorted(by_doc.items()):
        doc_recall = sum(r["retrieval_recall"] for r in rows) / len(rows)
        doc_fact_vals = [r["fact_score"] for r in rows if r.get("fact_score") is not None]
        doc_fact = sum(doc_fact_vals) / len(doc_fact_vals) if doc_fact_vals else None
        line = f"  {doc:<30} {doc_recall:>6.1%}"
        line += f" {doc_fact:>6.1%}" if doc_fact is not None else f" {'N/A':>6}"
        if has_judge:
            doc_judge = [r["judge_score"] for r in rows if r.get("judge_score", -1) >= 0]
            avg_dj = sum(doc_judge) / len(doc_judge) if doc_judge else None
            line += f" {avg_dj:>6.2f}" if avg_dj is not None else f" {'N/A':>6}"
        line += f"  ({len(rows)} questions)"
        print(line)

    # Flag retrieval failures
    recall_failures = [r for r in in_scope if r["retrieval_recall"] < 1.0]
    if recall_failures:
        print(f"\nRetrieval failures ({len(recall_failures)}):")
        for r in recall_failures:
            print(f"  {r['id']}: recall={r['retrieval_recall']:.0%}")

    # Flag citation failures
    citation_failures = [r for r in in_scope if not r["citation_accurate"]]
    if citation_failures:
        print(f"\nPhantom citation failures ({len(citation_failures)}):")
        for r in citation_failures:
            print(f"  {r['id']}: cited {r['phantom_citations']}")
    else:
        print("\nNo phantom citation failures.")

    # Flag fact completeness failures
    fact_failures = [r for r in in_scope if r.get("fact_score", 1.0) < 1.0]
    if fact_failures:
        print(f"\nFact completeness failures ({len(fact_failures)}):")
        for r in fact_failures:
            print(f"  {r['id']}: missing {r['facts_missing']}")

    # Flag judge failures (score < 2)
    if has_judge:
        judge_failures = [r for r in in_scope if r.get("judge_score", 2) < 2]
        if judge_failures:
            print(f"\nJudge failures (score < 2) ({len(judge_failures)}):")
            for r in judge_failures:
                print(f"  [{r['judge_score']}] {r['id']}: {r.get('judge_reason', '')[:100]}")

    # Out-of-scope responses for manual review
    if out_scope:
        print(f"\nOut-of-scope responses (manual review):")
        for r in out_scope:
            snippet = r["response"][:120].replace("\n", " ")
            print(f"  [{r['id']}] {snippet}...")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the EU legal RAG pipeline")
    parser.add_argument("--dataset", default="data/eval_dataset.json",
                        help="Path to eval_dataset.json (default: data/eval_dataset.json)")
    parser.add_argument("--top-k", type=int, default=10,
                        help="similarity_top_k for retrieval (default: 10)")
    parser.add_argument("--out", default=None,
                        help="Output JSON path (default: data/eval_results_<timestamp>.json)")
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip the LLM-as-judge step to save time")
    args = parser.parse_args()

    results = run_eval(args.dataset, args.top_k, run_judge=not args.no_judge)
    print_summary(results)

    out_path = args.out or f"data/eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved → {out_path}")
