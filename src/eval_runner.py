"""Evaluation runner for the EU legal RAG pipeline.

Measures for each in-scope query:
  - retrieval_recall        : fraction of expected article numbers found in retrieved chunks
  - cross_doc_recall        : (cross-document entries only) per-document article coverage
  - citation_accurate       : no phantom citations (uses existing verify_citations)
  - phantom_citations       : list of article numbers cited but absent from retrieved chunks
  - attribution_accurate    : no phantom document attributions (doc+article pairs)
  - phantom_attributions    : list of (document, article) pairs cited but absent from chunks
  - fact_score              : fraction of key numeric facts from expected_answer found in response
  - facts_missing           : list of expected facts absent from the response
  - judge_score             : LLM-as-judge correctness score (0=wrong, 1=partial, 2=correct)
  - judge_reason            : one-sentence explanation from the judge
  - latency_s               : wall-clock time for query_engine.query()

For out-of-scope queries:
  - oos_refused             : bool — True if the model correctly declined to answer
  - response                : recorded for manual review

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

from rag_pipeline import (  # noqa: E402
    create_legal_rag_tool,
    verify_citations,
    reset_query_profile,
    get_query_profile,
    _CHUNK_HEADER_RE,
)

# ---------------------------------------------------------------------------
# Fact-based completeness
# ---------------------------------------------------------------------------

_FACT_PATTERNS = [
    # EUR amounts: "EUR 20,000,000" (ASCII commas) or "EUR 20 000 000" / "EUR 20\xa0000\xa0000"
    # (space/NBSP thousands separator as used in EUR-Lex source texts)
    re.compile(r'EUR[\s\xa0]+\d{1,3}(?:[\s\xa0,]\d{3})*', re.IGNORECASE),
    # Percentages: "4%" or "4 %" or "4\xa0%"
    re.compile(r'\d+(?:\.\d+)?[\s\xa0]*%'),
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
            # Fallback: match just the numeric token (handles comma/space/NBSP formatting differences)
            num = re.search(r'[\d,]+', fact)
            if num and num.group().replace(',', '') in re.sub(r'[\s\xa0,]', '', response):
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
# OOS refusal detection
# ---------------------------------------------------------------------------

# Patterns that indicate the model correctly declined to answer an OOS query.
_OOS_REFUSAL_RE = re.compile(
    r"outside (the )?scope"
    r"|not covered"
    r"|not (contain|include|address|discuss|mention|found|available)\b"
    r"|cannot (find|answer|provide|determine)\b"
    r"|no (information|content|data) (found|available|provided|in the)"
    r"|beyond (the )?scope"
    r"|does not (cover|address|include|contain|discuss)\b"
    r"|unable to (find|answer|provide)\b"
    r"|not part of (the|this)\b"
    r"|not in the (provided|retrieved|available)\b",
    re.IGNORECASE,
)


def detect_oos_refusal(response_text: str) -> bool:
    """Return True if the response contains a clear refusal/out-of-scope signal."""
    return bool(_OOS_REFUSAL_RE.search(response_text))


# ---------------------------------------------------------------------------
# Cross-document retrieval recall
# ---------------------------------------------------------------------------


def cross_doc_retrieval_recall(
    source_nodes: list,
    expected_articles: list[str],
    expected_documents: list[str],
) -> dict:
    """For cross-document queries: check that expected_articles[i] is retrieved
    from expected_documents[i].

    Returns:
        cross_doc_recall   : float 0-1 overall (one point per (article, doc) pair)
        cross_doc_details  : list of {"article": ..., "document": ..., "found": bool}
    """
    pairs = list(zip(expected_articles, expected_documents))
    if not pairs:
        return {"cross_doc_recall": 1.0, "cross_doc_details": []}

    details = []
    hits = 0
    for art, doc in pairs:
        found = False
        for node_with_score in source_nodes:
            node = node_with_score.node if hasattr(node_with_score, "node") else node_with_score
            if node.metadata.get("document", "") != doc:
                continue
            node_arts = [a.strip() for a in str(node.metadata.get("chunk_articles", "")).split(",")]
            if art in node_arts:
                found = True
                break
        details.append({"article": art, "document": doc, "found": found})
        if found:
            hits += 1

    return {
        "cross_doc_recall": round(hits / len(pairs), 3),
        "cross_doc_details": details,
    }


# ---------------------------------------------------------------------------
# Source attribution accuracy
# ---------------------------------------------------------------------------

# Canonical display names as they appear in chunk metadata.
_DOC_DISPLAY_NAMES = [
    "GDPR",
    "AI Act",
    "Data Act",
    "Data Governance Act",
    "Cyber Resilience Act",
]

# Matches "GDPR Article 5", "AI Act Article 99", "Data Governance Act Article 14", etc.
_ATTR_DOC_FIRST_RE = re.compile(
    r"(GDPR|AI\s+Act|Data\s+Governance\s+Act|Data\s+Act|Cyber\s+Resilience\s+Act)"
    r"[\s,]*Article[\s\xa0]+(\d+)",
    re.IGNORECASE,
)
# Matches "Article 5 of the GDPR", "Article 99 of the AI Act", etc.
_ATTR_ART_FIRST_RE = re.compile(
    r"Article[\s\xa0]+(\d+)[^.]{0,40}of\s+(?:the\s+)?"
    r"(GDPR|AI\s+Act|Data\s+Governance\s+Act|Data\s+Act|Cyber\s+Resilience\s+Act)",
    re.IGNORECASE,
)

_DOC_NORMALISE = {
    "gdpr": "GDPR",
    "ai act": "AI Act",
    "data governance act": "Data Governance Act",
    "data act": "Data Act",
    "cyber resilience act": "Cyber Resilience Act",
}


def _normalise_doc(raw: str) -> str:
    return _DOC_NORMALISE.get(re.sub(r"\s+", " ", raw).strip().lower(), raw)


def source_attribution_accuracy(response_text: str, source_nodes: list) -> dict:
    """Detect phantom document attributions: (document, article) pairs cited in the
    response that do not appear in any retrieved chunk.

    Beyond the existing phantom-citation check (which only checks article numbers),
    this verifies that when the model writes 'AI Act Article 5' the retrieved chunks
    actually contain an AI Act Article 5 chunk — not just *some* Article 5 chunk.

    Returns:
        attribution_accurate   : bool — True iff phantom_attributions is empty
        phantom_attributions   : list of "DOC Art.N" strings that are uncovered
    """
    # Build set of (document, article) pairs available in retrieved chunks
    available: set[tuple[str, str]] = set()
    for node_with_score in source_nodes:
        node = node_with_score.node if hasattr(node_with_score, "node") else node_with_score
        doc = node.metadata.get("document", "")
        for art in str(node.metadata.get("chunk_articles", "")).split(","):
            art = art.strip()
            if doc and art:
                available.add((doc, art))

    # Extract (document, article) mentions from the response
    cited: set[tuple[str, str]] = set()
    for m in _ATTR_DOC_FIRST_RE.finditer(response_text):
        cited.add((_normalise_doc(m.group(1)), m.group(2)))
    for m in _ATTR_ART_FIRST_RE.finditer(response_text):
        cited.add((_normalise_doc(m.group(2)), m.group(1)))

    phantoms = [f"{doc} Art.{art}" for doc, art in sorted(cited) if (doc, art) not in available]
    return {
        "attribution_accurate": len(phantoms) == 0,
        "phantom_attributions": phantoms,
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
    reranker: bool = False,
    reranker_top_n: int = 5,
) -> list[dict]:
    """Run every dataset entry through the query engine and return raw results."""
    tool = create_legal_rag_tool(
        similarity_top_k=similarity_top_k,
        reranker=reranker,
        reranker_top_n=reranker_top_n,
    )
    query_engine = tool.query_engine

    with open(dataset_path) as f:
        dataset = json.load(f)

    results: list[dict] = []

    retriever = query_engine._retriever
    synthesizer = query_engine._response_synthesizer

    for i, item in enumerate(dataset, 1):
        print(f"[{i:>2}/{len(dataset)}] {item['id']}", flush=True)

        reset_query_profile()

        # ── retrieval: raw retriever (excludes postprocessors) ───────────────
        t_ret = time.time()
        raw_nodes = retriever.retrieve(item["query"])
        retrieval_latency = time.time() - t_ret

        # ── postprocessing: re-ranker and other node postprocessors ──────────
        # query_engine.retrieve() = raw retrieval + postprocessors; calling it
        # directly ensures the re-ranker (if enabled) is applied correctly.
        from llama_index.core.schema import QueryBundle
        query_bundle = QueryBundle(query_str=item["query"])
        source_nodes = query_engine._apply_node_postprocessors(raw_nodes, query_bundle=query_bundle)

        # ── synthesis (timed separately via profiling hook) ─────────────────
        t_total = time.time()
        response = synthesizer.synthesize(item["query"], nodes=source_nodes)
        total_synthesis = time.time() - t_total

        latency = retrieval_latency + total_synthesis

        profile = get_query_profile()
        gen_latency = profile.get("gen_latency_s", 0.0)
        prompt_tokens = profile.get("prompt_token_count", 0)
        headers_seen: set[str] = set(profile.get("headers_seen", []))

        # ── truncation / reranker drop detection ────────────────────────────
        # Compare raw_nodes (pre-postprocessor) vs headers_seen (in prompt).
        headers_retrieved: set[str] = set()
        for nws in raw_nodes:
            node = nws.node if hasattr(nws, "node") else nws
            m = _CHUNK_HEADER_RE.search(node.text)
            if m:
                headers_retrieved.add(m.group(1).strip())

        dropped_headers = sorted(headers_retrieved - headers_seen)
        truncated = len(dropped_headers) > 0

        response_text = str(response)

        if item["out_of_scope"]:
            results.append({
                "id": item["id"],
                "out_of_scope": True,
                "query": item["query"],
                "response": response_text,
                "oos_refused": detect_oos_refusal(response_text),
                "latency_s": round(latency, 2),
                "retrieval_latency_s": round(retrieval_latency, 3),
                "gen_latency_s": round(gen_latency, 3),
                "prompt_tokens": prompt_tokens,
                "chunks_retrieved": len(source_nodes),
                "chunks_in_prompt": len(headers_seen),
                "truncated": truncated,
                "dropped_chunks": dropped_headers,
            })
        else:
            is_cross_doc = item.get("cross_document", False)

            if is_cross_doc:
                recall = cross_doc_retrieval_recall(
                    source_nodes,
                    item["expected_articles"],
                    item.get("expected_documents", []),
                )
                recall_value = recall["cross_doc_recall"]
            else:
                recall_value = retrieval_recall(
                    source_nodes,
                    item["expected_articles"],
                    expected_document=item.get("expected_document"),
                )

            citation = verify_citations(response_text, source_nodes)
            attribution = source_attribution_accuracy(response_text, source_nodes)
            fact = fact_completeness(response_text, item["expected_answer"])

            result = {
                "id": item["id"],
                "out_of_scope": False,
                "cross_document": is_cross_doc,
                "expected_document": item.get("expected_document"),
                "query": item["query"],
                "expected_answer": item["expected_answer"],
                "response": response_text,
                "retrieval_recall": round(recall_value, 3),
                "phantom_citations": citation["phantom_articles"],
                "phantom_recital_citations": citation["phantom_recitals"],
                "citation_accurate": citation["is_accurate"],
                "attribution_accurate": attribution["attribution_accurate"],
                "phantom_attributions": attribution["phantom_attributions"],
                "fact_score": fact["fact_score"],
                "facts_found": fact["facts_found"],
                "facts_missing": fact["facts_missing"],
                "latency_s": round(latency, 2),
                "retrieval_latency_s": round(retrieval_latency, 3),
                "gen_latency_s": round(gen_latency, 3),
                "prompt_tokens": prompt_tokens,
                "chunks_retrieved": len(source_nodes),
                "chunks_in_prompt": len(headers_seen),
                "truncated": truncated,
                "dropped_chunks": dropped_headers,
            }

            if is_cross_doc:
                result["cross_doc_details"] = recall["cross_doc_details"]

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
    cross_doc = [r for r in in_scope if r.get("cross_document")]

    avg_recall = sum(r["retrieval_recall"] for r in in_scope) / len(in_scope) if in_scope else 0
    n_accurate = sum(1 for r in in_scope if r["citation_accurate"])
    citation_rate = n_accurate / len(in_scope) if in_scope else 0

    n_attr_accurate = sum(1 for r in in_scope if r.get("attribution_accurate", True))
    attribution_rate = n_attr_accurate / len(in_scope) if in_scope else 0

    fact_scores = [r["fact_score"] for r in in_scope if r.get("fact_score") is not None]
    avg_fact = sum(fact_scores) / len(fact_scores) if fact_scores else None

    has_judge = any("judge_score" in r for r in in_scope)
    judge_scores = [r["judge_score"] for r in in_scope if r.get("judge_score", -1) >= 0]
    avg_judge = sum(judge_scores) / len(judge_scores) if judge_scores else None

    avg_latency = sum(r["latency_s"] for r in results) / len(results) if results else 0

    n_oos_refused = sum(1 for r in out_scope if r.get("oos_refused", False))
    oos_refusal_rate = n_oos_refused / len(out_scope) if out_scope else None

    print("\n" + "=" * 56)
    print("EVAL SUMMARY")
    print("=" * 56)
    print(f"Total entries         : {len(results)}")
    print(f"In-scope              : {len(in_scope)}  (cross-document: {len(cross_doc)})")
    print(f"Out-of-scope          : {len(out_scope)}")
    print(f"Avg retrieval recall  : {avg_recall:.1%}")
    if cross_doc:
        avg_cd_recall = sum(r["retrieval_recall"] for r in cross_doc) / len(cross_doc)
        print(f"  Cross-doc recall    : {avg_cd_recall:.1%}  ({len(cross_doc)} entries)")
    print(f"Citation accuracy     : {citation_rate:.1%}  ({n_accurate}/{len(in_scope)} clean)")
    print(f"Attribution accuracy  : {attribution_rate:.1%}  ({n_attr_accurate}/{len(in_scope)} clean)")
    if avg_fact is not None:
        n_full = sum(1 for r in in_scope if r.get("fact_score", 0) == 1.0)
        print(f"Avg fact completeness : {avg_fact:.1%}  ({n_full}/{len(in_scope)} perfect)")
    if has_judge and avg_judge is not None:
        n_correct = sum(1 for s in judge_scores if s == 2)
        n_partial = sum(1 for s in judge_scores if s == 1)
        n_wrong   = sum(1 for s in judge_scores if s == 0)
        print(f"Avg judge score       : {avg_judge:.2f}/2  "
              f"({n_correct} correct / {n_partial} partial / {n_wrong} wrong)")
    if oos_refusal_rate is not None:
        print(f"OOS refusal rate      : {oos_refusal_rate:.1%}  ({n_oos_refused}/{len(out_scope)} correctly refused)")
    print(f"Avg latency           : {avg_latency:.1f}s")

    # Per-document breakdown
    by_doc: dict[str, list[dict]] = {}
    for r in in_scope:
        doc_key = r.get("expected_document") or "Cross-document"
        by_doc.setdefault(doc_key, []).append(r)

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

    # Flag attribution failures
    attr_failures = [r for r in in_scope if not r.get("attribution_accurate", True)]
    if attr_failures:
        print(f"\nPhantom attribution failures ({len(attr_failures)}):")
        for r in attr_failures:
            print(f"  {r['id']}: {r.get('phantom_attributions', [])}")

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

    # Out-of-scope responses
    if out_scope:
        print(f"\nOut-of-scope responses ({n_oos_refused}/{len(out_scope)} refused):")
        for r in out_scope:
            refused_tag = "REFUSED" if r.get("oos_refused") else "ANSWERED"
            snippet = r["response"][:100].replace("\n", " ")
            print(f"  [{refused_tag}] {r['id']}: {snippet}...")

    print_profiling_stats(results)


# ---------------------------------------------------------------------------
# Profiling stats
# ---------------------------------------------------------------------------


def _percentile(sorted_vals: list[float], p: int) -> float:
    if not sorted_vals:
        return 0.0
    idx = int(len(sorted_vals) * p / 100)
    return sorted_vals[min(idx, len(sorted_vals) - 1)]


def print_profiling_stats(results: list[dict]) -> None:
    """Print distribution stats for token counts and latency splits."""
    entries = [r for r in results if "prompt_tokens" in r]
    if not entries:
        return

    def _stats(vals: list[float], label: str, unit: str = "") -> None:
        s = sorted(vals)
        print(f"  {label:<30} mean={sum(s)/len(s):.1f}{unit}  "
              f"p90={_percentile(s, 90):.1f}{unit}  max={max(s):.1f}{unit}")

    print("\n" + "=" * 56)
    print("PROFILING STATS  (n={})".format(len(entries)))
    print("=" * 56)

    _stats([r["prompt_tokens"] for r in entries], "Prompt tokens", "")
    _stats([r["retrieval_latency_s"] for r in entries], "Retrieval latency", "s")
    _stats([r["gen_latency_s"] for r in entries], "Generation latency", "s")
    _stats([r["latency_s"] for r in entries], "Total latency", "s")
    _stats([r["chunks_retrieved"] for r in entries], "Chunks retrieved", "")
    _stats([r["chunks_in_prompt"] for r in entries], "Chunks in prompt", "")

    n_truncated = sum(1 for r in entries if r.get("truncated"))
    print(f"\n  Truncated queries: {n_truncated}/{len(entries)}")
    if n_truncated:
        truncated = [r for r in entries if r.get("truncated")]
        dropped_counts = [len(r["dropped_chunks"]) for r in truncated]
        print(f"  Dropped chunks (when truncated): "
              f"mean={sum(dropped_counts)/len(dropped_counts):.1f}  "
              f"max={max(dropped_counts)}")
        for r in truncated:
            print(f"    {r['id']}: dropped {r['dropped_chunks']}")


# ---------------------------------------------------------------------------
# MLflow experiment tracking
# ---------------------------------------------------------------------------

_MLFLOW_EXPERIMENT = "eu-legal-rag-eval"


def _doc_key(name: str) -> str:
    """Normalise a document name to a valid MLflow metric key segment."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", name).strip("_")


def log_to_mlflow(
    results: list[dict],
    *,
    top_k: int,
    dataset: str,
    run_judge: bool,
    chunking_strategy: str,
    reranker: bool,
    reranker_top_n: int = 5,
    out_path: str,
) -> None:
    """Log an eval run to MLflow. Silently skips if MLflow is not installed.

    Parameters logged:
        top_k, dataset, run_judge, chunking_strategy, reranker, reranker_top_n

    Metrics logged (aggregate):
        avg_retrieval_recall, cross_doc_recall, citation_accuracy,
        attribution_accuracy, avg_fact_score, avg_judge_score,
        oos_refusal_rate, avg_latency_s

    Metrics logged (per document):
        recall_<doc>, fact_<doc>, judge_<doc>

    Artifact: the full results JSON file.
    """
    try:
        import mlflow
    except ImportError:
        print("MLflow not installed — skipping experiment tracking.")
        return

    in_scope = [r for r in results if not r["out_of_scope"]]
    out_scope = [r for r in results if r["out_of_scope"]]
    cross_doc = [r for r in in_scope if r.get("cross_document")]

    mlflow.set_experiment(_MLFLOW_EXPERIMENT)
    with mlflow.start_run():
        # ── parameters ──────────────────────────────────────────────────────
        mlflow.log_params({
            "top_k": top_k,
            "dataset": dataset,
            "run_judge": run_judge,
            "chunking_strategy": chunking_strategy,
            "reranker": reranker,
            "reranker_top_n": reranker_top_n if reranker else None,
            "n_entries": len(results),
            "n_in_scope": len(in_scope),
            "n_oos": len(out_scope),
        })

        # ── aggregate metrics ────────────────────────────────────────────────
        if in_scope:
            avg_recall = sum(r["retrieval_recall"] for r in in_scope) / len(in_scope)
            mlflow.log_metric("avg_retrieval_recall", round(avg_recall, 4))

            if cross_doc:
                cd_recall = sum(r["retrieval_recall"] for r in cross_doc) / len(cross_doc)
                mlflow.log_metric("cross_doc_recall", round(cd_recall, 4))

            n_cit = sum(1 for r in in_scope if r.get("citation_accurate", True))
            mlflow.log_metric("citation_accuracy", round(n_cit / len(in_scope), 4))

            n_attr = sum(1 for r in in_scope if r.get("attribution_accurate", True))
            mlflow.log_metric("attribution_accuracy", round(n_attr / len(in_scope), 4))

            fact_vals = [r["fact_score"] for r in in_scope if r.get("fact_score") is not None]
            if fact_vals:
                mlflow.log_metric("avg_fact_score", round(sum(fact_vals) / len(fact_vals), 4))

            judge_vals = [r["judge_score"] for r in in_scope if r.get("judge_score", -1) >= 0]
            if judge_vals:
                mlflow.log_metric("avg_judge_score", round(sum(judge_vals) / len(judge_vals), 4))

        if out_scope:
            n_refused = sum(1 for r in out_scope if r.get("oos_refused", False))
            mlflow.log_metric("oos_refusal_rate", round(n_refused / len(out_scope), 4))

        all_latencies = [r["latency_s"] for r in results if "latency_s" in r]
        if all_latencies:
            mlflow.log_metric("avg_latency_s", round(sum(all_latencies) / len(all_latencies), 3))

        prof_entries = [r for r in results if "prompt_tokens" in r]
        if prof_entries:
            mlflow.log_metric("avg_prompt_tokens",
                              round(sum(r["prompt_tokens"] for r in prof_entries) / len(prof_entries), 1))
            mlflow.log_metric("avg_retrieval_latency_s",
                              round(sum(r["retrieval_latency_s"] for r in prof_entries) / len(prof_entries), 3))
            mlflow.log_metric("avg_gen_latency_s",
                              round(sum(r["gen_latency_s"] for r in prof_entries) / len(prof_entries), 3))
            mlflow.log_metric("truncation_rate",
                              round(sum(1 for r in prof_entries if r.get("truncated")) / len(prof_entries), 4))

        # ── per-document metrics ─────────────────────────────────────────────
        by_doc: dict[str, list[dict]] = {}
        for r in in_scope:
            key = r.get("expected_document") or "Cross_document"
            by_doc.setdefault(key, []).append(r)

        for doc, rows in by_doc.items():
            k = _doc_key(doc)
            doc_recall = sum(r["retrieval_recall"] for r in rows) / len(rows)
            mlflow.log_metric(f"recall_{k}", round(doc_recall, 4))

            fact_vals = [r["fact_score"] for r in rows if r.get("fact_score") is not None]
            if fact_vals:
                mlflow.log_metric(f"fact_{k}", round(sum(fact_vals) / len(fact_vals), 4))

            judge_vals = [r["judge_score"] for r in rows if r.get("judge_score", -1) >= 0]
            if judge_vals:
                mlflow.log_metric(f"judge_{k}", round(sum(judge_vals) / len(judge_vals), 4))

        # ── artifact ────────────────────────────────────────────────────────
        mlflow.log_artifact(out_path)

        run_id = mlflow.active_run().info.run_id
        print(f"MLflow run logged → experiment='{_MLFLOW_EXPERIMENT}' run_id={run_id}")


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
    parser.add_argument("--no-mlflow", action="store_true",
                        help="Disable MLflow experiment tracking")
    parser.add_argument("--chunking-strategy", default="article-boundary",
                        help="Label for the chunking strategy (default: article-boundary)")
    parser.add_argument("--reranker", action="store_true",
                        help="Enable BAAI/bge-reranker-base cross-encoder after retrieval")
    parser.add_argument("--reranker-top-n", type=int, default=5,
                        help="Chunks to keep after re-ranking (default: 5)")
    args = parser.parse_args()

    results = run_eval(
        args.dataset,
        args.top_k,
        run_judge=not args.no_judge,
        reranker=args.reranker,
        reranker_top_n=args.reranker_top_n,
    )
    print_summary(results)

    out_path = args.out or f"data/eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved → {out_path}")

    if not args.no_mlflow:
        log_to_mlflow(
            results,
            top_k=args.top_k,
            dataset=args.dataset,
            run_judge=not args.no_judge,
            chunking_strategy=args.chunking_strategy,
            reranker=args.reranker,
            reranker_top_n=args.reranker_top_n,
            out_path=out_path,
        )
