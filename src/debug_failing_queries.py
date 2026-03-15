"""Deep-dive debug script for the 3 known bad queries (AOC-05).

Loads the cached vector index (no GPU needed), runs retrieval for each
query, renders the full prompt, and records:
  - exact chunks returned in RRF order (with scores and metadata)
  - fully constructed prompt (as would be sent to the LLM)
  - position of the relevant fact within the context window
  - root-cause confirmation/refinement

Run from project root:
    python src/debug_failing_queries.py
"""

from __future__ import annotations

import re
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Avoid loading the heavy QwenLLM — we only need retrieval + prompt rendering.
# Override Settings.llm BEFORE importing rag_pipeline so QwenLLM is never
# constructed (it defers GPU loading to _get_pipeline(), but the __init__
# still touches pydantic field registration).
from llama_index.core import Settings
from llama_index.core.llms import MockLLM  # lightweight stand-in
Settings.llm = MockLLM()

from rag_pipeline import (  # noqa: E402
    build_index,
    LEGAL_QA_PROMPT,
    _build_article_nodes,
    DOCS_DIR,
)
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# ── Targets ─────────────────────────────────────────────────────────────────

QUERIES = [
    {
        "id": "ai_act_max_fine_prohibited",
        "query": (
            "What is the maximum administrative fine for non-compliance with "
            "the prohibited AI practices listed in Article 5 of the AI Act?"
        ),
        "expected_articles": ["99"],
        "expected_document": "AI Act",
        "key_facts": ["35,000,000", "7%"],
        "root_cause_hypothesis": (
            "Context confusion: Article 99 context contains tiers for 7%, 3%, and 1.5%. "
            "LLM substitutes the 3% value for 7%."
        ),
    },
    {
        "id": "ai_act_social_scoring_prohibition",
        "query": (
            "Does the AI Act prohibit AI systems used by public or private actors "
            "for social scoring of natural persons based on their behaviour or "
            "personal characteristics?"
        ),
        "expected_articles": ["5"],
        "expected_document": "AI Act",
        "key_facts": ["Article 5", "5(1)(c)"],
        "root_cause_hypothesis": (
            "Source attribution error: Preamble recital about social scoring ranked higher "
            "than normative Article 5. LLM cites Recital 31 instead of Article 5(1)(c)."
        ),
    },
    {
        "id": "data_act_data_holder_definition",
        "query": "How does the Data Act define 'data holder'?",
        "expected_articles": ["2"],
        "expected_document": "Data Act",
        "key_facts": ["data holder", "natural or legal person"],
        "root_cause_hypothesis": (
            "Hallucinated absence: correct chunk retrieved but LLM claims term is not defined. "
            "Likely caused by dense definition list in Article 2 exceeding 512-token sub-chunk budget."
        ),
    },
]

SIMILARITY_TOP_K = 10
SEP = "─" * 72


def _build_retriever(index, nodes):
    depth = SIMILARITY_TOP_K * 2
    dense = index.as_retriever(similarity_top_k=depth)
    bm25 = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=depth)
    return QueryFusionRetriever(
        retrievers=[dense, bm25],
        similarity_top_k=SIMILARITY_TOP_K,
        num_queries=1,
        mode="reciprocal_rerank",
        use_async=False,
    )


def _render_prompt(query: str, nodes: list) -> str:
    context_parts = []
    for nws in nodes:
        node = nws.node if hasattr(nws, "node") else nws
        context_parts.append(node.text)
    context_str = "\n\n".join(context_parts)
    return LEGAL_QA_PROMPT.format(context_str=context_str, query_str=query)


def _find_fact_positions(prompt: str, key_facts: list[str]) -> list[dict]:
    results = []
    for fact in key_facts:
        positions = [m.start() for m in re.finditer(re.escape(fact), prompt, re.IGNORECASE)]
        if positions:
            for pos in positions:
                # Show surrounding context
                snippet_start = max(0, pos - 80)
                snippet_end = min(len(prompt), pos + len(fact) + 80)
                snippet = prompt[snippet_start:snippet_end].replace("\n", " ")
                results.append({
                    "fact": fact,
                    "char_offset": pos,
                    "pct_through_prompt": round(pos / len(prompt) * 100, 1),
                    "snippet": f"...{snippet}...",
                })
        else:
            results.append({"fact": fact, "char_offset": -1, "snippet": "(NOT FOUND IN PROMPT)"})
    return results


def _approx_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return len(text) // 4


def main():
    print("Loading embed model and index …")
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    index, nodes = build_index()
    retriever = _build_retriever(index, nodes)
    print(f"Index loaded. {len(nodes)} total chunks.\n")

    for target in QUERIES:
        qid = target["id"]
        query = target["query"]
        expected_doc = target["expected_document"]
        expected_arts = target["expected_articles"]

        print(SEP)
        print(f"QUERY: {qid}")
        print(SEP)
        print(f"  Text : {query}")
        print(f"  Wants: {expected_doc} Article(s) {', '.join(expected_arts)}")
        print()

        # ── 1. Retrieval ─────────────────────────────────────────────────────
        raw_nodes = retriever.retrieve(query)

        print(f"  RETRIEVED CHUNKS ({len(raw_nodes)} total, in RRF rank order):")
        relevant_chunk_indices = []
        for rank, nws in enumerate(raw_nodes, 1):
            node = nws.node if hasattr(nws, "node") else nws
            doc = node.metadata.get("document", "?")
            art = node.metadata.get("article", "?")
            chunk_type = node.metadata.get("chunk_type", "article")
            chunk_part = node.metadata.get("chunk_part", "")
            part_suffix = f" part {chunk_part}" if chunk_part else ""
            score = round(nws.score, 5) if hasattr(nws, "score") and nws.score else 0.0
            tokens_est = _approx_tokens(node.text)

            # Flag if this chunk contains the expected article from the expected doc
            is_target = (
                doc == expected_doc
                and str(art) in expected_arts
            )
            flag = " ◄ TARGET" if is_target else ""
            if is_target:
                relevant_chunk_indices.append(rank)

            print(f"    [{rank:>2}] score={score:.5f}  {doc} | {chunk_type} {art}{part_suffix}"
                  f"  (~{tokens_est} tok){flag}")

        if relevant_chunk_indices:
            print(f"\n  Target chunk(s) at rank(s): {relevant_chunk_indices}")
        else:
            print("\n  !! Target chunk NOT FOUND in retrieved set !!")

        # ── 2. Full prompt ───────────────────────────────────────────────────
        prompt = _render_prompt(query, raw_nodes)
        prompt_tokens = _approx_tokens(prompt)

        print(f"\n  PROMPT STATS:")
        print(f"    Total chars  : {len(prompt):,}")
        print(f"    Approx tokens: {prompt_tokens:,}")
        print()
        print("  FULL PROMPT (truncated to 6000 chars for readability):")
        print()
        # Indent and wrap
        for line in prompt[:6000].split("\n"):
            print("    " + line)
        if len(prompt) > 6000:
            print(f"    … [+{len(prompt)-6000:,} chars truncated]")

        # ── 3. Fact position analysis ────────────────────────────────────────
        print(f"\n  KEY FACT POSITIONS IN PROMPT:")
        fact_positions = _find_fact_positions(prompt, target["key_facts"])
        for fp in fact_positions:
            if fp["char_offset"] >= 0:
                print(f"    '{fp['fact']}' @ char {fp['char_offset']:,} "
                      f"({fp['pct_through_prompt']}% through prompt)")
                print(f"      context: {fp['snippet']}")
            else:
                print(f"    '{fp['fact']}' — NOT FOUND IN PROMPT")

        # ── 4. Root-cause assessment ─────────────────────────────────────────
        print(f"\n  HYPOTHESIS (from technical report):")
        for line in textwrap.wrap(target["root_cause_hypothesis"], width=64):
            print(f"    {line}")

        # Assess whether preamble chunks outranked normative articles
        preamble_ranks = [
            rank for rank, nws in enumerate(raw_nodes, 1)
            if (nws.node if hasattr(nws, "node") else nws).metadata.get("chunk_type") == "preamble"
        ]
        article_ranks = [
            (rank, (nws.node if hasattr(nws, "node") else nws).metadata.get("article"))
            for rank, nws in enumerate(raw_nodes, 1)
            if (nws.node if hasattr(nws, "node") else nws).metadata.get("chunk_type") != "preamble"
        ]

        print(f"\n  STRUCTURAL ANALYSIS:")
        print(f"    Preamble chunks in top-10 : {len(preamble_ranks)}  (ranks: {preamble_ranks})")
        print(f"    Article chunks in top-10  : {len(article_ranks)}")
        if relevant_chunk_indices:
            top_preamble = min(preamble_ranks) if preamble_ranks else None
            top_target = min(relevant_chunk_indices)
            if top_preamble and top_preamble < top_target:
                print(f"    !! Preamble chunk ranked #{top_preamble} ABOVE target article at #{top_target}")
            elif top_preamble:
                print(f"    Target article at #{top_target} ranked ABOVE first preamble at #{top_preamble}")
            else:
                print(f"    No preamble chunks retrieved. Target article at #{top_target}.")

        # For ai_act_max_fine: count how many chunks contain competing percentages
        if qid == "ai_act_max_fine_prohibited":
            competing_pct_chunks = []
            for rank, nws in enumerate(raw_nodes, 1):
                node = nws.node if hasattr(nws, "node") else nws
                pcts = re.findall(r'\d+(?:\.\d+)?\s*%', node.text)
                if pcts:
                    competing_pct_chunks.append((rank, pcts))
            print(f"\n  COMPETING PERCENTAGE VALUES IN CONTEXT:")
            for rank, pcts in competing_pct_chunks:
                node = raw_nodes[rank-1].node if hasattr(raw_nodes[rank-1], "node") else raw_nodes[rank-1]
                doc = node.metadata.get("document", "?")
                art = node.metadata.get("article", "?")
                print(f"    Rank {rank:>2} ({doc} Art.{art}): {pcts}")

        print()

    print(SEP)
    print("Debug run complete.")
    print(SEP)


if __name__ == "__main__":
    main()
