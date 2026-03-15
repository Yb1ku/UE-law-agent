"""Ablation: identity sentence × BM25 hybrid — retrieval-only (no LLM generation).

Conditions (2×2):
  A — Dense only,   no identity sentence
  B — Dense only,   with identity sentence
  C — Hybrid BM25,  no identity sentence
  D — Hybrid BM25,  with identity sentence  ← current production setup

Only retrieval recall is measured (no LLM call needed), so all 4 conditions
run in a few minutes instead of hours.

Usage (run from project root):
    python src/ablation_identity_bm25.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import Document, TextNode
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever

DOCS_DIR = Path("data/rag")
DATASET_PATH = Path("data/eval_dataset.json")
TOP_K = 10

_DOC_NAMES: dict[str, str] = {
    "gdpr.txt": "GDPR",
    "ai_act.txt": "AI Act",
    "data_act.txt": "Data Act",
    "data_governance_act.txt": "Data Governance Act",
    "cyber_resilience_act.txt": "Cyber Resilience Act",
}
_ARTICLE_LINE_RE = re.compile(r"^Article[\s\xa0]+(\d+)[\s\xa0]*$")
_PARA_RE = re.compile(r"^\d+\.[\s\xa0]")


# ---------------------------------------------------------------------------
# Chunker with optional identity sentence
# ---------------------------------------------------------------------------

def build_nodes(use_identity: bool, sub_chunk_tokens: int = 512) -> list[TextNode]:
    max_chars = sub_chunk_tokens * 4
    splitter = SentenceSplitter(chunk_size=sub_chunk_tokens, chunk_overlap=64)
    nodes: list[TextNode] = []

    for txt_file in sorted(DOCS_DIR.glob("*.txt")):
        doc_name = _DOC_NAMES.get(txt_file.name, txt_file.stem)
        raw_lines = txt_file.read_text(encoding="utf-8").splitlines()

        article_starts: list[tuple[int, str]] = []
        for i, line in enumerate(raw_lines):
            m = _ARTICLE_LINE_RE.match(line)
            if m:
                article_starts.append((i, m.group(1)))

        preamble_end = article_starts[0][0] if article_starts else len(raw_lines)
        preamble_text = "\n".join(raw_lines[:preamble_end]).strip()
        if preamble_text:
            header = f"[{doc_name} | Preamble]\n\n"
            for sub in splitter.get_nodes_from_documents([Document(text=preamble_text)]):
                nodes.append(TextNode(
                    text=header + sub.text,
                    metadata={"document": doc_name, "article": "preamble",
                               "chunk_articles": "", "chunk_type": "preamble"},
                ))

        article_starts.append((len(raw_lines), None))

        for idx, (start_i, art_no) in enumerate(article_starts[:-1]):
            end_i = article_starts[idx + 1][0]
            article_lines = raw_lines[start_i:end_i]

            raw_title = article_lines[2].strip().rstrip("`") if len(article_lines) > 2 else ""
            if raw_title and re.match(r"^[\d(]", raw_title):
                raw_title = ""

            header = f"[{doc_name} | Article {art_no}]"
            if raw_title:
                header += f" — {raw_title}"
            header += "\n\n"

            # Identity sentence — present or absent depending on condition
            identity = ""
            if use_identity:
                identity = f"The following is Article {art_no} of the {doc_name}"
                if raw_title:
                    identity += f", titled '{raw_title}'"
                identity += ".\n\n"

            article_body = identity + "\n".join(article_lines).strip()
            full_text = header + article_body
            base_meta = {
                "document": doc_name,
                "article": art_no,
                "chunk_articles": art_no,
                "chunk_type": "article",
            }

            if len(full_text) <= max_chars:
                nodes.append(TextNode(text=full_text, metadata=base_meta))
            else:
                para_breaks = [i for i, line in enumerate(article_lines) if _PARA_RE.match(line)]

                if len(para_breaks) >= 2:
                    para_breaks.append(len(article_lines))
                    for p_idx in range(len(para_breaks) - 1):
                        seg = article_lines[para_breaks[p_idx]: para_breaks[p_idx + 1]]
                        seg_text = header + identity + "\n".join(seg).strip()
                        meta = {**base_meta, "chunk_part": p_idx + 1}
                        if len(seg_text) <= max_chars:
                            nodes.append(TextNode(text=seg_text, metadata=meta))
                        else:
                            for j, sub in enumerate(splitter.get_nodes_from_documents(
                                    [Document(text="\n".join(seg))])):
                                nodes.append(TextNode(
                                    text=header + identity + sub.text,
                                    metadata={**meta, "chunk_subpart": j + 1},
                                ))
                else:
                    for j, sub in enumerate(splitter.get_nodes_from_documents(
                            [Document(text=article_body)])):
                        nodes.append(TextNode(
                            text=header + sub.text,
                            metadata={**base_meta, "chunk_part": j + 1},
                        ))

    return nodes


# ---------------------------------------------------------------------------
# Retriever factory
# ---------------------------------------------------------------------------

def build_retriever(nodes: list[TextNode], use_bm25: bool, top_k: int):
    index = VectorStoreIndex(nodes, show_progress=False)
    dense = index.as_retriever(similarity_top_k=top_k)

    if not use_bm25:
        return dense

    bm25 = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=top_k)
    return QueryFusionRetriever(
        retrievers=[dense, bm25],
        similarity_top_k=top_k,
        num_queries=1,
        mode="reciprocal_rerank",
        use_async=False,
    )


# ---------------------------------------------------------------------------
# Retrieval recall (document-scoped)
# ---------------------------------------------------------------------------

def retrieval_recall(source_nodes, expected_articles: list[str], expected_document: str) -> float:
    if not expected_articles:
        return 1.0
    found: set[str] = set()
    for nws in source_nodes:
        node = nws.node if hasattr(nws, "node") else nws
        if node.metadata.get("document", "") != expected_document:
            continue
        for art in str(node.metadata.get("chunk_articles", "")).split(", "):
            if art.strip():
                found.add(art.strip())
    hits = sum(1 for art in expected_articles if art in found)
    return hits / len(expected_articles)


# ---------------------------------------------------------------------------
# Run one condition
# ---------------------------------------------------------------------------

def run_condition(label: str, use_identity: bool, use_bm25: bool, dataset: list[dict]) -> dict:
    print(f"\n{'='*56}")
    print(f"Condition {label}  |  identity={use_identity}  bm25={use_bm25}")
    print(f"{'='*56}")

    print("  Building nodes...", end=" ", flush=True)
    nodes = build_nodes(use_identity=use_identity)
    print(f"{len(nodes)} nodes")

    print("  Embedding...", end=" ", flush=True)
    retriever = build_retriever(nodes, use_bm25=use_bm25, top_k=TOP_K)
    print("done")

    in_scope = [e for e in dataset if not e["out_of_scope"]]
    recalls: list[float] = []
    per_doc: dict[str, list[float]] = {}
    failures: list[str] = []

    for entry in in_scope:
        source_nodes = retriever.retrieve(entry["query"])
        recall = retrieval_recall(source_nodes, entry["expected_articles"], entry["expected_document"])
        recalls.append(recall)
        per_doc.setdefault(entry["expected_document"], []).append(recall)
        if recall < 1.0:
            failures.append(f"  {entry['id']}: {recall:.0%}")

    avg = sum(recalls) / len(recalls)
    print(f"\n  Overall recall: {avg:.1%}")
    for doc, vals in sorted(per_doc.items()):
        doc_avg = sum(vals) / len(vals)
        print(f"    {doc:<30} {doc_avg:.1%}")
    if failures:
        print(f"  Failures:")
        for f in failures:
            print(f)

    return {
        "label": label,
        "use_identity": use_identity,
        "use_bm25": use_bm25,
        "avg_recall": round(avg, 4),
        "per_doc": {doc: round(sum(v)/len(v), 4) for doc, v in per_doc.items()},
        "failures": failures,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    # No LLM needed — retrieval only
    Settings.llm = None  # type: ignore[assignment]

    with open(DATASET_PATH) as f:
        dataset = json.load(f)

    conditions = [
        ("A", False, False),  # Dense only, no identity
        ("B", True,  False),  # Dense only, with identity
        ("C", False, True),   # Hybrid BM25, no identity
        ("D", True,  True),   # Hybrid BM25, with identity  ← production
    ]

    all_results = []
    for label, use_identity, use_bm25 in conditions:
        result = run_condition(label, use_identity, use_bm25, dataset)
        all_results.append(result)

    # Summary table
    print(f"\n{'='*56}")
    print("ABLATION SUMMARY (retrieval recall @ top_k=10)")
    print(f"{'='*56}")
    docs = sorted(all_results[0]["per_doc"].keys())
    col_w = 8
    header = f"  {'Condition':<28}" + "".join(f"{d[:col_w-1]:>{col_w}}" for d in docs) + f"  {'OVERALL':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in all_results:
        identity_str = "identity=Y" if r["use_identity"] else "identity=N"
        bm25_str     = "bm25=Y" if r["use_bm25"] else "bm25=N"
        label_str = f"{r['label']} ({identity_str}, {bm25_str})"
        row = f"  {label_str:<28}"
        for doc in docs:
            val = r["per_doc"].get(doc, 0)
            row += f"{val:.0%}".rjust(col_w)
        row += f"  {r['avg_recall']:.1%}".rjust(10)
        print(row)
