"""
RAG pipeline over EU legal documents (GDPR, AI Act, Data Governance Act,
Data Act, Cyber Resilience Act) using LlamaIndex + Qwen2.5-7B-Instruct local LLM.

Usage:
    from rag_pipeline import create_legal_rag_tool

    tool = create_legal_rag_tool()          # builds / loads index
    result = tool.query_engine.query("...")  # direct query
    # or pass `tool` to a LlamaIndex agent as a QueryEngineTool

Agent usage (LlamaIndex 0.12 workflow-based ReActAgent):
    from llama_index.core.agent.workflow import ReActAgent
    agent = ReActAgent(tools=[tool], llm=Settings.llm)
    response = await agent.run("What are the prohibited AI practices?")
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any

import torch
from dotenv import load_dotenv
from threading import Thread

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer, pipeline

from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.base.llms.types import CompletionResponse, LLMMetadata
from llama_index.core.llms import CustomLLM
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import Document, TextNode
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DOCS_DIR = Path("data/rag")
PERSIST_DIR = Path("data/rag_index")

# ---------------------------------------------------------------------------
# Article-boundary chunker
# ---------------------------------------------------------------------------

# Maps filename → display name used in chunk headers and metadata.
_DOC_NAMES: dict[str, str] = {
    "gdpr.txt": "GDPR",
    "ai_act.txt": "AI Act",
    "data_act.txt": "Data Act",
    "data_governance_act.txt": "Data Governance Act",
    "cyber_resilience_act.txt": "Cyber Resilience Act",
}

# Matches "Article 5" / "Article\xa05" (non-breaking space used by EUR-Lex PDFs)
# anchored to start-of-line with optional trailing whitespace.
_ARTICLE_LINE_RE = re.compile(r"^Article[\s\xa0]+(\d+)[\s\xa0]*$")

# Top-level numbered paragraph inside an article body, e.g. "1.   " or "1.\xa0\xa0"
_PARA_RE = re.compile(r"^\d+\.[\s\xa0]")


def _build_article_nodes(docs_dir: Path, sub_chunk_tokens: int = 512) -> list[TextNode]:
    """Parse every .txt file in *docs_dir* on Article-N boundaries.

    Each article becomes one TextNode prefixed with a ``[Doc | Article N — Title]``
    header.  Articles whose text exceeds *sub_chunk_tokens* (rough 4-chars-per-token
    estimate) are further split on their top-level numbered-paragraph boundaries
    (``1.``, ``2.``, …), with the same header prepended to every sub-chunk.

    Content before the first article (preamble / recitals) is split with
    SentenceSplitter and tagged ``[Doc | Preamble]``.
    """
    max_chars = sub_chunk_tokens * 4  # rough chars-per-token budget
    splitter = SentenceSplitter(chunk_size=sub_chunk_tokens, chunk_overlap=64)
    nodes: list[TextNode] = []

    for txt_file in sorted(docs_dir.glob("*.txt")):
        doc_name = _DOC_NAMES.get(txt_file.name, txt_file.stem)
        raw_lines = txt_file.read_text(encoding="utf-8").splitlines()

        # ── locate every "Article N" header line ────────────────────────────
        article_starts: list[tuple[int, str]] = []
        for i, line in enumerate(raw_lines):
            m = _ARTICLE_LINE_RE.match(line)
            if m:
                article_starts.append((i, m.group(1)))

        # ── preamble (recitals, whereas clauses, …) ─────────────────────────
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

        # ── one chunk (or more) per article ─────────────────────────────────
        article_starts.append((len(raw_lines), None))  # sentinel

        for idx, (start_i, art_no) in enumerate(article_starts[:-1]):
            end_i = article_starts[idx + 1][0]
            article_lines = raw_lines[start_i:end_i]

            # Title is normally two lines after the "Article N" line
            raw_title = article_lines[2].strip().rstrip("`") if len(article_lines) > 2 else ""
            # Discard if it looks like body text (starts with digit or open-paren)
            if raw_title and re.match(r"^[\d(]", raw_title):
                raw_title = ""

            header = f"[{doc_name} | Article {art_no}]"
            if raw_title:
                header += f" — {raw_title}"
            header += "\n\n"

            # Prepend a plain-language identity sentence so the embedding model
            # captures the document name as body content, not just as a bracket tag.
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
                # ── sub-split on top-level paragraph boundaries ──────────────
                # Collect indices of lines that start a new top-level paragraph
                para_breaks = [
                    i for i, line in enumerate(article_lines)
                    if _PARA_RE.match(line)
                ]

                if len(para_breaks) >= 2:
                    # Build one chunk per paragraph (or per pair when they're tiny)
                    para_breaks.append(len(article_lines))  # sentinel
                    for p_idx in range(len(para_breaks) - 1):
                        seg = article_lines[para_breaks[p_idx]: para_breaks[p_idx + 1]]
                        seg_text = header + identity + "\n".join(seg).strip()
                        meta = {**base_meta, "chunk_part": p_idx + 1}
                        if len(seg_text) <= max_chars:
                            nodes.append(TextNode(text=seg_text, metadata=meta))
                        else:
                            # Paragraph still too long → SentenceSplitter fallback
                            for j, sub in enumerate(splitter.get_nodes_from_documents(
                                    [Document(text="\n".join(seg))])):
                                nodes.append(TextNode(
                                    text=header + identity + sub.text,
                                    metadata={**meta, "chunk_subpart": j + 1},
                                ))
                else:
                    # No paragraph markers (e.g. pure definition list) → SentenceSplitter
                    # article_body already contains the identity sentence.
                    for j, sub in enumerate(splitter.get_nodes_from_documents(
                            [Document(text=article_body)])):
                        nodes.append(TextNode(
                            text=header + sub.text,
                            metadata={**base_meta, "chunk_part": j + 1},
                        ))

    return nodes


# ---------------------------------------------------------------------------
# Qwen2.5-7B-Instruct custom LLM wrapper
# ---------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
CONTEXT_WINDOW = 32768  # Qwen2.5-7B supports up to 32K tokens; 4096 was causing silent truncation
MAX_NEW_TOKENS = 512

# ---------------------------------------------------------------------------
# Per-query profiling state
# ---------------------------------------------------------------------------

# Matches bracketed chunk headers like "[GDPR | Article 5]" or "[AI Act | Preamble]"
_CHUNK_HEADER_RE = re.compile(r'\[([^\]]+\s*\|\s*[^\]]+)\]')

_query_profile: dict = {}


def reset_query_profile() -> None:
    """Clear the per-query profiling accumulator. Call before each query."""
    global _query_profile
    _query_profile = {}


def get_query_profile() -> dict:
    """Return a snapshot of the current per-query profiling data."""
    p = dict(_query_profile)
    # Convert set to sorted list for JSON-serializability
    if isinstance(p.get("headers_seen"), set):
        p["headers_seen"] = sorted(p["headers_seen"])
    return p


class QwenLLM(CustomLLM):
    """LlamaIndex-compatible wrapper around a local Qwen2.5-7B-Instruct pipeline."""

    model_id: str = MODEL_ID
    max_new_tokens: int = MAX_NEW_TOKENS
    _pipe: Any = None  # not a pydantic field

    def _get_pipeline(self):
        if self._pipe is None:
            hf_token = os.getenv("HF_TOKEN")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_enable_fp32_cpu_offload=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, token=hf_token
            )
            # Qwen has a dedicated pad token; fall back to eos if absent
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id

            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=quantization_config,
                device_map="auto",
                dtype=torch.float16, 
                token=hf_token,
                max_memory={0: "8GiB", "cpu": "20GiB"},
            )
            # Clear max_length from the stored generation_config so it doesn't
            # conflict with max_new_tokens passed at call time.
            model.generation_config.max_length = None
            self._pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                return_full_text=False,
            )
        return self._pipe

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=CONTEXT_WINDOW,
            num_output=self.max_new_tokens,
            model_name=self.model_id,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **_kwargs: Any) -> CompletionResponse:
        pipe = self._get_pipeline()
        # Apply Qwen's ChatML template manually so we pass a plain string.
        # With return_full_text=False this guarantees generated_text is a plain string.
        formatted = pipe.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

        # --- profiling: token count ---
        input_ids = pipe.tokenizer(formatted, return_tensors="pt").input_ids
        prompt_tokens = int(input_ids.shape[-1])
        _query_profile["prompt_token_count"] = (
            _query_profile.get("prompt_token_count", 0) + prompt_tokens
        )
        _query_profile["llm_calls"] = _query_profile.get("llm_calls", 0) + 1

        # Safety warning: flag if prompt consumes >80% of the generation budget.
        # Static analysis shows worst-case ~7800 tokens with top_k=10 and
        # CONTEXT_WINDOW=32768, so this should never fire under normal operation.
        _budget = CONTEXT_WINDOW - MAX_NEW_TOKENS
        if prompt_tokens > int(_budget * 0.8):
            import warnings
            warnings.warn(
                f"Prompt is {prompt_tokens} tokens ({prompt_tokens/_budget:.0%} of budget "
                f"{_budget}). Consider reducing top_k or increasing CONTEXT_WINDOW.",
                RuntimeWarning,
                stacklevel=2,
            )

        # --- profiling: which chunk headers appear in this prompt call ---
        headers_in_call = set(_CHUNK_HEADER_RE.findall(prompt))
        all_headers: set = _query_profile.get("headers_seen", set())
        all_headers.update(h.strip() for h in headers_in_call)
        _query_profile["headers_seen"] = all_headers

        # --- profiling: generation latency ---
        _t_gen = time.time()
        output = pipe(
            formatted,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )[0]["generated_text"]
        _query_profile["gen_latency_s"] = (
            _query_profile.get("gen_latency_s", 0.0) + (time.time() - _t_gen)
        )

        return CompletionResponse(text=output)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **_kwargs: Any):
        pipe = self._get_pipeline()
        formatted = pipe.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = pipe.tokenizer(formatted, return_tensors="pt").to(pipe.model.device)
        streamer = TextIteratorStreamer(
            pipe.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        thread = Thread(
            target=pipe.model.generate,
            kwargs={
                **inputs,
                "max_new_tokens": self.max_new_tokens,
                "do_sample": False,
                "streamer": streamer,
            },
        )
        thread.start()

        text = ""
        for token in streamer:
            text += token
            yield CompletionResponse(text=text, delta=token)


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------


def build_index(force_rebuild: bool = False) -> tuple[VectorStoreIndex, list[TextNode]]:
    """Load index from disk or build it from scratch.

    Returns both the VectorStoreIndex and the list of TextNodes so that
    BM25Retriever can be constructed from the same node objects.

    Args:
        force_rebuild: If True, re-index even if a persisted index exists.
    """
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.llm = QwenLLM()

    nodes = _build_article_nodes(DOCS_DIR)

    if not force_rebuild and PERSIST_DIR.exists():
        print(f"Loading index from {PERSIST_DIR} …")
        storage_context = StorageContext.from_defaults(persist_dir=str(PERSIST_DIR))
        return load_index_from_storage(storage_context), nodes

    print(f"Building index from {DOCS_DIR} (article-boundary chunking) …")
    print(f"  → {len(nodes)} chunks created")
    index = VectorStoreIndex(nodes, show_progress=True)
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(PERSIST_DIR))
    print(f"Index persisted to {PERSIST_DIR}")
    return index, nodes


# ---------------------------------------------------------------------------
# Module-level retriever — used by app.py to fetch source nodes independently
# ---------------------------------------------------------------------------

_retriever = None

LEGAL_QA_PROMPT = PromptTemplate(
    "You are an EU legal expert assistant. Answer using ONLY the context provided.\n\n"
    "RULES:\n"
    "0. If the context does not contain information relevant to the question, respond ONLY with: "
    "This question is outside the scope of the available EU regulations.\n"
    "1. Only cite Article or Recital numbers that explicitly appear in the context below.\n"
    "2. Do NOT use training knowledge to add article numbers absent from the context.\n"
    "3. If unsure, say 'The retrieved documents indicate...' without citing specific numbers.\n"
    "4. When both an Article and a Recital cover the same topic, cite the Article as the authoritative source and the Recital only as supplementary context.\n\n"
    "Context:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n\n"
    "Question: {query_str}\n"
    "Answer:"
)


def get_retriever():
    """Return the retriever built during create_legal_rag_tool()."""
    return _retriever


# ---------------------------------------------------------------------------
# Citation verification
# ---------------------------------------------------------------------------


def verify_citations(response_text: str, source_nodes: list) -> dict:
    """
    Detect phantom citations: article/recital numbers the LLM mentioned
    that do NOT appear in the retrieved source chunks.
    Returns a dict with phantom_articles, phantom_recitals, and is_accurate flag.
    """
    cited_articles = set(re.findall(r'[Aa]rticle\s+(\d+)', response_text))
    cited_recitals = set(re.findall(r'[Rr]ecital\s+(\d+)', response_text))

    available_articles: set[str] = set()
    available_recitals: set[str] = set()
    for node_with_score in source_nodes:
        node = node_with_score.node if hasattr(node_with_score, "node") else node_with_score
        # Use chunk_articles metadata (set at index time) rather than scanning the full
        # text. Text scanning picks up cross-references within articles (e.g. "pursuant
        # to Article 83" inside an Article 5 chunk), causing false negatives where the
        # LLM cites a cross-referenced article and the verifier incorrectly accepts it.
        for art in str(node.metadata.get("chunk_articles", "")).split(","):
            art = art.strip()
            if art and art != "preamble":
                available_articles.add(art)
        # Recitals have no metadata equivalent — scan preamble chunk text only.
        if node.metadata.get("chunk_type") == "preamble":
            available_recitals.update(re.findall(r'[Rr]ecital\s+(\d+)', node.text))

    phantom_articles = sorted(cited_articles - available_articles, key=int)
    phantom_recitals = sorted(cited_recitals - available_recitals, key=int)

    return {
        "phantom_articles": phantom_articles,
        "phantom_recitals": phantom_recitals,
        "is_accurate": not phantom_articles and not phantom_recitals,
    }


# ---------------------------------------------------------------------------
# Agent tool factory
# ---------------------------------------------------------------------------


RERANKER_MODEL = "BAAI/bge-reranker-base"


def create_legal_rag_tool(
    similarity_top_k: int = 10,
    force_rebuild: bool = False,
    reranker: bool = False,
    reranker_top_n: int = 5,
) -> QueryEngineTool:
    """Return a QueryEngineTool backed by hybrid BM25 + dense retrieval.

    BM25 handles exact-term matching (e.g. regulation names like
    'Data Governance Act') while the dense retriever handles semantic
    similarity.  QueryFusionRetriever merges both ranked lists via
    Reciprocal Rank Fusion before passing context to the LLM.

    Args:
        similarity_top_k: Chunks retrieved by each individual retriever.
            The fused list contains up to 2 × similarity_top_k candidates
            before deduplication.
        force_rebuild: Rebuild the vector index even if a cached one exists.
        reranker: If True, apply a BAAI/bge-reranker-base cross-encoder after
            retrieval to re-score the top-k candidates and keep only the top
            reranker_top_n for synthesis.
        reranker_top_n: Number of chunks to keep after re-ranking (default 5).
            Only used when reranker=True.
    """
    global _retriever
    index, nodes = build_index(force_rebuild=force_rebuild)

    # Each individual retriever fetches 2× the desired final output so that RRF
    # has a richer candidate pool to rerank before trimming to similarity_top_k.
    retriever_depth = similarity_top_k * 2
    dense_retriever = index.as_retriever(similarity_top_k=retriever_depth)
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=retriever_depth,
    )
    hybrid_retriever = QueryFusionRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        similarity_top_k=similarity_top_k,  # final fused output
        num_queries=1,          # no query rewriting — use the original query only
        mode="reciprocal_rerank",
        use_async=False,
    )

    _retriever = hybrid_retriever

    node_postprocessors = []
    if reranker:
        print(f"Re-ranker enabled: {RERANKER_MODEL}  top_n={reranker_top_n}")
        node_postprocessors.append(
            SentenceTransformerRerank(
                model=RERANKER_MODEL,
                top_n=reranker_top_n,
            )
        )

    query_engine = RetrieverQueryEngine.from_args(
        hybrid_retriever,
        node_postprocessors=node_postprocessors,
    )
    query_engine.update_prompts({"response_synthesizer:text_qa_template": LEGAL_QA_PROMPT})

    return QueryEngineTool(
        query_engine=query_engine,  # _SourceCapturingQueryEngine
        metadata=ToolMetadata(
            name="eu_legal_rag",
            description=(
                "Search and answer questions about EU data and AI legislation. "
                "Documents available: "
                "GDPR (General Data Protection Regulation — data privacy rights and obligations), "
                "AI Act (risk-based rules for AI systems, prohibited practices, conformity), "
                "Data Governance Act (data sharing frameworks and data intermediaries), "
                "Data Act (rights to access and share data generated by connected products), "
                "Cyber Resilience Act (cybersecurity requirements for products with digital elements). "
                "Use this tool for questions about compliance requirements, definitions, "
                "obligations, rights, prohibitions, penalties, or any other legal matter "
                "covered by these EU regulations."
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tool = create_legal_rag_tool()

    queries = [
        "What are the main obligations for providers of high-risk AI systems under the AI Act?",
        "What rights does a data subject have under the GDPR?",
        "How does the Data Act define 'data holder'?",
    ]

    for q in queries:
        print(f"\nQ: {q}")
        response = tool.query_engine.query(q)
        print(f"A: {response}")
