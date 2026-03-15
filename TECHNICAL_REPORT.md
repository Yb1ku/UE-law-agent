# Technical Report: EU Legal Documents RAG Pipeline

**Project:** EU Legal Q&A Agent
**Stack:** LlamaIndex 0.12 · Qwen2.5-7B-Instruct (4-bit) · BGE-small-en-v1.5 · BM25 Hybrid Retrieval · BGE-reranker-base (optional)
**Date:** March 2026 — updated after improvement sprint AOC-00 through GQ-03

---

## Table of Contents

1. [Overview](#1-overview)
2. [Document Corpus](#2-document-corpus)
3. [Chunking Strategy](#3-chunking-strategy)
4. [Embedding Model](#4-embedding-model)
5. [Retrieval Architecture](#5-retrieval-architecture)
6. [Language Model](#6-language-model)
7. [Prompt Engineering](#7-prompt-engineering)
8. [Citation Verification](#8-citation-verification)
9. [Evaluation Framework](#9-evaluation-framework)
10. [Results & Analysis](#10-results--analysis)
11. [Known Failures & Root Cause Analysis](#11-known-failures--root-cause-analysis)
12. [Limitations & Future Work](#12-limitations--future-work)

---

## 1. Overview

This system is a Retrieval-Augmented Generation (RAG) pipeline designed to answer questions about five EU regulations. It runs entirely on-premises using quantized open-source models, with no dependency on external APIs.

The high-level flow is:

```
User query
    │
    ▼
Hybrid Retriever (BM25 + Dense, fused via Reciprocal Rank Fusion)
    │
    ▼
Top-K article chunks (with document + article metadata)
    │
    ▼ [optional]
Cross-encoder re-ranker (BGE-reranker-base → top-N)
    │
    ▼
Qwen2.5-7B-Instruct (4-bit quantized, custom legal QA prompt)
    │
    ▼
Answer + source citations
    │
    ▼
Citation verifier (phantom citation check)
```

The pipeline is implemented in `src/rag_pipeline.py` and exposed as a LlamaIndex `QueryEngineTool`, making it compatible with both direct query and agent orchestration (e.g., LlamaIndex `ReActAgent`).

---

## 2. Document Corpus

Five EU regulations are stored as plain-text files in `data/rag/`, extracted from EUR-Lex official PDFs:

| File | Display Name | Lines |
|------|-------------|-------|
| `gdpr.txt` | GDPR | 4,770 |
| `ai_act.txt` | AI Act | 7,194 |
| `data_act.txt` | Data Act | 3,169 |
| `data_governance_act.txt` | Data Governance Act | 2,126 |
| `cyber_resilience_act.txt` | Cyber Resilience Act | 4,037 |
| **Total** | | **21,296** |

**Preprocessing notes:** EUR-Lex PDFs use Unicode non-breaking spaces (`\xa0`) heavily — in article headers, paragraph numbering, and mid-sentence line wraps. All regex patterns used for chunking explicitly account for `[\s\xa0]` rather than `\s` alone to avoid missed matches. Numeric values in EUR-Lex text also use Unicode non-breaking spaces as thousand separators (e.g. `35 000 000`, `7 %`), which required corresponding updates to the fact-extraction regex patterns in the evaluator (GQ-11).

---

## 3. Chunking Strategy

### 3.1 Motivation for Article-Boundary Chunking

The initial implementation used LlamaIndex's `SentenceSplitter` with a flat 1,024-token window and 64-token overlap. This approach had two critical failure modes for legal text:

1. **Article boundary crossings:** A single chunk could contain the tail of Article 8 and the beginning of Article 9, causing the retriever to return misleading context when queried about a specific article.
2. **Lost structural signal:** The flat splitter had no awareness of document structure, so the embedding for a chunk about "consent" in the GDPR looked nearly identical to one about "consent" in the AI Act — the document name only appeared as metadata, not in the chunk body.

These problems manifested as 74.2% average retrieval recall and 50% recall on the AI Act in baseline evaluation.

### 3.2 Article-Boundary Chunker (`_build_article_nodes`)

The custom chunker implemented in `src/rag_pipeline.py` operates on each `.txt` file independently:

**Step 1 — Article header detection**

```python
_ARTICLE_LINE_RE = re.compile(r"^Article[\s\xa0]+(\d+)[\s\xa0]*$")
```

Each line is matched against this pattern. A match marks the start of a new article. The article number is captured for metadata.

**Step 2 — Preamble handling**

Content before the first `Article N` line (recitals, whereas clauses, definitions preamble) is extracted as a separate block and split using `SentenceSplitter(chunk_size=512, chunk_overlap=64)`. Each preamble sub-chunk is tagged with `chunk_type: "preamble"`.

**Step 3 — Article chunk construction**

For each article, the title is extracted from the third line of the article block (line index 2, since EUR-Lex formats articles as `Article N\n\n<Title>`). If the third line starts with a digit or parenthesis (i.e., it is body text rather than a title), the title is treated as absent.

Each article chunk is prefixed with two structural elements:

- **Header:** `[{doc_name} | Article {N}] — {Title}\n\n`
  This provides a machine-readable tag for citation extraction and human-readable context.

- **Identity sentence:** `The following is Article {N} of the {doc_name}, titled '{Title}'.\n\n`
  This critical addition embeds the document name and article number as natural language *within the chunk body*, ensuring the embedding model encodes the document provenance into the dense vector — not just as metadata. Without this, `bge-small-en-v1.5` frequently returned cross-document false positives for thematically similar articles (e.g., GDPR Article 5 vs. Data Act Article 5).

**Step 4 — Sub-chunking for long articles**

Articles whose full text exceeds `sub_chunk_tokens * 4` characters (default: 512 × 4 = 2,048 chars, a rough chars-per-token estimate) are split further. The splitter first attempts to break on top-level numbered paragraphs:

```python
_PARA_RE = re.compile(r"^\d+\.[\s\xa0]")
```

If two or more paragraph breaks are found, each paragraph becomes its own chunk. If a single paragraph still exceeds the budget, `SentenceSplitter` is applied as a fallback. In all sub-chunk cases, the header and identity sentence are prepended to every chunk so that no sub-chunk loses document provenance.

**Step 5 — Atomic chunking for definitions articles (GQ-04)**

Articles whose title matches `/^Definitions?$/i` receive special treatment. Instead of paragraph-level sub-chunking, each numbered definition entry is extracted as its own `TextNode` using the pattern `r'^\d+\.\s+"'` (numbered quoted terms). This ensures that a definitions article with 20+ entries (such as Data Act Article 2 or DGA Article 2) never splits a single definition across chunk boundaries — which was the root cause of hallucinated absences for definition lookup queries. If the entry-by-entry pattern yields fewer than two matches, the article falls back to the standard sub-chunking path.

**Step 6 — Metadata**

Every `TextNode` carries:

```python
{
    "document":      "GDPR",          # display name of the regulation
    "article":       "83",            # article number (string)
    "chunk_articles": "83",           # used by retrieval_recall metric
    "chunk_type":    "article",       # or "preamble"
    "chunk_part":    2,               # present on sub-chunks only
}
```

**Output:** 1,685 `TextNode` objects across all five documents.

### 3.3 Impact of Article-Boundary Chunking

| Metric | Flat SentenceSplitter | Article-Boundary |
|--------|-----------------------|------------------|
| AI Act retrieval recall | 50.0% | 100.0% |
| Avg retrieval recall | 74.2% | 100.0% (after full pipeline) |
| Avg latency | 14.1s | 6.7s |

The latency improvement is a side effect: smaller, more targeted chunks give the LLM less text to process, reducing generation time.

---

## 4. Embedding Model

**Model:** `BAAI/bge-small-en-v1.5`
**Parameters:** 33M
**Max sequence length:** 512 tokens
**Loaded via:** `llama-index-embeddings-huggingface`

`bge-small-en-v1.5` was chosen as a balance between embedding quality and hardware constraints. It fits comfortably in CPU/GPU memory alongside the quantized LLM and produces embeddings of dimension 384. On a legal corpus, `bge` family models consistently outperform general-purpose models (e.g., `all-MiniLM`) on domain-specific retrieval benchmarks.

The 512-token limit is not a practical concern for this corpus: after article-boundary chunking, the vast majority of chunks are well under 512 tokens. The identity sentence + header contribute roughly 20–30 tokens of overhead.

Embeddings are generated at index build time and persisted to `data/rag_index/` via LlamaIndex's `StorageContext`. Subsequent runs load from disk, skipping re-embedding.

---

## 5. Retrieval Architecture

### 5.1 Baseline: Dense-Only Retrieval

The initial implementation used a single `VectorStoreIndex.as_retriever(similarity_top_k=K)` for dense cosine similarity search. This worked well for semantic queries but failed on exact-term disambiguation — for example, a question about the "Data Governance Act" could return chunks from the "Data Act" because the two documents are thematically similar and the embedding model conflated them.

### 5.2 Hybrid Retrieval (BM25 + Dense)

The production implementation uses `QueryFusionRetriever` to combine two independent retrievers:

**Dense retriever:** Standard `VectorStoreIndex.as_retriever(similarity_top_k=10)`
**BM25 retriever:** `BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=10)` from `llama-index-retrievers-bm25` (backed by `bm25s` and `pystemmer`)

**Fusion method:** Reciprocal Rank Fusion (RRF)

RRF computes a combined score for each document as:

```
RRF_score(d) = Σ_r  1 / (k + rank_r(d))
```

where `k = 60` (default), and `rank_r(d)` is the rank of document `d` in retriever `r`. Documents that rank highly in both retrievers receive a strong boost; documents that appear in only one list are penalised but not discarded.

**Configuration:**

```python
hybrid_retriever = QueryFusionRetriever(
    retrievers=[dense_retriever, bm25_retriever],
    similarity_top_k=10,
    num_queries=1,       # original query only, no query rewriting
    mode="reciprocal_rerank",
    use_async=False,
)
```

`num_queries=1` disables LlamaIndex's query rewriting feature, which would generate multiple phrasings of the user's question and issue separate retrieval calls. For a structured legal corpus with precise terminology, query rewriting adds noise rather than signal.

**Why hybrid retrieval matters for legal text:**

Legal queries frequently contain regulation-specific terminology ("Data Governance Act", "Article 11", "notified body") that dense embeddings can fail to distinguish from semantically similar terms in other regulations. BM25 exact-term matching provides a direct lexical signal that anchors retrieval to the correct document even when the dense vector space is ambiguous. The DGA recall improvement from 14.3% to 100% was the clearest demonstration of this: BM25 correctly retrieved DGA Article 11 on the first pass where dense-only retrieval was returning GDPR and Data Act chunks.

### 5.3 Cross-Encoder Re-Ranker (RQ-02)

An optional cross-encoder re-ranking stage was added after the hybrid retriever. When enabled, `SentenceTransformerRerank` from `llama_index.core.postprocessor` applies `BAAI/bge-reranker-base` to the top-K candidates and retains the top-N for synthesis.

**Configuration:**

```python
create_legal_rag_tool(
    reranker=True,       # enable re-ranker
    reranker_top_n=8,    # retain top-8 after re-ranking
)
```

**Pipeline with re-ranker:**

```
Hybrid retriever → top-10 candidates
    → BGE cross-encoder re-scores all 10
    → top-8 pass to synthesis
```

Cross-encoders jointly encode query and document, capturing relevance cues that bi-encoder embeddings miss (e.g., precise term co-occurrence at specific positions). For legal retrieval, this is particularly valuable when multiple regulations share similar terminology — the cross-encoder can distinguish "Data Act Article 2" from "GDPR Article 2" more reliably than a dense retriever operating on independently encoded representations.

**Evaluation note:** In the current ablation (top-8 re-ranker), the re-ranker did not improve aggregate judge scores (1.79 vs 1.92 baseline) despite reducing latency from 5.7s to 5.1s. The primary known failure (`ai_act_social_scoring_prohibition`) is structurally unaffected — 8 of 10 retrieved chunks are preamble regardless of re-ranking, since both BM25 and dense signals favour preamble recitals for social-scoring queries. A re-ranker trained specifically to demote preamble chunks when normative articles are present would be needed to address this class of failure.

---

## 6. Language Model

**Model:** `Qwen/Qwen2.5-7B-Instruct`
**Quantization:** 4-bit NF4 via BitsAndBytes
**Compute dtype:** `float16`
**Device map:** `auto` (GPU primary, CPU offload for overflow)
**Memory budget:** GPU 8 GiB, CPU 20 GiB

### 6.1 Quantization Configuration

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    llm_int8_enable_fp32_cpu_offload=True,
)
```

NF4 (NormalFloat 4-bit) is the quantization scheme from QLoRA. Double quantization further compresses the quantization constants, saving an additional ~0.4 bits per parameter. The compute dtype is kept at `float16` so that matrix multiplications happen in half precision (not 4-bit), preserving numerical stability.

### 6.2 LlamaIndex Integration

Qwen is wrapped in a custom `QwenLLM(CustomLLM)` class to integrate with LlamaIndex's `Settings.llm`. The wrapper:

- Uses HuggingFace `pipeline("text-generation")` under the hood
- Applies Qwen's ChatML template via `tokenizer.apply_chat_template()` before passing prompts
- Sets `return_full_text=False` so the pipeline returns only the generated portion
- Implements both `complete()` (synchronous) and `stream_complete()` (streaming via `TextIteratorStreamer`) to support both batch eval and the Chainlit UI

**Generation parameters:**

```python
max_new_tokens=512
do_sample=False   # greedy decoding — deterministic, appropriate for factual legal Q&A
```

### 6.3 Context Window

`CONTEXT_WINDOW = 32768`. Profiling (AOC-04, RQ-01) confirmed that with `top_k=10` and the current chunk distribution (mean ~303 tokens, p90 ~677 tokens), worst-case prompt size is approximately 7,800 tokens — well within the 32,768 budget with zero truncation. A runtime warning in `QwenLLM.complete()` fires if any single call exceeds 80% of the budget (~25,800 tokens) as a future safety net.

The earlier setting of `CONTEXT_WINDOW = 4096` was responsible for the `data_act_data_holder_definition` failure: the definition chunk was silently truncated before reaching the model. This was resolved by expanding the constant.

### 6.4 Known Warning

Qwen2.5's bundled `generation_config.json` sets `max_length=20`. This conflicts with the `max_new_tokens=512` passed at inference time, causing a HuggingFace warning on every call. The fix `model.generation_config.max_length = None` suppresses the warning in most paths, but the HuggingFace pipeline internally re-merges configs before calling `model.generate()`, so the warning reappears. The warning is cosmetic — `max_new_tokens` always takes precedence — but remains an open issue.

---

## 7. Prompt Engineering

The QA prompt is a LlamaIndex `PromptTemplate` (`LEGAL_QA_PROMPT`) injected via `query_engine.update_prompts()`. It has been revised iteratively as failures were diagnosed.

### 7.1 Current Prompt

```
You are an EU legal expert assistant. Answer using ONLY the context provided.

RULES:
0. If the context does not contain information relevant to the question, respond ONLY with:
   This question is outside the scope of the available EU regulations.
1. Only cite Article or Recital numbers that explicitly appear in the context below.
2. Do NOT use training knowledge to add article numbers absent from the context.
3. If unsure, say 'The retrieved documents indicate...' without citing specific numbers.
4. When both an Article and a Recital cover the same topic, cite the Article as the
   authoritative source and the Recital only as supplementary context.

[chain-of-thought instruction for tiered numerical queries]

Context:
---------------------
{context_str}
---------------------

Question: {query_str}
Answer:
```

### 7.2 Design Rationale by Rule

**Rule 0 — Hard OOS refusal (GQ-03):** When context is irrelevant, the model must respond with an exact fixed phrase rather than attempting a partial answer. This was added after baseline measurement showed a 30% OOS refusal rate: 7 of 10 OOS queries produced partial answers using loosely retrieved context rather than clean refusals. The exact fixed phrase (`This question is outside the scope of the available EU regulations.`) is matched by the `_OOS_REFUSAL_RE` pattern in `eval_runner.py`.

**Rule 1** is the primary anti-hallucination guard for citations. Legal answers are particularly prone to phantom citations because the LLM was trained on legal text and has strong priors about article numbers.

**Rule 2** reinforces Rule 1 by explicitly naming the failure mode (using training knowledge) rather than just saying "don't make things up".

**Rule 3** provides a safe fallback that still lets the model give a useful answer when context is present but citation-specific claims would be unverifiable.

**Rule 4 — Article over recital preference (GQ-02):** Addresses the `ai_act_social_scoring_prohibition` class of failure where the LLM cites a preamble recital rather than the normative article. Both Recital 31 and Article 5 are typically retrieved for social-scoring queries, but preamble chunks rank higher in RRF fusion due to their denser descriptive language. This rule instructs the model to use the article as the authoritative source when both types are present.

**Chain-of-thought for tiered numerical facts (GQ-01):** For queries involving tiered penalty or threshold structures (e.g., AI Act Article 99 with 7%/3%/1.5% fine tiers), the model is instructed to first identify which tier applies to the specific scenario described in the question before stating the figure. This prevents the context-confusion failure observed in early evaluation where the 3% tier (most frequently repeated across the context window) was substituted for the correct 7% tier.

---

## 8. Citation Verification

`verify_citations()` in `src/rag_pipeline.py` detects phantom citations — article or recital numbers mentioned in the answer that do not appear in any retrieved chunk:

```python
cited_articles  = set(re.findall(r'[Aa]rticle\s+(\d+)', response_text))
cited_recitals  = set(re.findall(r'[Rr]ecital\s+(\d+)', response_text))

available_articles = set()  # union of all article numbers mentioned in retrieved chunks
available_recitals = set()

phantom_articles = cited_articles - available_articles
phantom_recitals = cited_recitals - available_recitals
```

A response is considered citation-accurate (`is_accurate=True`) if and only if both phantom sets are empty.

**Limitation:** This is a conservative check — it flags any article number the LLM cites that doesn't appear textually in the retrieved chunks. It does not verify that the cited article actually supports the claim made. A more robust check would require grounding verification, which is partially addressed by the LLM-as-judge metric.

Cross-reference false positives remain an edge case: when a retrieved chunk for Article 20 internally references Articles 6 and 9 (e.g. GDPR portability conditions), the verifier's available set includes those numbers, so citations to them pass — even when the query was only about Article 20.

---

## 9. Evaluation Framework

The evaluation pipeline is implemented in `src/eval_runner.py`. It runs every entry in the golden dataset through the full RAG pipeline and computes metrics per query. Experiment results are tracked in MLflow (IN-02).

### 9.1 Golden Dataset

**File:** `data/eval_dataset.json`
**Total entries:** 48 (38 in-scope, 10 out-of-scope)
**Coverage:** All five EU regulations, plus cross-document and negative questions

| Regulation | In-scope questions |
|------------|--------------------|
| GDPR | 10 |
| AI Act | 6 |
| Data Act | 6 |
| Data Governance Act | 7 |
| Cyber Resilience Act | 7 |
| Cross-document | 2 |

Each entry contains:

```json
{
  "id": "gdpr_breach_notification",
  "query": "How many hours does a controller have...",
  "expected_answer": "72 hours after becoming aware of the breach.",
  "expected_articles": ["33"],
  "expected_document": "GDPR",
  "out_of_scope": false
}
```

**Dataset construction history:**

The original 33-entry dataset (31 in-scope, 2 OOS) covered only single-article factual lookups. It was expanded in two stages:

1. *Structural corrections (pre-AOC-00):* 6 DGA entries contained systematic errors where paragraph-level numbers within articles were misidentified as article numbers (e.g., "paragraph 12 of Article 11" recorded as `expected_articles: ["12"]`). Corrected by programmatic verification against the actual text. The corrected article numbers are 11, 11, 14, 14, 19, and 24.

2. *Dataset expansion (AOC-00):* 15 new entries added — 8 in-scope (multi-hop within a document, cross-document synthesis, negative scope questions) and 8 OOS entries covering topics absent from the corpus: ePrivacy Directive, MiFID II, NIS2 Directive, Digital Markets Act, Product Liability Directive, EU copyright law, consumer protection, and EU competition law (TFEU Article 102). Total OOS entries: 10.

The new question types expose failure modes invisible to single-article recall: multi-hop queries require the retriever to surface chunks from two articles simultaneously; cross-document queries require chunks from two different regulations; negative questions test whether the model correctly identifies the scope boundary of a regulation.

### 9.2 Metric 1 — Retrieval Recall

```python
retrieval_recall = (number of expected articles found in top-K chunks) / (number of expected articles)
```

**Document-scoped:** When `expected_document` is provided, only chunks whose `document` metadata matches are considered. This prevents a GDPR Article 5 chunk from counting as a hit for a DGA Article 5 question.

### 9.3 Metric 2 — Citation Accuracy

Binary pass/fail per query (see Section 8). Aggregated as the fraction of in-scope queries with zero phantom citations.

### 9.4 Metric 3 — Fact Completeness

Checks whether key numeric facts from `expected_answer` appear verbatim (after whitespace normalisation) in the model's response. Facts are extracted using three regex patterns:

| Pattern | Examples extracted |
|---------|-------------------|
| `EUR[\s\xa0][\d\s,]+` | `EUR 20 000 000`, `EUR 35 000 000` |
| `\d+(?:\.\d+)?[\s\xa0]*%` | `4 %`, `7 %`, `2.5%` |
| `\d+(?:\s+calendar)?\s+(?:hours?\|days?\|...)` | `72 hours`, `30 calendar days`, `5 years` |

Note: patterns use `[\s\xa0]` to handle EUR-Lex's Unicode non-breaking space thousand separators (GQ-11). A fallback numeric match strips all separators for comparison.

```
fact_score = (facts found in response) / (total facts in expected_answer)
```

Entries with no extractable numeric facts receive `fact_score = 1.0` by default.

### 9.5 Metric 4 — LLM-as-Judge

A second call to the same Qwen model evaluates each response against the expected answer:

```
SCORING CRITERIA:
  2 – Correct  : Response captures all key facts from the reference answer.
  1 – Partial  : Response captures some but not all key facts.
  0 – Incorrect: Response is wrong, irrelevant, or missing critical information.
```

The model responds with a single-line JSON object:

```json
{"score": 2, "reason": "Response correctly identifies 72 hours and Article 33."}
```

A regex extraction layer handles cases where the model wraps the JSON in prose. A digit-only fallback handles further degradation.

**Self-judge bias:** Using the same model as both the responder and the judge introduces a known bias. A more rigorous setup would use a separate, stronger judge model. The current approach is a practical compromise given on-premises constraints.

### 9.6 Metric 5 — OOS Refusal Rate (AOC-01)

For each OOS entry, `detect_oos_refusal(response_text)` returns `True` if the response contains a clear refusal signal:

```python
_OOS_REFUSAL_RE = re.compile(
    r"outside (the )?scope"
    r"|not covered"
    r"|not (contain|include|address|discuss|mention|found|available)\b"
    r"|cannot (find|answer|provide|determine)\b"
    r"|no (information|content|data) (found|available|provided|in the)"
    r"|not in the (provided|retrieved|available)\b",
    re.IGNORECASE,
)
```

Aggregated as `oos_refusal_rate = refused / total_oos`. A rate of 0% means the system attempts to answer every OOS query; 100% means it cleanly refuses all of them.

### 9.7 Metric 6 — Cross-Document Retrieval Recall (AOC-01)

For entries with multiple `(article, document)` pairs, each pair must be satisfied independently. A cross-doc entry requiring GDPR Art.33 and CRA Art.14 scores 0.5 if only one is retrieved — even if the overall article-number hit is 100%. Implemented by paired traversal of `expected_articles` and `expected_documents` lists.

### 9.8 Metric 7 — Source Attribution Accuracy (AOC-01)

Detects document-name phantom citations: cases where the model correctly states an article number but attributes it to the wrong regulation (e.g., "GDPR Article 5" when the answer required "AI Act Article 5"). Two regex patterns extract qualified `(document, article)` mentions from the response; each extracted pair is checked against the retrieved chunks' metadata.

### 9.9 Latency and Context Profiling (AOC-04)

Each eval entry also records:

- `prompt_tokens` — estimated tokens in the LLM prompt (accumulated across all LLM calls for the query)
- `retrieval_latency_s` — time spent in the retriever
- `gen_latency_s` — time spent inside the LLM pipeline
- `chunks_retrieved` / `chunks_in_prompt` — count before and after context assembly
- `truncated` / `dropped_chunks` — whether any retrieved chunks were silently dropped

`print_profiling_stats()` summarises mean/p90/max for all profiling metrics at the end of each eval run.

### 9.10 MLflow Experiment Tracking (IN-02)

Each eval run logs to an MLflow experiment (`EU Legal RAG Evaluation`). Parameters logged: `top_k`, `chunking_strategy`, `reranker`, `reranker_top_n`, `n_entries`, `n_in_scope`, `n_oos`. Metrics logged: `avg_retrieval_recall`, `cross_doc_recall`, `citation_accuracy`, `attribution_accuracy`, `avg_fact_score`, `avg_judge_score`, `oos_refusal_rate`, `avg_latency_s`, `avg_prompt_tokens`, `truncation_rate`. The full results JSON is attached as an artifact.

### 9.11 CLI Interface

```bash
python src/eval_runner.py                            # full eval with judge
python src/eval_runner.py --no-judge                 # skip judge, faster
python src/eval_runner.py --top-k 8                 # adjust retrieval K
python src/eval_runner.py --reranker                 # enable re-ranker (top-n=5 default)
python src/eval_runner.py --reranker --reranker-top-n 8
python src/eval_runner.py --out results.json         # custom output path
```

Results are saved as a JSON array to `data/eval_results_<timestamp>.json`, with one object per dataset entry containing all raw scores and the model's response text.

---

## 10. Results & Analysis

### 10.1 Iterative Improvement History (original pipeline)

| Run | Key Change | Recall | Citation | Notes |
|-----|-----------|--------|----------|-------|
| 1 – Baseline | Flat `SentenceSplitter`, `top_k=5` | 74.2% | 87.1% | AI Act 50%, 14.1s latency |
| 2 – Article chunking | `_build_article_nodes` | 64.5% | — | AI Act 100%, but DGA 14.3% (wrong dataset article numbers) |
| 3 – Fix metric | Document-scoped `retrieval_recall` | 77.4% | — | Cross-doc false positives eliminated |
| 4 – Identity sentences | Document name in chunk body | 71.0% | 96.8% | Citation improved; DGA still broken |
| 5 – BM25 hybrid + dataset fix | `QueryFusionRetriever` + corrected DGA entries | **100.0%** | **100.0%** | DGA 100%, 6.7s latency |

### 10.2 Baseline After Expansion (AOC-03)

After expanding the dataset from 33 to 48 entries (harder question types), a fresh baseline was established:

```
=== Baseline (top_k=10, no re-ranker) ===
Total entries         : 48
In-scope              : 38  (cross-document: 2)
Out-of-scope          : 10
Avg retrieval recall  : 90.8%
Cross-doc recall      : 50.0%
Citation accuracy     : 86.8%  (33/38 clean)
Source attribution    : 97.4%  (37/38)
Avg fact completeness : 98.7%
Avg judge score       : 1.92/2  (35 correct / 3 partial / 0 wrong)
OOS refusal rate      : 30.0%  (3/10 correctly refused)
Avg latency           : 5.7s
```

The drop from 100% recall (on the original 33-entry set) to 90.8% is entirely attributable to the new harder entries: 5 retrieval failures appeared, all involving multi-hop, cross-document, or negative-scope questions.

**Per-document breakdown:**

| Document | n | Recall | Judge |
|----------|---|--------|-------|
| AI Act | 6 | 100.0% | 1.83 |
| Cyber Resilience Act | 7 | 100.0% | 2.00 |
| Data Act | 6 | 100.0% | 2.00 |
| Data Governance Act | 7 | 85.7% | 2.00 |
| GDPR | 10 | 85.0% | 1.90 |
| Cross-document | 2 | 50.0% | 1.50 |

### 10.3 Re-Ranker Ablation (RQ-02)

| Configuration | Recall | Citation | Judge | OOS | Latency |
|---------------|--------|----------|-------|-----|---------|
| Baseline (no reranker) | 90.8% | 86.8% | 1.92/2 | 30% | 5.7s |
| Reranker top-8 | 90.8% | 81.6% | 1.79/2 | 40% | 5.1s |

The re-ranker did not improve recall or judge scores at top-8. The slight decrease in citation accuracy and judge score is likely due to the re-ranker demoting some relevant chunks below the top-8 cutoff. The OOS refusal rate improved marginally (30% → 40%), possibly because stricter top-N filtering reduces the volume of loosely relevant context that enables partial answers. Latency improved by 0.6s due to fewer tokens being synthesised.

---

## 11. Known Failures & Root Cause Analysis

All retrieval recalls for failing queries are 1.0 — the correct chunks are being retrieved in every case. The failures are attributable to generation behaviour, retrieval ranking, or query type.

### 11.1 `ai_act_social_scoring_prohibition` — Judge score 1 (partial) *(open)*

**Query:** *Does the AI Act prohibit AI systems used by public or private actors for social scoring...?*
**Expected:** `Yes. Article 5(1)(c) prohibits...`
**Response excerpt:** *"...This prohibition is explicitly stated in the preamble (Article 31) of the AI Act."*

**Root cause:** 8 of 10 retrieved chunks are preamble recitals; Article 5 is at ranks 3 and 7. The BM25 and dense retrievers both favour preamble text because recitals use rich descriptive language about social scoring, while normative Article 5(1)(c) is buried in a list of 8+ prohibited practices. Additionally, the sub-paragraph notation `5(1)(c)` does not appear verbatim in any retrieved chunk — the relevant sub-article is in a part not retrieved.

**Applied mitigations:** Rule 4 added to prompt (GQ-02) — instructs the model to prefer normative articles over recitals when both are present. **Not yet re-evaluated** — a new eval run is needed to measure impact.

**Remaining gap:** The prompt fix addresses generation but not retrieval ranking. A retrieval-side fix (e.g., a post-processor that demotes preamble chunks when normative articles for the same topic are present) would be more robust.

### 11.2 `negative_gdpr_household_exemption` — Judge score 0 *(open)*

**Query:** *Does the GDPR apply to data processing carried out by a natural person in the course of purely personal or household activity?*
**Expected:** No — Article 2(2)(c) explicitly excludes it.
**Response:** The model suggests the GDPR may apply, contradicting the reference answer.

**Root cause:** Negative-scope questions (does a regulation NOT apply?) are a systematically harder retrieval case. The query activates chunks about GDPR applicability (Article 2, Article 3) but the exclusion sub-paragraph `2(2)(c)` may not appear prominently in the retrieved text. The model defaults to affirming applicability because the bulk of retrieved context describes when the GDPR applies rather than when it does not.

### 11.3 `cra_vulnerability_dual_notification` — Judge score 0 *(open)*

**Query:** *What are the specific notification deadlines for actively exploited vulnerabilities under the CRA?*
**Expected:** 24-hour early warning + 72-hour full notification.
**Response:** Missing or incorrect deadlines.

**Root cause:** Dual-deadline tiered structures (similar to the AI Act fine tiers) are prone to the same context-confusion failure pattern. The 24h/72h split may appear across multiple chunks or be retrievable only partially.

### 11.4 `dga_data_altruism_definition` — Judge score 1 (partial) *(open)*

**Query:** *How does the DGA define data altruism?*
**Expected:** Full definition from Article 2.
**Response:** Partial definition.

**Root cause:** Same definitions-article density problem as `data_act_data_holder_definition` (resolved earlier via GQ-04). The DGA Article 2 definitions entry for "data altruism" may still be split across atomic chunks, or the definition may lack sufficient lexical signal to rank the correct sub-chunk first.

### 11.5 `gdpr_data_minimisation_and_fine` — Judge score 1 (partial) *(open)*

**Query:** Multi-hop: requires combining Article 5 (data minimisation principle) and Article 83 (penalty for violation).
**Root cause:** Cross-article retrieval gap. The hybrid retriever must surface chunks from both Article 5 and Article 83 simultaneously. At `top_k=10` one of the two is occasionally missed, resulting in an incomplete answer.

### 11.6 Out-of-scope refusal rate: 40% (4/10) *(partially mitigated)*

6 of 10 OOS queries still receive partial answers using loosely retrieved context. The hard refusal Rule 0 (GQ-03) was added to the prompt but has not been re-evaluated. The baseline was 30% (3/10); re-ranker top-8 showed 40% (4/10), which predates the GQ-03 change.

---

## 12. Limitations & Future Work

### 12.1 Current Limitations

**Evaluation dataset size:** 48 entries is sufficient to identify systematic failure modes but too small for statistically stable per-document scores. Single question failures shift a 6–7 entry document score by ~14–17%.

**Preamble retrieval dominance:** For queries whose natural-language form closely matches preamble recital text (which is descriptive and explanatory), the RRF fusion consistently ranks preamble chunks above normative articles. This is a retrieval architecture limitation — neither BM25 nor dense vectors penalise `chunk_type: "preamble"`. A simple metadata filter or score penalty for preamble chunks in normative-query contexts would address this.

**Cross-document recall at 50%:** Queries that require retrieving chunks from two different regulations simultaneously are harder for the hybrid retriever. At `top_k=10`, there is a fixed budget shared across all documents, and one regulation's chunks may crowd out the other. Larger `top_k` or per-document sub-retrievers would help.

**OOS handling:** Despite the hard-refusal instruction (GQ-03), a new eval run is needed to confirm its effect. The fundamental challenge is that the retriever will always return *something* — the model must then decide whether that context is relevant to the question. Without a query classifier or retrieval confidence threshold, the model relies entirely on the prompt instruction.

**Self-judge bias:** The LLM-as-judge uses the same model as the responder. This likely inflates scores for partially correct responses and produces inconsistent scoring on borderline cases.

**Static corpus:** The document texts are static snapshots. The pipeline has no mechanism for detecting or incorporating regulatory amendments.

### 12.2 Open Improvements

| Priority | Improvement | Target failure |
|----------|------------|----------------|
| High | New eval run after GQ-02 + GQ-03 prompt changes | `ai_act_social_scoring_prohibition`, OOS refusal |
| High | Preamble chunk demotion post-processor | `ai_act_social_scoring_prohibition` class |
| High | Chain-of-thought tiered deadline handling | `cra_vulnerability_dual_notification` |
| Medium | Negative-scope query detection | `negative_gdpr_household_exemption` class |
| Medium | Per-document sub-retrievers or larger `top_k` for cross-doc entries | Cross-doc recall 50% |
| Medium | Use a stronger judge model (larger Qwen or API-based) | Self-judge bias |
| Medium | HierarchicalNodeParser + AutoMergingRetriever | Definitions-article partial misses |
| Low | Add article titles from EUR-Lex to source `.txt` files | Richer chunk headers |

---

*Report reflects evaluation runs through `data/eval_results_20260315_reranker_top8.json`. Prompt changes from GQ-02 and GQ-03 are implemented but not yet reflected in a new eval run.*
