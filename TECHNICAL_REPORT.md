# Technical Report: EU Legal Documents RAG Pipeline

**Project:** EU Legal Q&A Agent
**Stack:** LlamaIndex 0.12 · Qwen2.5-7B-Instruct (4-bit) · BGE-small-en-v1.5 · BM25 Hybrid Retrieval
**Date:** March 2026

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

**Preprocessing notes:** EUR-Lex PDFs use Unicode non-breaking spaces (`\xa0`) heavily — in article headers, paragraph numbering, and mid-sentence line wraps. All regex patterns used for chunking explicitly account for `[\s\xa0]` rather than `\s` alone to avoid missed matches.

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

**Step 5 — Metadata**

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

### 6.3 Known Warning

Qwen2.5's bundled `generation_config.json` sets `max_length=20`. This conflicts with the `max_new_tokens=512` passed at inference time, causing a HuggingFace warning on every call. The fix `model.generation_config.max_length = None` suppresses the warning in most paths, but the HuggingFace pipeline internally re-merges configs before calling `model.generate()`, so the warning reappears. The warning is cosmetic — `max_new_tokens` always takes precedence — but remains an open issue.

---

## 7. Prompt Engineering

The QA prompt is a LlamaIndex `PromptTemplate` injected via `query_engine.update_prompts()`:

```
You are an EU legal expert assistant. Answer using ONLY the context provided.

RULES:
1. Only cite Article or Recital numbers that explicitly appear in the context below.
2. Do NOT use training knowledge to add article numbers absent from the context.
3. If unsure, say 'The retrieved documents indicate...' without citing specific numbers.

Context:
---------------------
{context_str}
---------------------

Question: {query_str}
Answer:
```

**Design rationale:**

- **Rule 1** is the primary anti-hallucination guard for citations. Legal answers are particularly prone to phantom citations because the LLM was trained on legal text and has strong priors about article numbers.
- **Rule 2** reinforces Rule 1 by explicitly naming the failure mode (using training knowledge) rather than just saying "don't make things up".
- **Rule 3** provides a safe fallback that still lets the model give a useful answer when context is present but citation-specific claims would be unverifiable.

The same prompt template is reused for the LLM-as-judge call (see Section 9.4), with different content injected into `query_str`.

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

---

## 9. Evaluation Framework

The evaluation pipeline is implemented in `src/eval_runner.py`. It runs every entry in the golden dataset through the full RAG pipeline and computes four metrics per query.

### 9.1 Golden Dataset

**File:** `data/eval_dataset.json`
**Total entries:** 33 (31 in-scope, 2 out-of-scope)
**Coverage:** All five EU regulations

| Regulation | In-scope questions |
|------------|--------------------|
| GDPR | 6 |
| AI Act | 6 |
| Data Act | 6 |
| Data Governance Act | 7 |
| Cyber Resilience Act | 6 |

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

The 2 out-of-scope entries cover topics not in the corpus (ePrivacy Directive, EU financial market regulation). Their responses are recorded for manual review only.

**Dataset construction note:** The initial dataset was generated by a research agent (Claude) and contained systematic errors in 6 DGA entries where paragraph-level numbers within articles were misidentified as article numbers (e.g., "paragraph 12 of Article 11" was recorded as `expected_articles: ["12"]`). These were corrected by programmatic verification against the actual text, cross-referencing article boundary positions in `data_governance_act.txt`. The corrected article numbers are 11, 11, 14, 14, 19, and 24.

### 9.2 Metric 1 — Retrieval Recall

```python
retrieval_recall = (number of expected articles found in top-K chunks) / (number of expected articles)
```

**Document-scoped:** When `expected_document` is provided, only chunks whose `document` metadata matches are considered. This prevents a GDPR Article 5 chunk from counting as a hit for a DGA Article 5 question — a real failure mode observed in early evaluation when the metric was document-agnostic.

A score of 1.0 means all articles that should have been retrieved were in the top-K results. A score of 0.0 means none were.

### 9.3 Metric 2 — Citation Accuracy

Binary pass/fail per query (see Section 8). Aggregated as the fraction of in-scope queries with zero phantom citations.

### 9.4 Metric 3 — Fact Completeness

Checks whether key numeric facts from `expected_answer` appear verbatim (after whitespace normalisation) in the model's response. Facts are extracted using three regex patterns:

| Pattern | Examples extracted |
|---------|-------------------|
| `EUR[\s\xa0][\d,]+` | `EUR 20,000,000`, `EUR 35,000,000` |
| `\d+(?:\.\d+)?\s*%` | `4%`, `7%`, `2.5%` |
| `\d+(?:\s+calendar)?\s+(?:hours?\|days?\|...)` | `72 hours`, `30 calendar days`, `5 years` |

A fallback numeric match (stripping commas and spaces) handles formatting differences between the expected answer and the model's response (e.g., `20,000,000` vs `20000000`).

```
fact_score = (facts found in response) / (total facts in expected_answer)
```

Entries with no extractable numeric facts receive `fact_score = 1.0` by default (they are not penalised).

**Scope and limitations:** This metric is intentionally narrow — it only checks numeric/monetary/temporal facts, not qualitative claims. It is fast, deterministic, and catches the most critical errors in legal Q&A (wrong penalty amounts, wrong time limits). Non-numeric definitional answers (e.g., "data altruism") score 1.0 regardless of response quality, which is why the LLM-as-judge metric is necessary as a complement.

### 9.5 Metric 4 — LLM-as-Judge

A second call to the same Qwen model (reusing the already-loaded `Settings.llm` instance) evaluates each response against the expected answer with a structured scoring prompt:

```
SCORING CRITERIA:
  2 – Correct  : Response captures all key facts from the reference answer.
  1 – Partial  : Response captures some but not all key facts.
  0 – Incorrect: Response is wrong, irrelevant, or missing critical information.
```

The model is instructed to respond with a single-line JSON object:

```json
{"score": 2, "reason": "Response correctly identifies 72 hours and Article 33."}
```

A regex extraction layer (`re.search(r'\{.*?"score"\s*:\s*([012]).*?\}', raw, re.DOTALL)`) handles cases where the model wraps the JSON in prose. A digit-only fallback handles further degradation.

**Self-judge bias:** Using the same model as both the responder and the judge introduces a known bias — models tend to rate their own outputs more favourably. This is partially mitigated by the structured prompt and explicit scoring criteria, but a more rigorous setup would use a separate, stronger judge model (e.g., `Qwen2.5-72B` or a frontier model via API). The current approach is a practical compromise given on-premises constraints.

**Efficiency:** The judge call reuses the loaded model weights with no reloading overhead. Each judge call adds roughly 2–4 seconds to per-query latency (not included in the reported `latency_s`, which measures only the RAG query).

### 9.6 CLI Interface

```bash
python src/eval_runner.py                          # full eval with judge
python src/eval_runner.py --no-judge               # skip judge, faster
python src/eval_runner.py --top-k 8               # adjust retrieval K
python src/eval_runner.py --out results.json       # custom output path
```

Results are saved as a JSON array to `data/eval_results_<timestamp>.json`, with one object per dataset entry containing all raw scores and the model's response text.

---

## 10. Results & Analysis

### 10.1 Iterative Improvement History

| Run | Key Change | Recall | Citation | Notes |
|-----|-----------|--------|----------|-------|
| 1 – Baseline | Flat `SentenceSplitter`, `top_k=5` | 74.2% | 87.1% | AI Act 50%, 14.1s latency |
| 2 – Article chunking | `_build_article_nodes` | 64.5% | — | AI Act 100%, but DGA 14.3% (wrong dataset article numbers) |
| 3 – Fix metric | Document-scoped `retrieval_recall` | 77.4% | — | Cross-doc false positives eliminated |
| 4 – Identity sentences | Document name in chunk body | 71.0% | 96.8% | Citation improved; DGA still broken |
| 5 – BM25 hybrid + dataset fix | `QueryFusionRetriever` + corrected DGA entries | **100.0%** | **100.0%** | DGA 100%, 6.7s latency |

### 10.2 Final Evaluation Results

Evaluated with `top_k=10`, `run_judge=True`, on 33 dataset entries (31 in-scope).

```
==================================================
EVAL SUMMARY
==================================================
Total entries         : 33
In-scope              : 31
Out-of-scope          : 2
Avg retrieval recall  : 100.0%
Citation accuracy     : 100.0%  (31/31 clean)
Avg fact completeness : 98.4%   (30/31 perfect)
Avg judge score       : 1.84/2  (28 correct / 1 partial / 2 wrong)
Avg latency           : 6.6s
```

**Per-document breakdown:**

| Document | Recall | Fact | Judge |
|----------|--------|------|-------|
| AI Act | 100.0% | 91.7% | 1.50 |
| Cyber Resilience Act | 100.0% | 100.0% | 2.00 |
| Data Act | 100.0% | 100.0% | 1.67 |
| Data Governance Act | 100.0% | 100.0% | 2.00 |
| GDPR | 100.0% | 100.0% | 2.00 |

The AI Act is the only document with degraded answer quality despite perfect retrieval, pointing to a generation-side problem rather than a retrieval problem.

---

## 11. Known Failures & Root Cause Analysis

Three queries scored below the maximum on quality metrics. All had retrieval recall = 1.0 — the correct chunks were retrieved in every case. The failures are therefore attributable to the LLM's generation behaviour.

### 11.1 `ai_act_max_fine_prohibited` — Fact score 0.5, Judge score 0

**Query:** *What is the maximum administrative fine for non-compliance with the prohibited AI practices listed in Article 5 of the AI Act?*
**Expected:** `Up to EUR 35,000,000 or, for an undertaking, up to 7% of total worldwide annual turnover, whichever is higher.`
**Response excerpt:** *"...the fine can be up to EUR 35,000,000 or 3% of their total worldwide annual turnover..."*

**Root cause:** AI Act Article 99 defines a tiered fine structure: 7% for prohibited practices (Article 5), 3% for other obligations, 1.5% for incorrect information. The retrieved context contained all three tiers. The LLM conflated the tiers and substituted the 3% threshold (the most frequently mentioned percentage across the context window) for the correct 7%. This is a **context confusion** failure: the model correctly retrieved the relevant article but failed to select the right value from a dense, numerically similar list.

**Mitigation options:** Reduce `top_k` to return fewer competing chunks; use a re-ranker to surface the most relevant paragraph first; or add a chain-of-thought instruction to the prompt requiring the model to identify which tier applies before stating the fine.

### 11.2 `ai_act_social_scoring_prohibition` — Judge score 1

**Query:** *Does the AI Act prohibit AI systems used by public or private actors for social scoring...?*
**Expected:** `Yes. Article 5(1)(c) prohibits...`
**Response excerpt:** *"...This prohibition is reflected in the Preamble, specifically in paragraph (31)..."*

**Root cause:** The response correctly identifies the prohibition but attributes it to the Preamble rather than Article 5(1)(c). This is a **source attribution error**: the retrieved chunks included both the Preamble recitals (which discuss the rationale for the prohibition) and Article 5 (which enacts it). The LLM cited the recital, which was likely ranked higher in the fused retrieval list, rather than the normative article. The system prompt rule *"Only cite Article or Recital numbers that explicitly appear in the context below"* was followed — Recital 31 was present — but the model did not distinguish between normative and explanatory text.

**Mitigation options:** Add a prompt instruction to prefer article-level citations over recital-level citations when both are available; filter preamble chunks from the retrieval pool for questions with known normative targets.

### 11.3 `data_act_data_holder_definition` — Judge score 0

**Query:** *How does the Data Act define 'data holder'?*
**Expected:** Full legal definition from Article 2.
**Response:** *"The retrieved documents indicate that the term 'data holder' is not explicitly defined in the provided context."*

**Root cause:** The model incorrectly claimed the definition was not in the context, despite the correct chunk being retrieved. This is a **hallucinated absence** — a failure mode where the model incorrectly asserts that information is missing. It likely occurs because the retrieved chunk contains many definitions in sequence (Article 2 of the Data Act is a definitions article with ~20 entries), and the model failed to locate the specific term within the dense list. The 512-token sub-chunk budget may have caused the definition to be split across sub-chunks, with `data holder` appearing at the boundary.

**Mitigation options:** Increase sub-chunk overlap for definition articles; add a pre-processing step that identifies and preserves complete definition entries as atomic units.

---

## 12. Limitations & Future Work

### 12.1 Current Limitations

**Evaluation dataset size:** 33 entries is sufficient to identify systematic failure modes but too small for statistically stable per-document scores. A single question failure changes a document's score by ~17%. The dataset also covers only factual lookup questions — no multi-hop reasoning, no cross-document synthesis, no negative questions.

**Out-of-scope handling:** With only 2 OOS entries, the system's ability to decline out-of-scope questions cannot be measured quantitatively. The current responses to OOS queries suggest the model is partially answering rather than cleanly refusing — a behaviour driven by the prompt's "Answer using ONLY the context" instruction, which doesn't explicitly instruct refusal when context is irrelevant.

**Self-judge bias:** The LLM-as-judge uses the same model as the responder. This likely inflates judge scores for responses that are partially correct, and produces inconsistent scoring on borderline cases.

**Static corpus:** The document texts are static snapshots. The pipeline has no mechanism for detecting or incorporating regulatory amendments.

**Context window pressure:** `CONTEXT_WINDOW = 4096`. With `top_k=10` chunks of up to ~500 tokens each, the retrieved context can be up to 5,000 tokens — exceeding the declared context window. In practice, LlamaIndex truncates to fit, which may silently drop relevant chunks. This has not been measured.

### 12.2 Planned Improvements

| Priority | Improvement | Expected Impact |
|----------|------------|-----------------|
| High | Expand OOS dataset to ~10 entries; add hard refusal instruction to prompt | Measurable OOS handling; fewer partial answers on irrelevant queries |
| High | Investigate `ai_act_max_fine_prohibited` with chain-of-thought prompting | Fix tier-confusion failure in AI Act fine questions |
| Medium | Add multi-hop and cross-document questions to eval dataset | Expose retrieval and generation failures not detectable with single-article lookups |
| Medium | HierarchicalNodeParser + AutoMergingRetriever | Better handling of long articles where the relevant fact is split across sub-chunks |
| Medium | Use a stronger judge model (larger Qwen or API-based) | Reduce self-judge bias; more reliable quality scores |
| Low | Add article titles from EUR-Lex to source `.txt` files | Richer chunk headers; improved retrieval for title-based queries |
| Low | MLflow experiment tracking | Persist per-run metrics for longitudinal comparison |

---

*Report generated from evaluation run `data/eval_results_20260313_202253.json`.*
