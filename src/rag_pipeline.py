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
from pathlib import Path
from typing import Any

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.base.llms.types import CompletionResponse, LLMMetadata
from llama_index.core.llms import CustomLLM
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.prompts import PromptTemplate
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DOCS_DIR = Path("data/rag")
PERSIST_DIR = Path("data/rag_index")

# ---------------------------------------------------------------------------
# Legal metadata extraction
# ---------------------------------------------------------------------------


def _extract_legal_metadata(text: str) -> dict:
    """Extract article numbers, recital numbers and chapter references from a text chunk."""
    articles = sorted(set(re.findall(r'[Aa]rticle\s+(\d+)', text)), key=int)
    recitals = sorted(set(re.findall(r'\((\d+)\)\s+[A-Z]', text)), key=int)
    chapters = list(dict.fromkeys(re.findall(r'CHAPTER\s+[IVX\d]+', text)))
    return {
        "chunk_articles": ", ".join(articles) if articles else "",
        "chunk_recitals": ", ".join(recitals) if recitals else "",
        "chunk_chapters": ", ".join(chapters) if chapters else "",
    }


# ---------------------------------------------------------------------------
# Qwen2.5-7B-Instruct custom LLM wrapper
# ---------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
CONTEXT_WINDOW = 4096
MAX_NEW_TOKENS = 512


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
                dtype=torch.float16,  # replaces deprecated torch_dtype
                token=hf_token,
                max_memory={0: "8GiB", "cpu": "20GiB"},
            )
            # Pass generation params at call time, not in the constructor,
            # to avoid conflicts with the model's stored generation_config.
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
        output = pipe(
            formatted,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )[0]["generated_text"]
        return CompletionResponse(text=output)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **_kwargs: Any):
        response = self.complete(prompt)
        yield CompletionResponse(text=response.text, delta=response.text)


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------


def build_index(force_rebuild: bool = False) -> VectorStoreIndex:
    """Load index from disk or build it from scratch.

    Args:
        force_rebuild: If True, re-index even if a persisted index exists.
    """
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.llm = QwenLLM()
    # NOTE: If you change chunk_size or chunk_overlap, delete data/rag_index/ to force a rebuild.
    Settings.chunk_size = 1024
    Settings.chunk_overlap = 128

    if not force_rebuild and PERSIST_DIR.exists():
        print(f"Loading index from {PERSIST_DIR} …")
        storage_context = StorageContext.from_defaults(persist_dir=str(PERSIST_DIR))
        return load_index_from_storage(storage_context)

    print(f"Building index from {DOCS_DIR} …")
    documents = SimpleDirectoryReader(str(DOCS_DIR)).load_data()
    splitter = SentenceSplitter(chunk_size=Settings.chunk_size, chunk_overlap=Settings.chunk_overlap)
    nodes = splitter.get_nodes_from_documents(documents)
    for node in nodes:
        node.metadata.update(_extract_legal_metadata(node.text))
    index = VectorStoreIndex(nodes, show_progress=True)
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(PERSIST_DIR))
    print(f"Index persisted to {PERSIST_DIR}")
    return index


# ---------------------------------------------------------------------------
# Module-level retriever — used by app.py to fetch source nodes independently
# ---------------------------------------------------------------------------

_retriever = None

LEGAL_QA_PROMPT = PromptTemplate(
    "You are an EU legal expert assistant. Answer using ONLY the context provided.\n\n"
    "RULES:\n"
    "1. Only cite Article or Recital numbers that explicitly appear in the context below.\n"
    "2. Do NOT use training knowledge to add article numbers absent from the context.\n"
    "3. If unsure, say 'The retrieved documents indicate...' without citing specific numbers.\n\n"
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
    for node in source_nodes:
        text = node.node.text if hasattr(node, "node") else node.text
        available_articles.update(re.findall(r'[Aa]rticle\s+(\d+)', text))
        available_recitals.update(re.findall(r'[Rr]ecital\s+(\d+)', text))

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


def create_legal_rag_tool(
    similarity_top_k: int = 5,
    force_rebuild: bool = False,
) -> QueryEngineTool:
    """Return a QueryEngineTool ready to be used inside a LlamaIndex agent.

    Args:
        similarity_top_k: Number of document chunks to retrieve per query.
        force_rebuild: Rebuild the vector index even if a cached one exists.
    """
    global _retriever
    index = build_index(force_rebuild=force_rebuild)
    _retriever = index.as_retriever(similarity_top_k=similarity_top_k)
    query_engine = index.as_query_engine(similarity_top_k=similarity_top_k)
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
