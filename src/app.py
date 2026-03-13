"""
Chainlit chatbot UI for the EU Legal Assistant RAG agent.

Run with:
    chainlit run app.py
"""

import asyncio

import chainlit as cl
from llama_index.core import Settings
from llama_index.core.agent.workflow import AgentStream, ReActAgent
from llama_index.core.workflow import Context

from rag_pipeline import create_legal_rag_tool, get_retriever, verify_citations

# Module-level singleton — loaded once, shared across all sessions.
# Prevents re-loading the model on websocket reconnects.
_tool = None
_tool_lock = asyncio.Lock()


async def _get_tool():
    global _tool
    if _tool is None:
        async with _tool_lock:
            if _tool is None:
                _tool = await asyncio.to_thread(create_legal_rag_tool)
    return _tool


@cl.on_chat_start
async def on_chat_start():
    # Welcome content is defined in chainlit.md — shown once as a static screen.
    async with cl.Step(name="Loading EU legal knowledge base", type="run") as step:
        tool = await _get_tool()
        step.output = "Knowledge base ready."

    agent = ReActAgent(tools=[tool], llm=Settings.llm)
    ctx = Context(agent)

    cl.user_session.set("agent", agent)
    cl.user_session.set("ctx", ctx)


@cl.on_message
async def on_message(message: cl.Message):
    agent: ReActAgent = cl.user_session.get("agent")
    ctx: Context = cl.user_session.get("ctx")

    # Create the message upfront so we can stream into it.
    msg = cl.Message(content="")
    await msg.send()

    # Stream the agent's final answer token by token.
    # The ReAct agent emits AgentStream events for every LLM token it generates,
    # including intermediate reasoning steps (Thought/Action). We buffer output
    # and only start forwarding once the final "Answer:" line begins.
    buffer = ""
    in_answer = False
    ANSWER_PREFIX = "Answer:"

    async with cl.Step(name="Consulting EU legal documents", type="tool") as step:
        handler = agent.run(message.content, ctx=ctx)
        async for event in handler.stream_events():
            if isinstance(event, AgentStream) and event.delta:
                if not in_answer:
                    buffer += event.delta
                    if ANSWER_PREFIX in buffer:
                        in_answer = True
                        idx = buffer.index(ANSWER_PREFIX) + len(ANSWER_PREFIX)
                        answer_so_far = buffer[idx:].lstrip(" \n")
                        if answer_so_far:
                            await msg.stream_token(answer_so_far)
                else:
                    await msg.stream_token(event.delta)
        response = await handler
        step.output = str(response)

    response_text = str(response)

    # Retrieve source chunks independently (pure vector search, no LLM call).
    source_elements = []
    source_footer = ""
    source_nodes = []
    retriever = get_retriever()
    if retriever:
        source_nodes = await asyncio.to_thread(retriever.retrieve, message.content)
        refs = []
        seen = set()
        for node in source_nodes:
            meta = node.node.metadata
            doc = meta.get("file_name", meta.get("file_path", "Unknown"))
            page = meta.get("page_label", None)
            articles = meta.get("chunk_articles", "")
            recitals = meta.get("chunk_recitals", "")

            label_parts = [doc]
            if page:
                label_parts.append(f"p. {page}")
            if articles:
                label_parts.append(f"Art. {articles}")
            if recitals:
                label_parts.append(f"Rec. {recitals}")
            label = " — ".join(label_parts)

            if label in seen:
                continue
            seen.add(label)

            name = f"Source {len(source_elements) + 1}"
            source_elements.append(
                cl.Text(name=name, content=f"**{label}**\n\n{node.node.text}", display="side")
            )
            refs.append(f"[{name}]")

        if refs:
            source_footer = "\n\n---\n📚 **Sources:** " + " ".join(refs)

    # Citation verification — warn if the LLM cited articles not in the retrieved chunks
    verification = verify_citations(response_text, source_nodes)
    if not verification["is_accurate"]:
        phantoms = []
        if verification["phantom_articles"]:
            phantoms.append("Article " + ", ".join(verification["phantom_articles"]))
        if verification["phantom_recitals"]:
            phantoms.append("Recital " + ", ".join(verification["phantom_recitals"]))
        source_footer += (
            "\n\n> ⚠️ **Citation notice:** The following references could not be verified "
            "in the retrieved sources: " + "; ".join(phantoms) + "."
        )

    # Replace streamed content with the canonical response + footer/sources.
    msg.content = response_text + source_footer
    msg.elements = source_elements
    await msg.update()
