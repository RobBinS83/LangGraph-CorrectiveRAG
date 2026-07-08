# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

A Corrective RAG (CRAG) implementation built with LangGraph, LangChain, and Chroma. It combines self-corrective retrieval (grade documents, fall back to web search, check for hallucinations) with adaptive routing (route each question to either the vector store or web search before retrieving anything).

## Commands

This project uses `uv` for dependency management (Python >=3.13, see `.python-version` and `pyproject.toml`).

- Install dependencies: `uv sync`
- Run the graph end-to-end: `uv run main.py`
- Populate the local Chroma vector store from source URLs: `uv run ingestion.py` (uncomment the `Chroma.from_documents(...)` block in `ingestion.py` first — it's commented out to avoid re-ingesting on every import)
- Run all tests: `uv run pytest`
- Run a single test: `uv run pytest graph/chains/tests/test_chains.py::test_router_to_vector_store`

Required environment variables live in `.env` (loaded via `python-dotenv`): `OPENAI_API_KEY`, `TAVILY_API_KEY`, `GOOGLE_API_KEY`, and optional LangSmith tracing vars (`LANGCHAIN_API_KEY`, `LANGCHAIN_TRACING`, `LANGCHAIN_ENDPOINT`, `LANGCHAIN_PROJECT`).

Tests in `graph/chains/tests/test_chains.py` make live LLM/API calls (no mocking) — they need a working `.env` and network access, and are non-deterministic since they assert on LLM-graded output.

## Architecture

The system is a `langgraph` `StateGraph` (built in [graph/graph.py](graph/graph.py)) over a shared `AgentState` (a `TypedDict` in [graph/state.py](graph/state.py)) with fields: `question`, `generation`, `web_search` (bool flag), and `documents` (list of `Document`, deduplicated on write via `reduce_similar_documents` — a custom reducer that uses fuzzy string similarity (`difflib`) to avoid appending near-duplicate documents, e.g. when web search results overlap with vector store hits).

**Nodes** (`graph/nodes/`, each a plain function `(state) -> dict`):
- `retrieve` — queries the Chroma retriever defined in [ingestion.py](ingestion.py)
- `grade_documents` — grades each retrieved doc for relevance via an LLM structured-output chain; sets `web_search=True` if any doc is irrelevant
- `web_search` — queries Tavily and appends results as a `Document`
- `generate` — runs the RAG generation chain

**Chains** (`graph/chains/`, each a `prompt | llm.with_structured_output(...)` pipeline built with `ChatOpenAI`):
- `router.py` — `RouteQuery` classifies a question as `vector_store` or `web_search` *before* any retrieval happens (adaptive routing, decided at the graph's conditional entry point)
- `retrieval_grader.py` — `GradeDocuments` binary relevance grade per document
- `hallucination_grader.py` — `GradeHallucinations` checks whether a generation is grounded in the retrieved documents
- `answer_grader.py` — `GradeAnswer` checks whether a generation actually answers the question
- `generation.py` — the actual answer-generation chain (context + question -> answer)

**Graph control flow** ([graph/graph.py](graph/graph.py)):
1. Conditional entry point `route_question` -> `RETRIEVE` (vector store path) or `WEB_SEARCH`
2. `RETRIEVE` -> `GRADE_DOCUMENTS` -> conditional `decide_to_generate` -> `GENERATE` or `WEB_SEARCH` (if any doc was graded irrelevant)
3. `WEB_SEARCH` -> `GENERATE`
4. After `GENERATE`, `grade_generation_grounded_in_documents_and_question` runs both the hallucination and answer graders and routes to: `GENERATE` again (not grounded/"not supported"), `WEB_SEARCH` (grounded but not useful), or `END` (useful)

Node/edge name constants live in [graph/consts.py](graph/consts.py) (`RETRIEVE`, `GRADE_DOCUMENTS`, `GENERATE`, `WEB_SEARCH`) — use these instead of string literals when touching graph wiring.

The vector store (`ingestion.py`) is a persistent local Chroma DB at `./.chroma`, collection `rag-chroma`, seeded from three Lilian Weng blog posts (agents, prompt engineering, adversarial attacks on LLMs) chunked with `RecursiveCharacterTextSplitter` (chunk_size=250). Because the vector store only covers those topics, the router's system prompt explicitly scopes `vector_store` to agents/prompt-engineering/adversarial-attack questions and sends everything else to `web_search`.

Running `graph/graph.py` (e.g. via `uv run main.py`, which imports it) regenerates `graph3.png`, a Mermaid diagram of the compiled graph, as a side effect of module import.
