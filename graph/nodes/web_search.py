from typing import Any, Dict

from langchain_core.documents import Document
from langchain_tavily import TavilySearch

from graph.state import AgentState
from dotenv import load_dotenv

load_dotenv()

web_search_tool = TavilySearch(max_results=3)

def web_search(state: AgentState) -> Dict[str, Any]:
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents", [])

    tavily_results = web_search_tool.invoke({"query": question})
    results = tavily_results.get("results", [])
    joined_tavily_results = "\n\n".join(
        [res.get("content", "") for res in results]
    )

    # transfer to LangChain Document
    web_results = Document(page_content=joined_tavily_results)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]

    return {"documents": documents, "question": question}

if __name__ == "__main__":
    web_search(
        state={
            "question": "agent memory", 
            "generation": "",
            "web_search": False,
            "documents": [],
        }
    )

