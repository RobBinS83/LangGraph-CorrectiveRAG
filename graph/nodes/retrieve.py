from typing import Any, Dict

from graph.state import AgentState
from ingestion import retriever

def retrieve(state: AgentState) -> Dict[str, Any]:
    print("---RETRIEVE---")
    question = state["question"]

    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}
