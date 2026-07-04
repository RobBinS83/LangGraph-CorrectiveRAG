from typing import List, TypedDict, Any
from langchain_core.documents import Document

class AgentState(TypedDict):
    """
    Represent the state of our graph

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add serach
        documents: list of Documents
    """

    question: str
    generation: str
    web_search: bool
    documents: List[Document]
