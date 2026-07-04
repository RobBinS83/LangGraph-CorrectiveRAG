from typing import List, TypedDict, Any, Annotated
from langchain_core.documents import Document
import operator

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
    documents: Annotated[List[Document], operator.add]
