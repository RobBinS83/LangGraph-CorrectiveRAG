from typing import List, TypedDict

class AgentState(TypedDict):
    """
    Represent the state of our graph

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add serach
        documents: list of documents
    """

    question: str
    generation: str
    web_search: bool
    documents: List[str]
