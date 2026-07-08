from typing import List, TypedDict, Any, Annotated
from langchain_core.documents import Document
import operator
import difflib

def is_highly_similar(new_text: str, existing_texts: List[str], threshold: float = 0.85) -> bool:
    """
    Compares new text against a list of existing texts.
    Returns True if similarity exceeds the threshold.
    """
    for text in existing_texts:
        similarity = difflib.SequenceMatcher(None, new_text, text).ratio()
        if similarity >= threshold:
            return True
    return False

def reduce_similar_documents(
        left: list[Document] | None,
        right: list[str] | list[Document] | str | Document | None,
) -> list[Document]:
    """
    Reducer that deduplicates incoming documnets based on fuzzy similarity.
    Return a strict list of LangChain Document objects.
    """
    if not left:
        left = []
    if not right:
        return left
    
    # normalize the incoming data to a list
    if not isinstance(right, list):
        right = [right]

    # Start with the existing list of Documents
    deduplicated_docs = left

    # Extract and normalize the text from existing docs once, so we don't have to 
    # do it repeatedly inside the loop
    existing_texts = [" ".join(doc.page_content.lower().split()) for doc in deduplicated_docs]
    for item in right:
        # 1. Coerce the incoming item into a Document if it's a string
        if isinstance(item, str):
            doc = Document(page_content=item)
        else:
            doc = item

        # 2. Extract and normalize the text for similarity check
        normalized_new_text = " ".join(doc.page_content.lower().split())

        # 3. Check if the text is highly similar to anything we already have
        if not is_highly_similar(normalized_new_text, existing_texts):
            deduplicated_docs.append(doc)
            existing_texts.append(normalized_new_text)

    return deduplicated_docs


class AgentState(TypedDict):
    """
    Represent the state of our graph

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of Documents
    """

    question: str
    generation: str
    web_search: bool
    documents: Annotated[List[Document], reduce_similar_documents]
