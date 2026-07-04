from typing import Any, Dict

from graph.state import AgentState
from graph.chains.retrieval_grader import retrieval_grader

def grade_documents(state: AgentState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question.
    If any document is not relevant, we will set a flag to run web_search

    Args:
        state (dict): the current AgentState

    Returns:
        state (dict): filtered out irrelevant documents and update web_search state
    """

    print("---CHECK DOCUMENTS RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    web_search = False
    for doc in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": doc.page_content}
        )
        grade = score.binary_score
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(doc)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            if web_search == False:
                web_search = True
            continue

    return {"documents": filtered_docs, "web_search": web_search, "question": question}
        