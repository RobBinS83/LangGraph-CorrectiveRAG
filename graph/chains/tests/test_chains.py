from dotenv import load_dotenv
from typing import cast

load_dotenv()

from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from ingestion import retriever



def test_retrieval_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[0].page_content

    res: GradeDocuments = cast(
        GradeDocuments,
        retrieval_grader.invoke(
            {"question": question, "document": doc_txt})
            )

    assert res.binary_score == "yes"


def test_retrieval_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[0].page_content

    res: GradeDocuments = cast(
        GradeDocuments,
        retrieval_grader.invoke(
            {"question": "How to make Pizza?", "document": doc_txt})
            )

    assert res.binary_score == "no"





    