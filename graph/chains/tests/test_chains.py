from dotenv import load_dotenv
from typing import cast
from pprint import pprint

load_dotenv()

from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from ingestion import retriever
from graph.chains.generation import generation_chain
from graph.chains.hallucination_grader import hallucination_grader, GradeHallucinations
from graph.chains.router import question_router, RouteQuery



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


def test_generation_chain() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question": question})
    pprint(generation)


def test_hallucination_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    generation = generation_chain.invoke({"context": docs, "question": question})
    res: GradeHallucinations = hallucination_grader.invoke(
        {"documents": docs, "generation": generation}
    )
    assert res.binary_score == 'yes'

def test_hallucination_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    #generation = generation_chain.invoke({"context": docs, "question": question})
    res: GradeHallucinations = hallucination_grader.invoke(
        {
            "documents": docs, 
            "generation": "In order to make pizza, we need to first start with the dough."
        }
    )
    assert res.binary_score == 'no'

def test_router_to_vector_store() -> None:
    question = "agent memory"
    res: RouteQuery = question_router.invoke({"question": question})
    assert res.data_source == "vector_store"

def test_router_to_web_search() -> None:
    question = "how to make pizza?"
    res: RouteQuery = question_router.invoke({"question": question})
    assert res.data_source == "web_search"