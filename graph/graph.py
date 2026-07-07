from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from graph.consts import RETRIEVE, GENERATE, GRADE_DOCUMENTS, WEB_SEARCH
from graph.nodes import retrieve, generate, grade_documents, web_search
from graph.state import AgentState
from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.router import question_router, RouteQuery

load_dotenv()

def decide_to_generate(state: AgentState):
    
    print("---ASSESS GRADED DOCUMENTS---")
    if state["web_search"]:
        print("---DECISION: NOT ALL DOCUMENTS ARE RELEVANT TO QUESTION, INCLUDE WEB SEARCH---")
        return WEB_SEARCH
    else:
        print("---DECISION: GENERATE---")
        return GENERATE
    
def grade_generation_grounded_in_documents_and_question(state: AgentState) -> str:
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )

    if score.binary_score == 'yes':
        print("---DECISION: GENERAION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke(
            {"question": question, "generation": generation}
        )
        if score.binary_score == 'yes':
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESSES QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERAION IS NOT GROUNDED IN DOCUMENTS, REGENERATE AGAIN---")
        return "not supported"
    
def route_question(state: AgentState) -> str:
    print("---ROUTE QUESTION---")
    question = state["question"]
    source: RouteQuery = question_router.invoke({"question": question})

    if source.data_source == WEB_SEARCH:
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return WEB_SEARCH
    elif source.data_source == "vector_store":
        print("---ROUTE QUESTION TO RAG---")
        return RETRIEVE


builder = StateGraph(AgentState)

builder.add_node(RETRIEVE, retrieve)
builder.add_node(GRADE_DOCUMENTS, grade_documents)
builder.add_node(GENERATE, generate)
builder.add_node(WEB_SEARCH, web_search)

#builder.set_entry_point(RETRIEVE)

builder.set_conditional_entry_point(
    route_question,
    path_map={
        WEB_SEARCH: WEB_SEARCH,
        RETRIEVE: RETRIEVE,
    }
)

builder.add_edge(RETRIEVE, GRADE_DOCUMENTS)

builder.add_conditional_edges(GRADE_DOCUMENTS, 
                              decide_to_generate, 
                              path_map={
                                  WEB_SEARCH: WEB_SEARCH,
                                  GENERATE: GENERATE,
                              }
)

builder.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    path_map={
        "not supported": GENERATE,
        "useful": END,
        "not useful": WEB_SEARCH,
    }   
)

builder.add_edge(WEB_SEARCH, GENERATE)
#builder.add_edge(GENERATE, END)

workflow = builder.compile()

workflow.get_graph().draw_mermaid_png(output_file_path="graph3.png")

