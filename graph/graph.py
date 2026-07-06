from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from graph.consts import RETRIEVE, GENERATE, GRADE_DOCUMENTS, WEB_SEARCH
from graph.nodes import retrieve, generate, grade_documents, web_search
from graph.state import AgentState

load_dotenv()

def decide_to_generate(state: AgentState):
    
    print("---ASSESS GRADED DOCUMENTS---")
    if state["web_search"]:
        print("---DECISION: NOT ALL DOCUMENTS ARE RELEVANT TO QUESTION---")
        return WEB_SEARCH
    else:
        print("---DECISION: GENERATE---")
        return GENERATE
    
builder = StateGraph(AgentState)

builder.add_node(RETRIEVE, retrieve)
builder.add_node(GRADE_DOCUMENTS, grade_documents)
builder.add_node(GENERATE, generate)
builder.add_node(WEB_SEARCH, web_search)

builder.set_entry_point(RETRIEVE)
builder.add_edge(RETRIEVE, GRADE_DOCUMENTS)

builder.add_conditional_edges(GRADE_DOCUMENTS, 
                              decide_to_generate, 
                              path_map={
                                  WEB_SEARCH: WEB_SEARCH,
                                  GENERATE: GENERATE,
                              }
)

builder.add_edge(WEB_SEARCH, GENERATE)
builder.add_edge(GENERATE, END)

workflow = builder.compile()

workflow.get_graph().draw_mermaid_png(output_file_path="graph.png")

