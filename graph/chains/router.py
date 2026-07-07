from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

class RouteQuery(BaseModel):
    """
    Route a user query to the most relevant datasource
    """

    data_source: Literal["vector_store", "web_search"] = Field(
        ...,
        description="Given a user question, choose to route it to web_search or vector_store"
    )

llm = ChatOpenAI(temperature=0)
structured_llm_router = llm.with_structured_output(RouteQuery)

system = """
You are an expert at routing a user question to a vector store or web search.
The vector store contains documents related to agents, prompt engineering, and adversarial attacks. 
Use the vector store for questions on these topics. For everything else, use web search.
"""

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),  
    ]
)

question_router = route_prompt | structured_llm_router
