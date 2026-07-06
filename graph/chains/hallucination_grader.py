from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0)


class GradeHallucinations(BaseModel):
    """
    Binary score for hallucination present in generation answer.
    """

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no' "
    )

structured_llm_grader = llm.with_structured_output(GradeHallucinations)

system = """
You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved documents.
Give a binary score 'yes' or 'no'. 'yes' means that the answer is grounded in / supported by retrieved documents. 
"""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

hallucination_grader: RunnableSequence = hallucination_prompt | structured_llm_grader