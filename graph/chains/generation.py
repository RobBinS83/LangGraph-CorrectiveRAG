from dotenv import load_dotenv
load_dotenv()

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


llm = ChatOpenAI(temperature=0)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Answer the user's question using the provided context."),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)

generation_chain = prompt | llm | StrOutputParser()


