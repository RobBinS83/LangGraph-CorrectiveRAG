from dotenv import load_dotenv

load_dotenv()

from graph.graph import workflow

if __name__ == "__main__":
    print("Hello from Corrective-RAG!")
    print(workflow.invoke(input={"question": "How to make pizza?"}))
