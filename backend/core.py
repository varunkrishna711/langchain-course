import os
from typing import Any, Dict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings

load_dotenv()

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = PineconeVectorStore(index_name="langchain-docs-2026", embedding=embeddings)

model = init_chat_model(model="llama3.2", model_provider="ollama")

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve relevant documentation to help answer question about Langchain"""

    retrieved_docs = vectorstore.as_retriever().invoke(query, k=4)

    serialized = "\n\n".join(
        (f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}")
            for doc in retrieved_docs        
        )
    
    return serialized, retrieved_docs


def run_llm(query: str):
    """
        Run a RAG pipeline to answer a query using retrieved documentation

        Args:
            query: The user's question
        
        Returns:
            Dictionary containing:
                - answer: The generated answer
                - context: List of retrieved documents
    """

    system_prompt = (
        "You are a helpful assistant for answering questions about the Langchain documentation."
        "You have access to a tool for retrieving relevant documentation."
        "Use the tool to get relevant context and then answer the user's question based on that context."
        "Always cite sources in your answer using the format [source: URL]."
        "If you don't know the answer, say you don't know instead of making something up."
    )

    agent = create_agent(model, tools=[retrieve_context], system_prompt=system_prompt)

    messages = [{"role": "user", "content": query}]

    response = agent.invoke({"messages": messages})

    answer = response["messages"][-1].content

    context_docs = []

    for message in response["messages"]:
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):

            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)

    return {"answer": answer, "context": context_docs}


if __name__ == "__main__":
    query = "What are deep agents in Langchain and how do I use them?"
    result = run_llm(query)

    print(result)