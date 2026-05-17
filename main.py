import os
from operator import itemgetter

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

print("Initialising Components...")

embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

vectorstore = PineconeVectorStore(index_name = os.environ.get("INDEX_NAME"),embedding=embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k":3})

prompt_template = ChatPromptTemplate.from_template(
    """Answer the following questions only based on the following context:
    
        {context}

        Question: {question}

        Provide a detailed answer:    
    """
)


def format_docs(docs):
    """Format retrieved docs into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def retrieval_chain_withot_lcel(query: str):

    # Step 1: Retrieve docs
    docs = retriever.invoke(query)

    # Step 2: Format docs into context string
    context = format_docs(docs)

    # Step 3: Format prompt with context and question
    messages = prompt_template.format_messages(context=context,question=query)

    # Step 4: Invoke LLM
    response = llm.invoke(messages)

    return response.content


def create_retrieval_chain_with_lcel():
    retrieval_chain = (
        Runnable.assign(context=itemgetter("question") | retriever | format_docs)
        | prompt_template 
        | llm
        | StrOutputParser()
    )

    return retrieval_chain

if __name__ == "__main__":
    print("Retrieving...")

    query = "What is Pinecone in Machine Learning?"

    ##########################################
    # Raw Invokation without RAG
    ##########################################

    # print("=" * 70)
    # print("IMPLEMENTATION 0: Raw LLM invokation, No RAG")
    # print("=" * 70)

    # result_raw = llm.invoke([HumanMessage(content=query)])
    # print("\n Answer:")
    # print(result_raw.content)


    ##########################################
    # Implementation without LCEL
    ##########################################

    # print("=" * 70)
    # print("IMPLEMENTATION 1: Without LCEL")
    # print("=" * 70)

    # result_without_lcel = retrieval_chain_withot_lcel(query)
    # print("\n Answer")
    # print(result_without_lcel)


    ##########################################
    # Implementation with LCEL (Better Approach)
    ##########################################

    print("=" * 70)
    print("IMPLEMENTATION 2: With LCEL")
    print("=" * 70)

    chain_with_lcel = create_retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"quesion":query})
    print("\n Answer")
    print(result_with_lcel)