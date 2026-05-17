from dotenv import load_dotenv
import os

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()


if __name__ == "__main__":
    print("Ingesting....")
    
    loader = TextLoader("/home/varun/Documents/medium-blog.txt")
    document = loader.load()

    print("splitting")

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"Created {len(texts)} chunks")

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    print("Ingesting")

    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ.get("INDEX_NAME"))
    print("Finish")
