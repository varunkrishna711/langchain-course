import asyncio
import os
import ssl
from typing import Any, Dict, List

import certifi
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from logger import (log_info, log_warning, log_error, log_debug, log_success)

load_dotenv()


ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = OllamaEmbeddings(model="nomic-embed-text")

# chroma = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

vectorstore = PineconeVectorStore(index_name="langchain-docs-2026", embedding=embeddings)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()


async def index_documents_async(documents: List[Document], batch_size: int = 50):
    log_info("VECTOR STORAGE PHASE")

    log_debug(f"Preparing to add {len(documents)} documents into vector storage...")

    batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]

    log_debug(f"Created {len(batches)} batches for indexing.")


    async def add_batch(batch: List[Document], batch_num: int):
        try:
            await vectorstore.aadd_documents(batch)
            log_success(f"Successfully indexed batch {batch_num} with {len(batch)} documents.")
        except Exception as e:
            log_error(f"Error indexing batch {batch_num}: {str(e)}")
            return False
        return True

    # Process batches concurrently
    tasks = [add_batch(batch, idx + 1) for idx, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Count successful batches

    successful = sum(1 for result in results if result is True)

    if successful == len(batches):
        log_success(f"Vectorstore indexing: All batches processed successfully! ({successful}/{len(batches)})")
    else:
        log_warning(f"Vector indexing: Processed {successful}/{len(batches)} successfully")




async def main():
    """Main function to run the ingestion process."""
    log_debug("DOCUMENTING INGESTION PIPELINE")

    log_info("Starting the tavily crawl...")

    result = tavily_crawl.invoke({
        "url": "https://docs.langchain.com/", 
        "max_depth": 1, 
        "extract_depth": "advanced",
    })

    all_docs = [Document(page_content=result["raw_content"], metadata={"source": result["url"]}) for result in result["results"]]
    log_success(f"Crawled {len(all_docs)} documents.")

    # Split the documents into smaller chunks
    log_info("DOCUMENT CHUNKING PHASE")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(f"Split into {len(splitted_docs)} from {len(all_docs)} chunks.")

    await index_documents_async(splitted_docs, batch_size=500)

    log_success("PIPELINE COMPLETE")
    log_info("Summary: ")
    log_info(f"=== Documents extracted: {len(all_docs)}")
    log_info(f"=== Chunks created: {len(splitted_docs)}")


if __name__ == "__main__":
    asyncio.run(main())