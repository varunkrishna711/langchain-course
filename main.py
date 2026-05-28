import asyncio
from typing import List
from dotenv import load_dotenv
from langchain_chroma import vectorstores
from langchain_core.documents import Document

from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from langchain_text_splitters import RecursiveCharacterTextSplitter
from logger import (log_info, log_warning, log_error, log_debug, log_success)

load_dotenv()

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
        await vectorstores.add_documents(batch)
        log_success(f"Successfully indexed batch {batch_num} with {len(batch)} documents.")
    except Exception as e:
        log_error(f"Error indexing batch {batch_num}: {str(e)}")
        return False
    return True


async def main():
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



if __name__ == "__main__":
    asyncio.run(main())
