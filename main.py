import asyncio
from dotenv import load_dotenv
from langchain_core.documents import Document

from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from logger import (log_info, log_warning, log_error, log_debug, log_success)

load_dotenv()

tavily_extract = TavilyExtract()   
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()


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


if __name__ == "__main__":
    asyncio.run(main())
