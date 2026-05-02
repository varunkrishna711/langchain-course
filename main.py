from typing import List
from pydantic import BaseModel, Field

from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch


class Source(BaseModel):
    """Schema for a source returned by the agent."""
    url:str = Field(description="The URL of the source.")

class AgentResponse(BaseModel):
    """Schema for the agent's response."""
    answer:str = Field(description="The answer to the user's query.")
    sources: List[Source] = Field(default_factory=list, description="A list of sources used to generate the answer.")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.9)
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": HumanMessage(content="Search for 3 job openings for software engineers in India and summarize the results.")})
    print(result)

if __name__ == "__main__":
    main()
