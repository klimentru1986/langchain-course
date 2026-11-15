from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from tavily import TavilyClient
from langchain_tavily import TavilySearch

tavily_client = TavilyClient()


@tool
def web_search(query: str) -> str:
    """Search the web for current, factual information.

    Use this tool when the answer depends on recent events, news, prices,
    versions, people's current roles, or any fact that may have changed
    or that you are not confident about. Do NOT use it for general
    knowledge, math, or reasoning you can do yourself.

    Args:
        query: A concise search query of 2-8 keywords, written as you would
            type it into a search engine (e.g. "LangChain latest release",
            "EUR USD exchange rate today"). Avoid full sentences and
            include specific names, dates or versions when relevant.

    Returns:
        A string with the top search results. Each result contains a title,
        URL and a short snippet. Returns an empty string or a "No results"
        message if nothing was found. In that case, rephrase the query
        and try again.
    """

    return tavily_client.search(query=query)


llm_gpt = ChatOpenAI(model="gpt-5.4-nano", temperature=0)
# tools = [web_search]
tools = [TavilySearch()]
agent = create_agent(model=llm_gpt, tools=tools)


def main() -> None:
    result = agent.invoke(
        {"messages": HumanMessage(content="What is the weather in Moscow")}
    )
    print(result)


if __name__ == "__main__":
    main()
