from langchain_community.tools.tavily_search import TavilySearchResults
from retrievers import get_retriever

def retrieve_docs(query:str):
    """Use to retrieve relevant documents from the vector store."""
    retriever = get_retriever()
    context = retriever.invoke(query)
    return context


def web_search(query:str):
    web_search_tool = TavilySearchResults()
    results = web_search_tool.invoke(query)
    return results