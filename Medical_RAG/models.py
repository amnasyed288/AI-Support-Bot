
from langchain_groq import ChatGroq
from config import MODEL_NAME, EMBEDDING_FUNCTION   
from langchain.vectorstores import Chroma
from tools import VectorStore, SearchEngine


embedding_function= EMBEDDING_FUNCTION
def get_embeddings():
    """Return HuggingFace embeddings function."""
    return embedding_function

def get_llm():
    """Return the ChatGroq LLM."""
    llm= ChatGroq(model=MODEL_NAME)
    return llm

def llm_with_tools():
    """Returns the llm binded with tools"""
    tools = [VectorStore, SearchEngine]
    llm = get_llm()
    llm_with_tools = llm.bind_tools(tools)
    return llm_with_tools







    
