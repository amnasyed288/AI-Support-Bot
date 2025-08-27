from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP, URLS

def load_docs(urls):
    """Load documents from predefined URLs."""
    loader = WebBaseLoader(urls)
    docs = loader.load()
    return docs

def split_documents():
    """Split documents into chunks using RecursiveCharacterTextSplitter."""
    docs = load_docs(URLS)
    splitter=RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    return chunks




