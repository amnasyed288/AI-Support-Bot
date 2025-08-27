from langchain.vectorstores import Chroma
from models import get_embeddings
from loaders import split_documents
from config import PERSIST_DIR
import os

def get_vectorstore():
    """Build and return Chroma vector store from documents."""
    embedding_fun = get_embeddings()
    chunks = split_documents()
    if os.path.exists(PERSIST_DIR):
        print("Loading existing vector store...")
        return Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding_fun)
    print("Creating new vector store...")
    

    vectordb = Chroma.from_documents(persist_directory=PERSIST_DIR,documents=chunks, embedding=embedding_fun)
    vectordb.persist()
    return vectordb

get_vectorstore()

def get_retriever():
    vectordb = get_vectorstore()
    retriever = vectordb.as_retriever()
    return retriever

