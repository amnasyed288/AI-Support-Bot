from prompts import get_router_prompt, get_fallback_prompt, get_grader_prompt, get_rag_prompt, get_hallucination_prompt, get_answer_grader_prompt
from models import llm_with_tools,get_llm
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from grader_models import Grader, HallucinationGrader, AnswerGrader
from query_processor import retrieve_docs
from langchain_core.documents import Document


def router_chain(query:str):
    """Creates a routing chain using the router prompt and LLM with tools."""
    llm = llm_with_tools()
    router_prompt = get_router_prompt()
    router_chain = router_prompt | llm
    return router_chain.invoke({"query": query})

def fallback_chain(query:str, chat_history: list):
    """Creates a fallback chain using the fallback prompt and LLM with tools."""
    llm = get_llm()
    fallback_prompt = get_fallback_prompt()
    fallback_chain=(
        {
            "query":itemgetter("query"),
            "chat_history":lambda x: "\n".join(
                [
                    (
                        f"human:{msg.context}"
                        if isinstance(msg, HumanMessage) else
                        f"ai:{msg.content}"
                    ) for msg in x["chat_history"]
                ] 
            )
        }
        | fallback_prompt
        | llm
        | StrOutputParser()
    )
    return fallback_chain.invoke({"query": query, "chat_history": chat_history})

def grader_chain(query:str, context:list[Document]):
    """Creates a grading chain using the grader prompt and LLM with tools."""
    llm = get_llm()
    grader_llm = llm.with_structured_output(Grader, method="json_mode")
    grader_prompt = get_grader_prompt()
    grader_chain = grader_prompt | grader_llm
    return grader_chain.invoke({"query": query, "context": context})

def rag_chain(query: str, context: list[Document]):
    """Creates a RAG chain using the RAG prompt and LLM with tools."""
    llm = get_llm()
    rag_prompt = get_rag_prompt()
    rag_chain = rag_prompt | llm |StrOutputParser()
    return rag_chain.invoke({"query": query, "context": context})

def hallucination_chain(response: str, context: list[Document]):
    """Creates a hallucination detection chain using the hallucination prompt and LLM with tools."""
    llm = get_llm()
    hallucination_prompt = get_hallucination_prompt()
    hallucination_llm = llm.with_structured_output(HallucinationGrader, method="json_mode")
    hallucination_chain = hallucination_prompt | hallucination_llm
    return hallucination_chain.invoke({"response": response, "context": context})

def answer_grader_chain(query: str, response: str):
    """Creates an answer grading chain using the grader prompt and LLM with tools."""
    llm = get_llm()
    answer_grader_llm = llm.with_structured_output(AnswerGrader, method="json_mode")
    answer_grader_prompt = get_answer_grader_prompt()
    answer_grader_chain = answer_grader_prompt | answer_grader_llm
    return answer_grader_chain.invoke({"query": query, "response": response})

