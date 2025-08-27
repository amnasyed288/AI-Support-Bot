from grader_models import AgentState
from chains import router_chain, fallback_chain, grader_chain, rag_chain, hallucination_chain, answer_grader_chain
from query_processor import retrieve_docs, web_search
from langchain.schema import Document

def question_router_node(state:AgentState):
    """Routes the query to the appropriate tool based on the router chain's output."""
    query = state['query']
    try:
        response = router_chain(query)
    except Exception:
        return "llm_fallback"
    
    if "tool_calls" not in response.additional_kwargs:
        print("NO TOOL CALLED")
        return "llm_fallback"
    if len(response.additional_kwargs["tool_calls"])==0:
        raise "Router can't decide route"
    
    if "VectorStore" in response.additional_kwargs["tool_calls"][0]["function"]["name"]:
        print("ROUTING TO VECTOR STORE")
        return "VectorStore"
    elif "SearchEngine" in response.additional_kwargs["tool_calls"][0]["function"]["name"]:
        print("ROUTING TO SEARCH ENGINE")
        return "SearchEngine"
    

def fallback_node(state:AgentState):
    """Handles fallback using the fallback chain."""
    query=state['query']
    chat_history=state['chat_history']
    response = fallback_chain(query, chat_history)
    return {"response": response}

def retrieve_node(state:AgentState):
    query=state['query']
    context = retrieve_docs(query)
    return {"context":context, "last_source":"VectorStore"}

def web_search_node(state:AgentState):
    """Handles web search using the TavilySearchResults tool."""
    query=state['query']
    results = web_search(query)
    documents = [
        Document(
            page_content=doc["content"], metadata={
                "source":doc["url"]
            }
        ) for doc in results
    ]
    
    return {"context":documents, "last_source":"SearchEngine"}

def filter_docs_node(state:AgentState):
    """Filters the retrieved documents using the grader chain."""
    query=state['query']
    documents=state['context']
    filtered_docs=[]
    for i, doc in enumerate(documents, start=1):
        grader_response = grader_chain(query,doc)
        if grader_response.grade=="relevant":
            print(f"RETRIEVED DOCUMENT {i} IS RELEVANT")
            filtered_docs.append(doc)
        else:
            print(f"RETRIEVED DOCUMENT {i} IS IRRELEVANT")
    
    return {"context":filtered_docs}
        

def should_generate(state:AgentState):
    """Decides whether to generate an answer or re-route based on the presence of relevant documents."""
    filtered_documents = state['context']
    last_source = state['last_source']
    if not filtered_documents:
        if last_source == "VectorStore":
            print("NO DOCUMENTS RETRIEVED FROM VECTOR STORE, ROUTING TO WEB SEARCH")
            return "SearchEngine"
        if last_source == "SearchEngine":
            print("NO DOCUMENTS RETRIEVED FROM WEB SEARCH, ROUTING TO VECTOR STORE")
            return "VectorStore"
        raise ValueError("No documents retrieved from either source.")  
    else:
        print("SOME DOCUMENTS RETRIEVED ARE RELEVANT")
        return "generate"
    
def rag_node(state:AgentState):
    """Generates a response using the RAG chain."""
    query=state['query']
    context = state['context']
    response = rag_chain(query,context)
    return {"response":response}

def hallucination_and_answer_relevance_node(state:AgentState):
    """Evaluates the generated response for hallucination and relevance."""
    query=state['query']
    context = state['context']
    response=state['response']
    last_source=state['last_source']


    hallucination_grade=hallucination_chain(response,context)
    if hallucination_grade.grade=="no":
        print("NOT HALLUCINATED")
        answer_relevance_grade=answer_grader_chain(query,response)
        if answer_relevance_grade.grade=="yes":
            print("ANSWER IS RELEVANT TO THE QUERY")
            return "useful"
        else:
            print("ANSWER IS NOT RELEVANT TO THE QUERY")
            if last_source=="VectorStore":
                print("ROUTING TO WEB SEARCH")
                return "SearchEngine"
            elif last_source=="SearchEngine":
                print("ROUTING TO VECTOR STORE")
                return "VectorStore"
            else:
                return "llm_fallback"
            
    print("HALLUCINATED")
    return "generate"
    