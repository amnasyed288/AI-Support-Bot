from langchain.prompts import ChatPromptTemplate

#Router prompt
def get_router_prompt():
    """
    Generates a routing prompt for handling medical queries.

    Parameters:
        query (str): The user query.

    Returns:
        ChatPromptTemplate: A formatted router prompt.
    """
    router_prompt_template = (
    "You are an expert in routing user queries to either VectorStore or SearchEngine.\n"
    "Use VectorStore for queries related to the following medical conditions:\n"
    "malaria, type 1 diabetes, type 2 diabetes, migraines, asthma, heart disease, "
    "cancer, skin problems (eczema), allergies, cold and flu, hepatitis, HIV/AIDS, "
    "mental health disorders (depression, anxiety, addiction), Alzheimer's, Parkinson's, "
    "epilepsy, arthritis, osteoporosis, stroke, pain management, and sexual health.\n"
    "Use SearchEngine for all other medical queries not covered by the VectorStore.\n"
    "If a query is not medically-related, you must output 'Not medically-related' "
    "and do not try to use any tool.\n\n"
    "query: {query}"
    )

    router_prompt = ChatPromptTemplate.from_template(router_prompt_template)
    return router_prompt

#fallback prompt
def get_fallback_prompt():
    """
    Generates a fallback prompt for handling non-medical queries.
    """
    fallback_prompt_template="""You are a medical expert who is responsible for responding to medical related queries.
    If a query is not medically-related then politely acknowledge your limitations and provide concise responses to only medical related queries.
    Current conversations:\n\n{chat_history}
    Query:{query}
    """
    fallback_prompt = ChatPromptTemplate.from_template(fallback_prompt_template)
    return fallback_prompt

#grader prompt
def get_grader_prompt():
    """
    Generates a grading prompt for assessing the relevance of retrieved documents to a query.

    Returns:
        ChatPromptTemplate: A formatted grader prompt.
    """
    grader_prompt_template=""""You are a grader tasked with assessing the relevance of a given context to a query. 
    If the context is relevant to the query, score it as "relevant". Otherwise, give "irrelevant".Answer in the form of json without any additional text with key 'grade'.

    Query: {query}

    Context: {context}"""
    grader_prompt = ChatPromptTemplate.from_template(grader_prompt_template)
    return grader_prompt

#rag prompt
def get_rag_prompt():
    """
    Generates a RAG (Retrieval-Augmented Generation) prompt for answering medical queries based on provided context.

    Returns:
        ChatPromptTemplate: A formatted RAG prompt.
    """
    rag_prompt_template = """You are a medical expert who is responsible for responding to medical related queries.
    Use the following context given below to answer the question. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    "context:{context}\n\n"
    "query:{query}")"""
    
    rag_prompt = ChatPromptTemplate.from_template(rag_prompt_template)
    return rag_prompt

#hallucination prompt
def get_hallucination_prompt():
    """
    Generates a hallucination detection prompt for assessing whether an LLM's response is based on provided context.

    Returns:
        ChatPromptTemplate: A formatted hallucination detection prompt.
    """
    hallucination_prompt_template = """You are a grader assessing whether a response from an llm is based on a given context.
    If the llm's response is not based on the given context give a score of 'yes' meaning it's a hallucination otherwise give 'no'.
    Answer in the form of json without any additional text with key 'grade'."""
    
    
    hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", hallucination_prompt_template),
        ("human","response:{response}\ncontext:{context}")
    ]
    )
    return hallucination_prompt


#answer grader prompt
def get_answer_grader_prompt():
    """
    Generates an answer grading prompt for assessing whether a provided answer addresses a given query.

    Returns:
        ChatPromptTemplate: A formatted answer grading prompt.
    """
    answer_grader_system_prompt_template = (
        "You are a grader assessing whether a provided answer is in fact an answer to the given query.\n"
        "If the provided answer does not answer the query give a score of 'no' otherwise give 'yes'\n"
        "Just give the grade in json with 'grade' as a key and a binary value of 'yes' or 'no' without additional explanation"
    )

    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", answer_grader_system_prompt_template),
            ("human", "query: {query}\n\nanswer: {response}")
        ]
    )
    return answer_prompt









