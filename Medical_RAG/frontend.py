import streamlit as st
from graph import build_graph
from grader_models import AgentState
from langchain.schema import HumanMessage, AIMessage

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "graph" not in st.session_state:
    st.session_state.graph = build_graph()
if "source_docs" not in st.session_state:
    st.session_state.source_docs = []

st.title("🩺 Medical RAG Chatbot")

if st.button("Clear Chat"):
    st.session_state.chat_history = []
    st.session_state.source_docs = []


user_input = st.chat_input("Ask me a medical question...")

if user_input:
    # Append user message to history
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    # Create agent state
    init_state = AgentState(
        query=user_input,
        response="",
        chat_history=st.session_state.chat_history,
        context=[],
        last_source=""
    )

    with st.spinner("Generating response..."):
    
        result = st.session_state.graph.invoke(init_state)
        response = result.get("response", "Sorry, I could not generate a response.")
        source_docs = result.get("context",[])


    st.session_state.chat_history.append(AIMessage(content=response))
    st.session_state.source_docs = source_docs


for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").markdown(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").markdown(msg.content)

if st.session_state.source_docs:
    st.subheader("📄 Source Documents")
    for doc in st.session_state.source_docs:
        source = doc.metadata.get("source", "") if hasattr(doc, "metadata") else ""
        st.markdown(f"**Source:** {source}")
