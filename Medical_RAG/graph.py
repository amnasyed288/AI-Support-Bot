from langgraph.graph import END, StateGraph
from nodes import (question_router_node, fallback_node, retrieve_node, web_search_node,
                   filter_docs_node, should_generate, rag_node, hallucination_and_answer_relevance_node)
import os
from grader_models import AgentState

def build_graph():
    """Builds the state graph for the medical RAG system."""
    graph= StateGraph(AgentState)
    graph.add_node("VectorStore", retrieve_node)
    graph.add_node("SearchEngine", web_search_node)
    graph.add_node("fallback", fallback_node)
    graph.add_node("filter_docs", filter_docs_node)
    graph.add_node("rag", rag_node)
    graph.set_conditional_entry_point(question_router_node,
                                      {
            "VectorStore":"VectorStore",
            "SearchEngine":"SearchEngine",
            "llm_fallback":"fallback"
        }
    )
    graph.add_edge("fallback", END)
    graph.add_edge("VectorStore", "filter_docs")
    graph.add_edge("SearchEngine", "filter_docs")
    graph.add_conditional_edges("filter_docs",should_generate,{
        "SearchEngine":"SearchEngine",
        "VectorStore":"VectorStore",
        "generate":"rag"
    })
    graph.add_conditional_edges("rag", hallucination_and_answer_relevance_node,{
        "useful":END, 
        "generate":"rag",
        "SearchEngine":"SearchEngine",
        "VectorStore":"VectorStore",
        "llm_fallback":"fallback"
    })

    graph = graph.compile()
    return graph

def visualize_graph(filename="graph.png"):
    """Builds and saves the state graph as an image"""
    graph = build_graph()
    img_bytes = graph.get_graph().draw_mermaid_png()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, filename)

    with open(save_path, "wb") as f:
        f.write(img_bytes)

    print(f"Graph saved to {save_path}")

visualize_graph()
 
