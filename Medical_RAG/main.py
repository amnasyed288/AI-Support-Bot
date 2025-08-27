from graph import build_graph
from grader_models import AgentState

def main():
    graph = build_graph()
    print("State graph successfully built.")

    #example usage of the graph
    test_input = "What are the symptoms of malaria?"
    init_state= AgentState(
        query=test_input,
        response="",
        chat_history=[],
        context=[],
        last_source=""
    )

    result = graph.invoke(init_state)
    print("Graph invocation result:\n", result)

if __name__ == "__main__":
    main()
 