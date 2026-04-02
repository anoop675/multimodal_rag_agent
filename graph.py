"""
LangGraph Agent — Graph Assembly
----------------------------------
Wires together all nodes into a compiled LangGraph app.
"""

import functools
from langgraph.graph import StateGraph, END

from state import AgentState
from nodes import (
    vision_node,
    fallback_node,
    encode_query_node,
    retrieval_node,
    synthesis_node,
    route_after_vision,
)


def build_graph(vision_processor, text_encoder, faiss_retriever, web_retriever, synthesis_llm):
    """
    Construct and compile the LangGraph agent.

    Parameters
    ----------
    vision_processor : VisionProcessor
    text_encoder     : TextEncoder
    faiss_retriever  : FAISSRetriever
    web_retriever    : WebSearchRetriever
    synthesis_llm    : SynthesisLLM

    Returns
    -------
    Compiled LangGraph application.
    """
    graph = StateGraph(AgentState)

    graph.add_node("vision",        functools.partial(vision_node,       vision_processor=vision_processor))
    graph.add_node("fallback",      fallback_node)
    graph.add_node("encode_query",  functools.partial(encode_query_node, text_encoder=text_encoder))
    graph.add_node("retrieval",     functools.partial(retrieval_node,    faiss_retriever=faiss_retriever,
                                                                          web_retriever=web_retriever,
                                                                          text_encoder=text_encoder))
    graph.add_node("synthesis",     functools.partial(synthesis_node,    synthesis_llm=synthesis_llm))

    graph.set_entry_point("vision")
    graph.add_conditional_edges("vision", route_after_vision,
                                {"encode_query": "encode_query", "fallback": "fallback"})
    graph.add_edge("fallback", END)
    graph.add_edge("encode_query", "retrieval")
    graph.add_edge("retrieval", "synthesis")
    graph.add_edge("synthesis", END)

    app = graph.compile()
    print("LangGraph agent compiled.")
    return app
