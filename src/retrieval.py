from typing import List

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore


def retrieve_context(
    index: VectorStoreIndex,
    question: str,
    top_k: int = 3,
) -> List[NodeWithScore]:
    """Retrieve the most relevant chunks for a user question."""
    retriever = index.as_retriever(similarity_top_k=top_k)
    retrieved_nodes = retriever.retrieve(question)
    return retrieved_nodes
