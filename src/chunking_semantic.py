from typing import List

from llama_index.core import Document
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import BaseNode

from src.embeddings import get_embedding_model


def build_semantic_nodes(
    documents: List[Document],
) -> List[BaseNode]:
    embed_model = get_embedding_model()

    splitter = SemanticSplitterNodeParser(
        embed_model=embed_model,
        buffer_size=1,
        breakpoint_percentile_threshold=95,
    )

    nodes = splitter.get_nodes_from_documents(documents)

    return nodes