"""
Baseline chunking strategy.

Uses fixed-size chunking with overlap
through SentenceSplitter.
"""

from typing import List

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode


def build_nodes(
    documents: List[Document],
    chunk_size: int = 350,
    chunk_overlap: int = 70,
) -> List[BaseNode]:
    """
    Splits documents into fixed-size text chunks.
    This represents the baseline chunking strategy.
    Each chunk has a predefined size and an overlap with the previous one
    to preserve part of the local context.
    """
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    nodes = splitter.get_nodes_from_documents(documents)

    return nodes