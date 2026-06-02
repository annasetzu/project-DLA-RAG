from typing import List

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode


def build_nodes(
    documents: List[Document],
    chunk_size: int = 350,
    chunk_overlap: int = 70,
) -> List[BaseNode]:
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    nodes = splitter.get_nodes_from_documents(documents)

    return nodes