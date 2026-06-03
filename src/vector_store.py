"""
Vector database management module.

Stores and retrieves embeddings
through ChromaDB.
"""

from pathlib import Path
from typing import List, Optional

import chromadb
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.schema import BaseNode
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.embeddings import get_embedding_model


COLLECTION_NAME = "university_material"


def create_or_load_index(
    nodes: Optional[List[BaseNode]],
    persist_dir: str,
    rebuild: bool = False,
) -> VectorStoreIndex:
    """
    Creates or loads a ChromaDB-backed vector index.
    If rebuild=True, the previous collection is deleted and a new index
    is created from the provided nodes.
    If rebuild=False, the existing persisted ChromaDB collection is loaded.
    """

    Settings.embed_model = get_embedding_model()

    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)

    chroma_client = chromadb.PersistentClient(path=str(persist_path))

    if rebuild:
        try:
            chroma_client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

    vector_store = ChromaVectorStore(
        chroma_collection=chroma_collection
    )

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    if rebuild:
        if nodes is None:
            raise ValueError("Per ricostruire l'indice devi fornire i nodes.")

        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
        )

        return index

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
    )

    return index