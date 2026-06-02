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
    Crea oppure carica un indice vettoriale usando ChromaDB.

    rebuild=True:
        cancella la collection precedente e crea un nuovo indice dai chunk.

    rebuild=False:
        carica la collection ChromaDB già esistente.
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