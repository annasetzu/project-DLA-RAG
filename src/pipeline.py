"""
Pipeline utilities.

Contains reusable functions for
building and querying RAG pipelines.
"""

from llama_index.core import VectorStoreIndex

from src.ingestion import load_documents_from_folder
from src.chunking import build_nodes
from src.chunking_semantic import build_semantic_nodes
from src.vector_store import create_or_load_index
from src.retrieval import retrieve_context
from src.generation import generate_answer

from src.config import (
    BASE_CHUNK_SIZE,
    BASE_CHUNK_OVERLAP,
    BASE_TOP_K,
    IMPROVED_TOP_K,
    BASE_DB_PATH,
    IMPROVED_DB_PATH,
    DATA_FOLDER,
)


def build_base_pipeline() -> VectorStoreIndex:
    """
    Builds the baseline RAG pipeline.
    """

    documents = load_documents_from_folder(DATA_FOLDER)

    nodes = build_nodes(
        documents=documents,
        chunk_size=BASE_CHUNK_SIZE,
        chunk_overlap=BASE_CHUNK_OVERLAP,
    )

    index = create_or_load_index(
        nodes=nodes,
        persist_dir=BASE_DB_PATH,
        rebuild=True,
    )

    return index


def build_improved_pipeline() -> VectorStoreIndex:
    """
    Builds the improved RAG pipeline.
    """

    documents = load_documents_from_folder(DATA_FOLDER)

    nodes = build_semantic_nodes(documents)

    index = create_or_load_index(
        nodes=nodes,
        persist_dir=IMPROVED_DB_PATH,
        rebuild=True,
    )

    return index


def query_pipeline(
    index: VectorStoreIndex,
    question: str,
    top_k: int,
):
    """
    Queries a RAG pipeline and returns
    both answer and retrieved chunks.
    """

    retrieved_nodes = retrieve_context(
        index=index,
        question=question,
        top_k=top_k,
    )

    answer = generate_answer(
        question=question,
        retrieved_nodes=retrieved_nodes,
    )

    return answer, retrieved_nodes