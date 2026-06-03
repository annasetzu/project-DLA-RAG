"""
Embedding generation module.

Uses HuggingFace embedding models
to encode text chunks.
"""

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from src.config import EMBED_MODEL


def get_embedding_model():
    """
    Initializes the local embedding model.
    The embedding model converts each text chunk into a vector
    representation used for semantic search in the vector database.
    """

    embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL
    )

    return embed_model