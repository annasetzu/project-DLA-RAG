from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def get_embedding_model():
    """
    Modello di embedding locale gratuito.
    """

    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )

    return embed_model