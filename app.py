"""
Streamlit demo application.

Interactive interface for comparing
baseline and improved RAG pipelines.
"""

import streamlit as st

from src.pipeline import (
    build_base_pipeline,
    build_improved_pipeline,
    query_pipeline,
)

from src.config import (
    BASE_TOP_K,
    IMPROVED_TOP_K,
)


st.set_page_config(
    page_title="RAG University QA",
    layout="wide",
)

st.title("📚 RAG-based University QA System")

st.markdown("""
This application compares:

- Baseline RAG pipeline
- Improved RAG pipeline
- Side-by-side comparison

The system answers questions on university material
using Retrieval-Augmented Generation (RAG).
""")

# -----------------------------
# SIDEBAR
# -----------------------------

st.sidebar.title("⚙️ Configuration")

pipeline_type = st.sidebar.selectbox(
    "Select pipeline",
    [
        "Baseline RAG",
        "Improved RAG",
        "Compare both",
    ],
)

st.sidebar.markdown("---")

if pipeline_type == "Baseline RAG":
    st.sidebar.info(
        """
        Fixed-size chunking

        - SentenceSplitter
        - chunk_size = 350
        - overlap = 70
        - top_k = 3
        """
    )

elif pipeline_type == "Improved RAG":
    st.sidebar.info(
        """
        Semantic chunking

        - SemanticSplitter
        - semantic boundaries
        - top_k = 2
        """
    )

else:
    st.sidebar.info(
        """
        Side-by-side comparison

        Baseline:
        - fixed-size chunking
        - top_k = 3

        Improved:
        - semantic chunking
        - top_k = 2
        """
    )

# -----------------------------
# METRICS
# -----------------------------

col1, col2, col3 = st.columns(3)

col1.metric("Documents", "12")
col2.metric("Embedding Model", "BGE-small")
col3.metric("LLM", "Llama3")

st.markdown("---")

# -----------------------------
# PIPELINE LOADING
# -----------------------------

@st.cache_resource
def load_base_pipeline():
    """
    Loads the baseline pipeline once
    and caches it for the Streamlit session.
    """

    return build_base_pipeline()


@st.cache_resource
def load_improved_pipeline():
    """
    Loads the improved pipeline once
    and caches it for the Streamlit session.
    """

    return build_improved_pipeline()


with st.spinner("Loading RAG pipelines..."):

    if pipeline_type == "Baseline RAG":
        index = load_base_pipeline()
        top_k = BASE_TOP_K

    elif pipeline_type == "Improved RAG":
        index = load_improved_pipeline()
        top_k = IMPROVED_TOP_K

    else:
        base_index = load_base_pipeline()
        improved_index = load_improved_pipeline()

# -----------------------------
# QUESTION INPUT
# -----------------------------

question = st.text_input(
    "❓ Ask a question about the uploaded university material:"
)

# -----------------------------
# QUERY
# -----------------------------

if st.button("Generate Answer"):

    if not question.strip():
        st.warning("Please insert a question.")
        st.stop()

    if pipeline_type == "Compare both":

        with st.spinner(
            "Running retrieval and generation on both pipelines..."
        ):

            base_answer, base_nodes = query_pipeline(
                index=base_index,
                question=question,
                top_k=BASE_TOP_K,
            )

            improved_answer, improved_nodes = query_pipeline(
                index=improved_index,
                question=question,
                top_k=IMPROVED_TOP_K,
            )

        st.success("Answers generated successfully.")

        col_base, col_improved = st.columns(2)

        with col_base:
            st.subheader("Baseline RAG")
            st.write(base_answer)

            st.markdown("#### Retrieved Chunks")

            for i, node in enumerate(base_nodes, start=1):
                with st.expander(f"Baseline Chunk {i}"):
                    st.markdown(
                        f"**Similarity Score:** `{round(node.score, 4)}`"
                    )
                    st.write(node.get_content())

        with col_improved:
            st.subheader("Improved RAG")
            st.write(improved_answer)

            st.markdown("#### Retrieved Chunks")

            for i, node in enumerate(improved_nodes, start=1):
                with st.expander(f"Improved Chunk {i}"):
                    st.markdown(
                        f"**Similarity Score:** `{round(node.score, 4)}`"
                    )
                    st.write(node.get_content())

    else:

        with st.spinner(
            "Running retrieval and generating answer..."
        ):

            answer, retrieved_nodes = query_pipeline(
                index=index,
                question=question,
                top_k=top_k,
            )

        st.success("Answer generated successfully.")

        st.subheader("🧠 Generated Answer")
        st.write(answer)

        st.subheader("📄 Retrieved Chunks")

        for i, node in enumerate(retrieved_nodes, start=1):
            with st.expander(f"Chunk {i}"):
                st.markdown(
                    f"**Similarity Score:** `{round(node.score, 4)}`"
                )
                st.write(node.get_content())

# -----------------------------
# FOOTER
# -----------------------------

st.markdown("---")

st.markdown("""
### Project Information

Deep Learning Applications Project  
University RAG Question Answering System

Technologies:

- LlamaIndex
- ChromaDB
- Ollama
- Streamlit
- HuggingFace Embeddings
""")