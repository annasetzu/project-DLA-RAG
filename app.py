import streamlit as st

from src.ingestion import load_documents_from_folder
from src.chunking import build_nodes
from src.vector_store import create_or_load_index
from src.retrieval import retrieve_context
from src.generation import generate_answer


DATA_FOLDER = "data/raw"
PERSIST_DIR = "data/processed/chroma_db"


st.set_page_config(
    page_title="University RAG QA",
    page_icon="📚",
    layout="wide",
)

st.title("📚 RAG-based Question Answering su Materiale Universitario")

st.write(
    "Sistema RAG per porre domande su slide, dispense e appunti universitari."
)

with st.sidebar:
    st.header("Configurazione")
    chunk_size = st.number_input("Chunk size", min_value=100, max_value=2000, value=512, step=50)
    chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=500, value=100, step=25)
    top_k = st.number_input("Top-k retrieval", min_value=1, max_value=10, value=3, step=1)

    build_index = st.button("Crea/Aggiorna indice")

if build_index:
    with st.spinner("Caricamento documenti e creazione indice..."):
        documents = load_documents_from_folder(DATA_FOLDER)
        nodes = build_nodes(
            documents=documents,
            chunk_size=int(chunk_size),
            chunk_overlap=int(chunk_overlap),
        )
        create_or_load_index(
            nodes=nodes,
            persist_dir=PERSIST_DIR,
            rebuild=True,
        )
    st.success("Indice creato correttamente.")

question = st.text_input("Inserisci una domanda sui documenti:")

if question:
    with st.spinner("Recupero del contesto e generazione della risposta..."):
        index = create_or_load_index(
            nodes=None,
            persist_dir=PERSIST_DIR,
            rebuild=False,
        )
        retrieved_nodes = retrieve_context(index=index, question=question, top_k=int(top_k))
        answer = generate_answer(question=question, retrieved_nodes=retrieved_nodes)

    st.subheader("Risposta")
    st.write(answer)

    st.subheader("Chunk recuperati")
    for i, node in enumerate(retrieved_nodes, start=1):
        with st.expander(f"Chunk {i}"):
            st.write(node.get_content())
