from src.ingestion import load_documents_from_folder
from src.chunking import build_nodes
from src.vector_store import create_or_load_index
from src.retrieval import retrieve_context
from src.generation import generate_answer


DATA_FOLDER = "data/raw"
PERSIST_DIR = "data/processed/chroma_db"


def build_rag_index():
    print("Caricamento documenti...")

    documents = load_documents_from_folder(DATA_FOLDER)

    print(f"Documenti caricati: {len(documents)}")

    print("Creazione dei chunk...")

    nodes = build_nodes(
        documents=documents,
        chunk_size=512,
        chunk_overlap=100,
    )

    print(f"Chunk creati: {len(nodes)}")

    print("Creazione indice vettoriale...")

    index = create_or_load_index(
        nodes=nodes,
        persist_dir=PERSIST_DIR,
        rebuild=True,
    )

    print("Indice creato correttamente.")

    return index


def ask_question(index):
    while True:
        question = input("\nInserisci una domanda, oppure scrivi 'exit': ")

        if question.lower() == "exit":
            break

        retrieved_nodes = retrieve_context(
            index=index,
            question=question,
            top_k=3,
        )

        answer = generate_answer(
            question=question,
            retrieved_nodes=retrieved_nodes,
        )

        print("\nRISPOSTA:")
        print(answer)

        print("\nCHUNK RECUPERATI:")
        for i, node in enumerate(retrieved_nodes, start=1):
            print(f"\n--- Chunk {i} ---")
            print(node.get_content()[:800])


def main():
    index = build_rag_index()
    ask_question(index)


if __name__ == "__main__":
    main()