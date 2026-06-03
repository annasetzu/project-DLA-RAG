"""
Baseline RAG pipeline.

Uses fixed-size chunking and
standard retrieval configuration.
"""

from src.pipeline import (
    build_base_pipeline,
    query_pipeline,
)

from src.config import BASE_TOP_K


def main():
    print("\n==============================")
    print("BASELINE RAG PIPELINE")
    print("==============================\n")

    print("Building baseline pipeline...")
    index = build_base_pipeline()

    print("Baseline pipeline ready.")

    while True:
        question = input(
            "\nInserisci una domanda, oppure scrivi 'exit': "
        )

        if question.lower() == "exit":
            break

        answer, retrieved_nodes = query_pipeline(
            index=index,
            question=question,
            top_k=BASE_TOP_K,
        )

        print("\nRISPOSTA:\n")
        print(answer)

        print("\nCHUNK RECUPERATI:")

        for i, node in enumerate(retrieved_nodes, start=1):
            print(f"\n--- Chunk {i} ---")
            print(f"Similarity Score: {round(node.score, 4)}")
            print(node.get_content()[:800])


if __name__ == "__main__":
    main()