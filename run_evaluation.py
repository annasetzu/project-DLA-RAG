import csv
from pathlib import Path

import pandas as pd

from src.ingestion import load_documents_from_folder
from src.chunking import build_nodes
from src.chunking_semantic import build_semantic_nodes
from src.vector_store import create_or_load_index
from src.retrieval import retrieve_context
from src.generation import generate_answer


DATA_FOLDER = "data/raw"

BASE_PERSIST_DIR = "data/processed/chroma_db_base"
IMPROVED_PERSIST_DIR = "data/processed/chroma_db_improved"

QUESTIONS_FILE = "evaluation_questions.csv"
OUTPUT_FILE = "evaluation_results.csv"


def keyword_score(answer: str, expected_keywords: str) -> float:
    """
    Calcola una metrica semplice:
    quante keyword attese compaiono nella risposta generata.
    """

    keywords = [
        keyword.strip().lower()
        for keyword in expected_keywords.split(",")
        if keyword.strip()
    ]

    if not keywords:
        return 0.0

    answer_lower = answer.lower()

    matched = 0

    for keyword in keywords:
        if keyword in answer_lower:
            matched += 1

    return matched / len(keywords)


def build_base_index():
    documents = load_documents_from_folder(DATA_FOLDER)

    nodes = build_nodes(
        documents=documents,
        chunk_size=350,
        chunk_overlap=70,
    )

    index = create_or_load_index(
        nodes=nodes,
        persist_dir=BASE_PERSIST_DIR,
        rebuild=True,
    )

    return index


def build_improved_index():
    documents = load_documents_from_folder(DATA_FOLDER)

    nodes = build_semantic_nodes(documents)

    index = create_or_load_index(
        nodes=nodes,
        persist_dir=IMPROVED_PERSIST_DIR,
        rebuild=True,
    )

    return index


def evaluate_pipeline(index, question: str, expected_keywords: str, top_k: int):
    retrieved_nodes = retrieve_context(
        index=index,
        question=question,
        top_k=top_k,
    )

    answer = generate_answer(
        question=question,
        retrieved_nodes=retrieved_nodes,
    )

    score = keyword_score(
        answer=answer,
        expected_keywords=expected_keywords,
    )

    retrieved_text = "\n\n".join(
        [node.get_content()[:500] for node in retrieved_nodes]
    )

    return {
        "answer": answer,
        "score": score,
        "retrieved_chunks": retrieved_text,
    }


def main():
    print("Creazione indice RAG base...")
    base_index = build_base_index()

    print("Creazione indice RAG migliorata...")
    improved_index = build_improved_index()

    questions = pd.read_csv(QUESTIONS_FILE)

    results = []

    for _, row in questions.iterrows():
        question = row["question"]
        expected_keywords = row["expected_keywords"]
        expected_answer = row["expected_answer"]

        print(f"\nValutazione domanda: {question}")

        base_result = evaluate_pipeline(
            index=base_index,
            question=question,
            expected_keywords=expected_keywords,
            top_k=3,
        )

        improved_result = evaluate_pipeline(
            index=improved_index,
            question=question,
            expected_keywords=expected_keywords,
            top_k=2,
        )

        results.append(
            {
                "question": question,
                "expected_answer": expected_answer,
                "expected_keywords": expected_keywords,

                "base_answer": base_result["answer"],
                "base_score": base_result["score"],
                "base_retrieved_chunks": base_result["retrieved_chunks"],

                "improved_answer": improved_result["answer"],
                "improved_score": improved_result["score"],
                "improved_retrieved_chunks": improved_result["retrieved_chunks"],
            }
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False)

    print("\nValutazione completata.")
    print(f"Risultati salvati in: {OUTPUT_FILE}")

    print("\nScore medio RAG base:")
    print(results_df["base_score"].mean())

    print("\nScore medio RAG migliorata:")
    print(results_df["improved_score"].mean())


if __name__ == "__main__":
    main()