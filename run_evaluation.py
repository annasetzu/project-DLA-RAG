"""
Evaluation pipeline.

Compares baseline and improved
RAG pipelines on a QA benchmark.
"""

import csv
from pathlib import Path

import pandas as pd

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

QUESTIONS_FILE = "evaluation/evaluation_questions.csv"
OUTPUT_FILE = "evaluation/evaluation_results.csv"


def keyword_score(answer: str, expected_keywords: str) -> float:
    """
    Computes a simple keyword-based evaluation score.
    The score is the fraction of expected keywords that appear
    in the generated answer. It provides a lightweight proxy
    for answer correctness.
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
    """
    Builds the baseline RAG index.
    Documents are chunked using fixed-size chunking and stored
    in a ChromaDB vector index.
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


def build_improved_index():
    """
    Builds the improved RAG index.
    Documents are chunked using semantic chunking and stored
    in a ChromaDB vector index.
    """
    documents = load_documents_from_folder(DATA_FOLDER)

    nodes = build_semantic_nodes(documents)

    index = create_or_load_index(
        nodes=nodes,
        persist_dir=IMPROVED_DB_PATH,
        rebuild=True,
    )

    return index


def evaluate_pipeline(index, question: str, expected_keywords: str, top_k: int):
    """
    Evaluates a single RAG pipeline on one question.
    The function retrieves relevant chunks, generates an answer,
    computes the keyword score, and returns both the answer
    and retrieved context.
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
    """
    Runs the full evaluation process.
    It builds both baseline and improved indexes, evaluates them
    on all questions, saves the results to CSV, and prints average scores.
    """
    print("Building baseline RAG pipeline...")
    base_index = build_base_index()

    print("Building improved RAG pipeline...")
    improved_index = build_improved_index()

    questions = pd.read_csv(QUESTIONS_FILE)

    results = []

    for _, row in questions.iterrows():
        question = row["question"]
        expected_keywords = row["expected_keywords"]
        expected_answer = row["expected_answer"]

        print(f"\nEvaluating question: {question}")

        base_result = evaluate_pipeline(
            index=base_index,
            question=question,
            expected_keywords=expected_keywords,
            top_k=BASE_TOP_K,
        )

        improved_result = evaluate_pipeline(
            index=improved_index,
            question=question,
            expected_keywords=expected_keywords,
            top_k=IMPROVED_TOP_K,
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

    print("\nEvaluation completed.")
    print(f"Results saved to: {OUTPUT_FILE}")

    print("\nAverage Score RAG base:")
    print(results_df["base_score"].mean())

    print("\nAverage Score RAG improved:")
    print(results_df["improved_score"].mean())


if __name__ == "__main__":
    main()