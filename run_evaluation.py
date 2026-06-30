"""
Evaluation pipeline.

Compares baseline and improved
RAG pipelines on a QA benchmark.

The evaluation includes:
- keyword-based score for end-to-end answer quality;
- Hit@k for retrieval quality.
"""

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
    for end-to-end answer correctness.
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


def hit_at_k(retrieved_nodes, expected_document: str) -> int:
    """
    Computes Hit@k for retrieval evaluation.

    Hit@k is equal to 1 if at least one of the retrieved chunks
    comes from the expected document, otherwise it is 0.

    This metric evaluates the retrieval step independently
    from the final LLM-generated answer.
    """

    expected_document = expected_document.lower().strip()

    for node in retrieved_nodes:
        file_name = node.metadata.get("file_name", "").lower()

        if expected_document in file_name:
            return 1

    return 0


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


def evaluate_pipeline(
    index,
    question: str,
    expected_keywords: str,
    expected_document: str,
    top_k: int,
):
    """
    Evaluates a single RAG pipeline on one question.

    The function:
    - retrieves relevant chunks;
    - computes Hit@k on the retrieved chunks;
    - generates an answer using the retrieved context;
    - computes the keyword score on the generated answer.
    """

    retrieved_nodes = retrieve_context(
        index=index,
        question=question,
        top_k=top_k,
    )

    retrieval_hit = hit_at_k(
        retrieved_nodes=retrieved_nodes,
        expected_document=expected_document,
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

    retrieved_sources = ", ".join(
        [
            node.metadata.get("file_name", "unknown")
            for node in retrieved_nodes
        ]
    )

    return {
        "answer": answer,
        "keyword_score": score,
        "hit_at_k": retrieval_hit,
        "retrieved_chunks": retrieved_text,
        "retrieved_sources": retrieved_sources,
    }


def main():
    """
    Runs the full evaluation process.

    It builds both baseline and improved indexes, evaluates them
    on all questions, saves the results to CSV, and prints
    average keyword scores and Hit@k values.
    """

    print("Building baseline RAG pipeline...")
    base_index = build_base_index()

    print("Building improved RAG pipeline...")
    improved_index = build_improved_index()

    questions = pd.read_csv(QUESTIONS_FILE)

    required_columns = {
        "question",
        "expected_keywords",
        "expected_answer",
        "expected_document",
    }

    missing_columns = required_columns - set(questions.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns in {QUESTIONS_FILE}: {missing_columns}"
        )

    results = []

    for _, row in questions.iterrows():
        question = row["question"]
        expected_keywords = row["expected_keywords"]
        expected_answer = row["expected_answer"]
        expected_document = row["expected_document"]

        print(f"\nEvaluating question: {question}")

        base_result = evaluate_pipeline(
            index=base_index,
            question=question,
            expected_keywords=expected_keywords,
            expected_document=expected_document,
            top_k=BASE_TOP_K,
        )

        improved_result = evaluate_pipeline(
            index=improved_index,
            question=question,
            expected_keywords=expected_keywords,
            expected_document=expected_document,
            top_k=IMPROVED_TOP_K,
        )

        results.append(
            {
                "question": question,
                "expected_answer": expected_answer,
                "expected_keywords": expected_keywords,
                "expected_document": expected_document,

                "base_answer": base_result["answer"],
                "base_keyword_score": base_result["keyword_score"],
                "base_hit_at_k": base_result["hit_at_k"],
                "base_retrieved_sources": base_result["retrieved_sources"],
                "base_retrieved_chunks": base_result["retrieved_chunks"],

                "improved_answer": improved_result["answer"],
                "improved_keyword_score": improved_result["keyword_score"],
                "improved_hit_at_k": improved_result["hit_at_k"],
                "improved_retrieved_sources": improved_result["retrieved_sources"],
                "improved_retrieved_chunks": improved_result["retrieved_chunks"],
            }
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False)

    print("\nEvaluation completed.")
    print(f"Results saved to: {OUTPUT_FILE}")

    print("\nAverage Keyword Score RAG base:")
    print(results_df["base_keyword_score"].mean())

    print("\nAverage Keyword Score RAG improved:")
    print(results_df["improved_keyword_score"].mean())

    print("\nAverage Hit@k RAG base:")
    print(results_df["base_hit_at_k"].mean())

    print("\nAverage Hit@k RAG improved:")
    print(results_df["improved_hit_at_k"].mean())


if __name__ == "__main__":
    main()