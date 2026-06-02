from dataclasses import dataclass
from typing import List, Dict

import pandas as pd


@dataclass
class EvaluationExample:
    question: str
    expected_answer: str


def exact_keyword_score(answer: str, expected_answer: str) -> float:
    """
    Simple baseline evaluation metric.

    It checks how many expected keywords are present in the generated answer.
    This is intentionally simple and can be improved later.
    """
    expected_keywords = set(expected_answer.lower().split())
    answer_words = set(answer.lower().split())

    if not expected_keywords:
        return 0.0

    overlap = expected_keywords.intersection(answer_words)
    return len(overlap) / len(expected_keywords)


def evaluate_answers(results: List[Dict]) -> pd.DataFrame:
    """
    Evaluate generated answers.

    Expected input format:
    [
        {
            "question": "...",
            "expected_answer": "...",
            "generated_answer": "...",
            "retrieved_chunks": [...]
        }
    ]
    """
    rows = []

    for item in results:
        score = exact_keyword_score(
            answer=item["generated_answer"],
            expected_answer=item["expected_answer"],
        )

        rows.append(
            {
                "question": item["question"],
                "expected_answer": item["expected_answer"],
                "generated_answer": item["generated_answer"],
                "keyword_score": score,
                "num_retrieved_chunks": len(item.get("retrieved_chunks", [])),
            }
        )

    return pd.DataFrame(rows)
