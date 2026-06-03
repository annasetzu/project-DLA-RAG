"""
Answer generation module.

Uses a local LLM through Ollama
to generate grounded answers.
"""

from typing import List

from llama_index.core.schema import NodeWithScore
from llama_index.llms.ollama import Ollama
from src.config import LLM_MODEL


SYSTEM_PROMPT = """
Sei un assistente per il question answering su materiale universitario.

Regole obbligatorie:
- Rispondi sempre in italiano, anche se la domanda è in inglese.
- Usa solo il contesto fornito.
- Se la risposta non è presente nel contesto, dillo chiaramente.
- Non inventare informazioni.
- Cita, quando possibile, i chunk o le fonti usate.
- Formula una risposta chiara, sintetica e adatta a uno studente universitario.
"""


def build_prompt(question: str, retrieved_nodes: List[NodeWithScore]) -> str:
    """
    Builds the final prompt for the language model.
    The prompt contains the system instructions, the retrieved context,
    and the user question. This helps the LLM generate an answer grounded
    in the retrieved documents.
    """

    context_parts = []

    for i, node in enumerate(retrieved_nodes, start=1):
        source = node.metadata.get("file_name", "documento sconosciuto")
        content = node.get_content()

        context_parts.append(
            f"[Chunk {i} - Fonte: {source}]\n{content}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
{SYSTEM_PROMPT}

Contesto recuperato:
{context}

Domanda:
{question}

Risposta:
"""

    return prompt


def generate_answer(
    question: str,
    retrieved_nodes: List[NodeWithScore],
) -> str:
    """
    Generates an answer using a local LLM.
    The function sends the constructed prompt to Ollama and returns
    the generated response. The model is instructed to answer only
    using the retrieved context.
    """

    llm = Ollama(
        model=LLM_MODEL,
        request_timeout=120.0,
    )

    prompt = build_prompt(
        question=question,
        retrieved_nodes=retrieved_nodes,
    )

    response = llm.complete(prompt)

    return str(response)