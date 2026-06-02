import os
from typing import List

from dotenv import load_dotenv
from llama_index.core.schema import NodeWithScore
from llama_index.llms.ollama import Ollama


SYSTEM_PROMPT = """
Sei un assistente per il question answering su materiale universitario.

Regole:
- Rispondi solo usando il contesto fornito.
- Se la risposta non è presente nel contesto, dillo chiaramente.
- Non inventare informazioni.
- Rispondi in italiano.
"""


def build_prompt(question: str, retrieved_nodes: List[NodeWithScore]) -> str:
    """Build the final prompt using the retrieved context."""
    context_parts = []

    for i, node in enumerate(retrieved_nodes, start=1):
        source = node.metadata.get("file_name", "documento sconosciuto")
        content = node.get_content()
        context_parts.append(f"[Chunk {i} - Fonte: {source}]\n{content}")

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


def generate_answer(question: str, retrieved_nodes: List[NodeWithScore]) -> str:
    """Generate an answer using an LLM and the retrieved context."""
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY non trovata. Inseriscila nel file .env."
        )

    llm = Ollama(model="llama3", request_timeout=120.0)
    prompt = build_prompt(question, retrieved_nodes)

    response = llm.complete(prompt)
    return str(response)
